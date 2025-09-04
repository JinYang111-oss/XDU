# ============== 网络（强化版，接口不变） ==============
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 基础组件 ----------------

class SE1D(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv1d(ch, ch // r, 1)
        self.fc2 = nn.Conv1d(ch // r, ch, 1)
    def forward(self, x):
        w = F.adaptive_avg_pool1d(x, 1)
        w = F.silu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=True, pdrop=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act   = nn.SiLU(inplace=True)
        self.down  = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        self.se = SE1D(out_ch) if use_se else nn.Identity()
        self.drop = nn.Dropout(pdrop) if pdrop > 0 else nn.Identity()
    def forward(self, x):
        idt = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        if self.down is not None:
            idt = self.down(idt)
        x = x + idt
        x = self.drop(x)
        x = self.act(x)
        return x

# ---------------- 现有两种编码器（保持不变） ----------------

class ResNet1DEncoder(nn.Module):
    def __init__(self, out_dim=20, use_se=True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True)
        )
        self.layer1 = nn.Sequential(ResBlock1D(64,  64,  stride=1, use_se=use_se),
                                    ResBlock1D(64,  64,  stride=1, use_se=use_se))
        self.layer2 = nn.Sequential(ResBlock1D(64,  128, stride=2, use_se=use_se),
                                    ResBlock1D(128, 128, stride=1, use_se=use_se))
        self.layer3 = nn.Sequential(ResBlock1D(128, 256, stride=2, use_se=use_se),
                                    ResBlock1D(256, 256, stride=1, use_se=use_se))
        self.layer4 = nn.Sequential(ResBlock1D(256, 256, stride=2, use_se=use_se),
                                    ResBlock1D(256, 256, stride=1, use_se=use_se))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(256, out_dim)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class LightConvEncoder(nn.Module):
    def __init__(self, out_dim=20):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3)
        self.bn3   = nn.BatchNorm1d(128)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(128, out_dim)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# ---------------- 新增最强主干：ResAtt1DEncoder ----------------
# 设计：卷积金字塔(降采样 ×4) -> 通道扩张 -> TransformerEncoder(多层) -> 注意力池化 -> FC
# 特点：结合局部时序模式（卷积）与长程依赖（自注意力），对EEG/THz这类长序列非常有效。

class PositionalEncoding1D(nn.Module):
    """正弦位置编码（Transformer常用），与序列长度自适应"""
    def __init__(self, d_model: int, max_len: int = 4000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)  # [max_len, d_model]
    def forward(self, x):  # x: [B, T, C]
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0)

class AttnPool1D(nn.Module):
    """学习式注意力池化：用可学习query对序列做注意力，得到全局向量"""
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):  # x: [B, T, C]
        B = x.size(0)
        q = self.query.expand(B, -1, -1)  # [B,1,C]
        y, _ = self.attn(q, self.ln(x), self.ln(x), need_weights=False)
        return y.squeeze(1)  # [B,C]

class ResAtt1DEncoder(nn.Module):
    """
    强化版编码器：卷积金字塔提取局部模式 + Transformer 建模长程依赖 + 注意力池化
    输出维度仍为 out_dim=20，接口与原版保持一致。
    """
    def __init__(self, out_dim=20, d_model=256, n_heads=8, n_layers=4, drop=0.1, use_se=True):
        super().__init__()
        # 卷积金字塔（4次降采样，提纯局部模式）
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64,  kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.SiLU(inplace=True),
            ResBlock1D(64, 64,  stride=1, use_se=use_se, pdrop=drop),

            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128), nn.SiLU(inplace=True),
            ResBlock1D(128,128, stride=1, use_se=use_se, pdrop=drop),

            nn.Conv1d(128,192, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(192), nn.SiLU(inplace=True),
            ResBlock1D(192,192, stride=1, use_se=use_se, pdrop=drop),

            nn.Conv1d(192,d_model, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(d_model), nn.SiLU(inplace=True),
            ResBlock1D(d_model,d_model, stride=1, use_se=use_se, pdrop=drop),
        )
        # 转为序列 [B,T,C]
        self.pe = PositionalEncoding1D(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=drop, batch_first=True, activation='gelu', norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = AttnPool1D(d_model, n_heads=n_heads)
        self.fc   = nn.Linear(d_model, out_dim)

    def forward(self, x):  # x: [B,1,T]
        x = self.stem(x)               # [B,C,T']
        x = x.transpose(1, 2)          # [B,T',C]
        x = self.pe(x)                 # 加位置
        x = self.transformer(x)        # [B,T',C]
        x = self.pool(x)               # [B,C]
        return self.fc(x)              # [B,out_dim]

# ---------------- PINN / 解码器（保持不变） ----------------

class ResFFBlock(nn.Module):
    def __init__(self, dim=384, mult=4, pdrop=0.2):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*mult),
            nn.SiLU(),
            nn.Dropout(pdrop),
            nn.Linear(dim*mult, dim),
        )
    def forward(self, x):
        return x + self.ff(self.ln(x))

class PINNResMLP(nn.Module):
    def __init__(self, in_dim=20, dim=256, depth=3, n_layers=1, alpha_mult=1, pdrop=0.2):
        super().__init__()
        self.L = n_layers
        self.A = int(alpha_mult)
        self.proj = nn.Linear(in_dim, dim)
        self.blocks = nn.Sequential(*[ResFFBlock(dim=dim, mult=4, pdrop=pdrop) for _ in range(depth)])
        # 输出 +1 为空气层 alpha
        self.head = nn.Linear(dim, self.L + self.L * self.A + 1)

    def forward(self, feat):
        x = self.proj(feat)
        x = self.blocks(x)
        z = self.head(x)
        d_raw      = z[:, :self.L]
        alpha_raw  = z[:, self.L: self.L + self.L * self.A]
        air_raw    = z[:, self.L + self.L * self.A : self.L + self.L * self.A + 1]
        return d_raw, alpha_raw, air_raw

class ResDeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.de1 = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3,
                                      output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.de2 = nn.ConvTranspose1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.up  = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=1, stride=stride, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_ch)
    def forward(self, x):
        idt = self.bn3(self.up(x))
        x = self.act(self.bn1(self.de1(x)))
        x = self.bn2(self.de2(x))
        x = self.act(x + idt)
        return x

class StrongDecoder(nn.Module):
    def __init__(self, in_dim=20, out_len=1500):
        super().__init__()
        self.init_len = 188
        self.fc = nn.Linear(in_dim, 256 * self.init_len)
        self.block1 = ResDeconvBlock(256, 128, stride=2)
        self.block2 = ResDeconvBlock(128, 64,  stride=2)
        self.block3 = ResDeconvBlock(64,  32,  stride=2)
        self.tail   = nn.ConvTranspose1d(32, 1, kernel_size=7, stride=1, padding=3)
        self.out_len = out_len
    def forward(self, feat):
        x = self.fc(feat).view(-1, 256, self.init_len)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.tail(x)
        return x[..., :self.out_len]

# ---------------- THzNet（接口与forward不变） ----------------

def _try_load_pretrained_encoder(module: nn.Module):
    """
    可选预训练加载（仅编码器）：若存在以下任意路径则按键匹配加载：
      1) 环境变量 THZ_PRETRAINED 指向的 .pth
      2) 相对路径 pretrained/thz_resatt_encoder.pth
    未找到则静默跳过，不影响正常训练。
    """
    paths = []
    env_p = os.environ.get("THZ_PRETRAINED", "").strip()
    if env_p:
        paths.append(env_p)
    paths.append(os.path.join("pretrained", "thz_resatt_encoder.pth"))

    for p in paths:
        if p and os.path.isfile(p):
            try:
                sd = torch.load(p, map_location="cpu")
                # 仅加载匹配键
                msd = module.state_dict()
                matched = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
                if matched:
                    msd.update(matched)
                    module.load_state_dict(msd)
                    print(f"[ResAtt1DEncoder] Loaded pretrained (matched {len(matched)} tensors) from: {p}")
                else:
                    print(f"[ResAtt1DEncoder] No matched keys in pretrained: {p}")
            except Exception as e:
                print(f"[ResAtt1DEncoder] Failed to load pretrained from {p}: {e}")
            break  # 只尝试第一个可用路径

class THzNet(nn.Module):
    def __init__(self, n_layers, alpha_mult=1, backbone='resatt'):
        super().__init__()
        self.L = n_layers
        self.A = int(alpha_mult)

        # 新增最强主干 'resatt'；其余保持原有选项不变
        if backbone == 'resatt':
            self.encoder = ResAtt1DEncoder(out_dim=20, d_model=256, n_heads=8, n_layers=4, drop=0.1, use_se=True)
            _try_load_pretrained_encoder(self.encoder)  # 可选预训练加载（安全无副作用）
        elif backbone == 'resnet':
            self.encoder = ResNet1DEncoder(out_dim=20)
        else:  # 'conv'
            self.encoder = LightConvEncoder(out_dim=20)

        self.pinn    = PINNResMLP(in_dim=20, dim=256, depth=3, n_layers=n_layers, alpha_mult=alpha_mult, pdrop=0.2)
        self.decoder = StrongDecoder(in_dim=20, out_len=1500)

    def forward(self, x):
        feat = self.encoder(x)
        d_pred_n, alpha_pred_n, air_pred_n = self.pinn(feat)  # 接口保持不变
        recon = self.decoder(feat)
        return d_pred_n, alpha_pred_n, air_pred_n, recon

