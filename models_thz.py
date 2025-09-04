# -*- coding: utf-8 -*-
import os, csv
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from models import THzNet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ============== 通用工具 ==============

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def save_fig_both(fig, base_path_no_ext: str, dpi=200, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(base_path_no_ext + ".png", dpi=dpi)
    fig.savefig(base_path_no_ext + ".pdf")
    plt.close(fig)

# ============== 数据读取（按行名，逐层取值 & 打印首值） ==============

def load_excel_data_v2(file_path: str, n_layers: int, show_progress: bool = True):

    df = pd.read_excel(file_path, header=None, dtype=str)
    name_col = df.iloc[:, 0].astype(str).str.strip()

    def find_row(label: str) -> int:
        idx = name_col[name_col == label].index
        if len(idx) == 0:
            raise ValueError(f"未找到行名：{label}")
        return int(idx[0])

    def to_float_np(arr) -> np.ndarray:
        return pd.to_numeric(arr, errors='coerce').values.astype(np.float32)

    # 样本数 N
    N = df.shape[1] - 1
    if N <= 0:
        raise ValueError("未检测到样本列（需要至少1列样本数据）")

    # 时间轴与波形
    r_time = find_row("时间(ps)")
    time_np = to_float_np(df.iloc[r_time+1:, 0].dropna())
    T = int(time_np.shape[0])
    if T <= 0:
        raise ValueError("时间向量长度为 0，请检查 Excel 的 时间(ps) 行下方第一列")
    signal_np = np.zeros((T, N), dtype=np.float32)
    if show_progress: print(">>> 读取波形 ...")
    for j in tqdm(range(N), desc="Loading signals", disable=not show_progress):
        col = to_float_np(df.iloc[r_time+1:r_time+1+T, 1 + j])
        if col.shape[0] < T:
            tmp = np.zeros((T,), dtype=np.float32)
            tmp[:col.shape[0]] = col
            col = tmp
        signal_np[:, j] = col

    # 空气层
    r_air_d     = find_row("空气层_厚度")
    r_air_alpha = find_row("空气层_alpha")
    air_d_np     = to_float_np(df.iloc[r_air_d, 1:1+N])
    air_alpha_np = to_float_np(df.iloc[r_air_alpha, 1:1+N])

    # 样本层：e_inf / tau / es / d / alpha
    e_inf_layers = []
    tau_layers   = []
    es_layers    = []
    d_layers     = []
    alpha_layers = []

    for i in range(1, n_layers+1):
        r_einf = find_row(f"层{i}_e_endless")
        r_tau  = find_row(f"层{i}_tau")
        r_es   = find_row(f"层{i}_es")
        r_d    = find_row(f"层{i}_厚度")
        r_a    = find_row(f"层{i}_alpha")

        e_inf_layers.append(to_float_np(df.iloc[r_einf, 1:1+N]))
        tau_layers.append(   to_float_np(df.iloc[r_tau,  1:1+N]))
        es_layers.append(    to_float_np(df.iloc[r_es,   1:1+N]))
        d_layers.append(     to_float_np(df.iloc[r_d,    1:1+N]))
        alpha_layers.append( to_float_np(df.iloc[r_a,    1:1+N]))

    e_inf_layers = np.stack(e_inf_layers, axis=0)  # [L,N]
    tau_layers   = np.stack(tau_layers,   axis=0)  # [L,N]
    es_layers    = np.stack(es_layers,    axis=0)  # [L,N]
    d_layers     = np.stack(d_layers,     axis=0)  # [L,N]
    alpha_layers = np.stack(alpha_layers, axis=0)  # [L,N]

    # 首值核对
    if show_progress:
        print(">>> 首值核对（样本#0）:")
        print(f"  时间(ps) 首个时间: {float(time_np[0]) if T>0 else 'NA'}")
        print(f"  空气层_厚度[0]={float(air_d_np[0])}, 空气层_alpha[0]={float(air_alpha_np[0])}")
        for i in range(1, n_layers+1):
            print(f"  层{i}_e_endless[0]={float(e_inf_layers[i-1,0])}  层{i}_tau[0]={float(tau_layers[i-1,0])}  层{i}_es[0]={float(es_layers[i-1,0])}")
            print(f"  层{i}_厚度[0]={float(d_layers[i-1,0])}  层{i}_alpha[0]={float(alpha_layers[i-1,0])}")

    return {
        "time_data": time_np,                   # [T]
        "signal_data": signal_np,              # [T,N]
        "air_d": air_d_np,                     # [N]
        "air_alpha": air_alpha_np,             # [N]
        "e_inf_layers": e_inf_layers,          # [L,N]
        "tau_layers":   tau_layers,            # [L,N]
        "es_layers":    es_layers,             # [L,N]
        "d_layers":     d_layers,              # [L,N]
        "alpha_layers": alpha_layers,          # [L,N]
    }

# ============== 数据集 ==============

class THzDataset(Dataset):
    def __init__(self, file_path, n_layers, alpha_mult: int = 1,
                 norm: dict | None = None, device=None, show_progress: bool = True):
        assert int(alpha_mult) == 1
        self.n_layers = int(n_layers)
        self.alpha_mult = 1
        self.data = load_excel_data_v2(file_path, self.n_layers, show_progress=show_progress)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # 时间与信号
        self.time    = torch.tensor(self.data['time_data'],    dtype=torch.float32, device=self.device)   # [T]
        self.signals = torch.tensor(self.data['signal_data'].T, dtype=torch.float32, device=self.device)  # [N,T]
        self.signals = torch.nan_to_num(self.signals, nan=0.0)

        # 空气层
        self.air_d     = torch.tensor(self.data['air_d'],     dtype=torch.float32, device=self.device)    # [N]
        self.air_alpha = torch.tensor(self.data['air_alpha'], dtype=torch.float32, device=self.device)    # [N]
        self.air_d     = torch.nan_to_num(self.air_d,     nan=0.0)
        self.air_alpha = torch.nan_to_num(self.air_alpha, nan=0.0)

        # 样本层介质常数（逐层）
        e_inf_np = self.data['e_inf_layers'].astype(np.float32)   # [L,N]
        tau_np   = self.data['tau_layers'].astype(np.float32)
        es_np    = self.data['es_layers'].astype(np.float32)
        self.e_inf_layers = torch.tensor(e_inf_np.T, dtype=torch.float32, device=self.device)  # [N,L]
        self.tau_layers   = torch.tensor(tau_np.T,   dtype=torch.float32, device=self.device)  # [N,L]
        self.es_layers    = torch.tensor(es_np.T,    dtype=torch.float32, device=self.device)  # [N,L]
        self.e_inf_layers = torch.nan_to_num(self.e_inf_layers, nan=1.0)
        self.tau_layers   = torch.nan_to_num(self.tau_layers,   nan=1e-12)
        self.es_layers    = torch.nan_to_num(self.es_layers,    nan=1.0)

        # 监督标签（样本层 d 与 alpha）
        d_np     = self.data['d_layers'].astype(np.float32)        # [L,N]
        alpha_np = self.data['alpha_layers'].astype(np.float32)    # [L,N]
        self.d_raw     = torch.tensor(d_np.T,     dtype=torch.float32, device=self.device)  # [N,L]
        self.alpha_raw = torch.tensor(alpha_np.T, dtype=torch.float32, device=self.device)  # [N,L]
        self.d_raw     = torch.nan_to_num(self.d_raw,     nan=0.0)
        self.alpha_raw = torch.nan_to_num(self.alpha_raw, nan=0.0)

        # 标准化（逐列） + 空气层 alpha 的独立标准化
        if norm is None:
            self.d_mean  = self.d_raw.mean(0, keepdim=True)                     # [1,L]
            self.d_std   = self.d_raw.std(0,  keepdim=True) + 1e-8              # [1,L]
            self.alpha_mean_flat = self.alpha_raw.mean(0, keepdim=True)         # [1,L]
            self.alpha_std_flat  = self.alpha_raw.std(0,  keepdim=True) + 1e-8  # [1,L]

            self.air_alpha_mean = self.air_alpha.mean().view(1)                 # [1]
            self.air_alpha_std  = self.air_alpha.std().view(1) + 1e-8           # [1]
        else:
            self.d_mean  = norm["d_mean"].view(1, -1).to(self.device)
            self.d_std   = norm["d_std"].view(1, -1).to(self.device)
            self.alpha_mean_flat = norm["alpha_mean"].view(1, -1).to(self.device)
            self.alpha_std_flat  = norm["alpha_std"].view(1, -1).to(self.device)

            # 空气层 alpha 的 mean/std
            self.air_alpha_mean = norm["air_alpha_mean"].view(1).to(self.device)
            self.air_alpha_std  = norm["air_alpha_std"].view(1).to(self.device)

        self.d     = (self.d_raw     - self.d_mean)  / self.d_std          # [N,L]
        self.alpha = (self.alpha_raw - self.alpha_mean_flat) / self.alpha_std_flat  # [N,L]
        self.air_alpha_n = (self.air_alpha - self.air_alpha_mean) / self.air_alpha_std  # [N]  # === NEW ===

    def __len__(self):
        return self.signals.size(0)

    def __getitem__(self, idx):
        return (self.signals[idx].unsqueeze(0), self.time,
                self.d[idx], self.alpha[idx], self.air_alpha_n[idx:idx+1], idx)  # air_alpha_n 保持形状 [1]

# ============== 可导物理模块 ==============

def complex_sqrt(z: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(z)

def debye_eps_layers(omega: torch.Tensor,
                     e_inf_layers: torch.Tensor, tau_layers: torch.Tensor, es_layers: torch.Tensor):
    if omega.dim() == 1:
        omega = omega.view(1, 1, -1)  # [1,1,F]
    elif omega.dim() == 2:
        omega = omega.unsqueeze(0)
    device = omega.device
    B, L = e_inf_layers.shape
    e_inf = e_inf_layers.view(B, L, 1).to(device)
    tau   = torch.clamp(tau_layers.view(B, L, 1).to(device), min=1e-12)
    e_s   = es_layers.view(B, L, 1).to(device)
    omega_c = omega.to(torch.complex64)
    eps = e_inf + (e_s - e_inf) / (1.0 + 1j * omega_c * tau)
    return eps.to(torch.complex64)  # [B,L,F]

def align_to_len(x: torch.Tensor, target_len: int = 1500) -> torch.Tensor:
    if x.dim() == 1:
        T = x.shape[0]
        return F.pad(x, (0, target_len - T)) if T < target_len else x[:target_len]
    elif x.dim() == 2:
        B, T = x.shape
        return F.pad(x, (0, target_len - T)) if T < target_len else x[:, :target_len]
    else:
        raise ValueError(f"Unsupported tensor shape {x.shape}")

def simulate_multilayer_time_torch_v2(
    time: torch.Tensor,
    d_layers_pos: torch.Tensor,      # [B, M]  (M = 1空气层 + n样本层)
    alpha_layers_pos: torch.Tensor,  # [B, M]
    e_inf_layers: torch.Tensor, tau_layers: torch.Tensor, es_layers: torch.Tensor,  # [B, n]（样本层）
    out_len: int,
    ref_signal: torch.Tensor | None = None,
    theta0_deg: float = 8.8,
    c: float = 3e8
):

    device = time.device
    B, M = d_layers_pos.shape  # M = 1(空气层) + n
    n_samples = B

    # 频域
    dt = torch.mean(torch.clamp(time[1:] - time[:-1], min=1e-12))
    fs = 1.0 / dt
    F = out_len // 2 + 1
    freqs = torch.linspace(0., fs / 2, F, device=device)
    omega = 2.0 * torch.pi * freqs  # [F]

    # 样本层复介电常数 & 折射率 [B, n, F]
    n_sample_layers = torch.ones((B, M-1, F), dtype=torch.complex64, device=device)  # 占位
    if M > 1:
        eps_layers = debye_eps_layers(omega, e_inf_layers, tau_layers, es_layers)    # [B, n, F]
        n_sample_layers = complex_sqrt(eps_layers)
        # 选主值分支（Im>=0）
        n_sample_layers = torch.where(n_sample_layers.imag < 0, torch.conj(n_sample_layers), n_sample_layers)

    # 组装 n 列表：n0(空气) | n_airlayer(=1) | n_sample_layers[0..n-1] | n_out(空气)
    n_air = torch.ones((B, F), dtype=torch.complex64, device=device)
    n_list = [n_air]  # 入射空气
    # 空气层（有厚度的介质，但折射率=1）
    n_list.append(n_air)
    # 样本层
    for k in range(M-1):
        n_list.append(n_sample_layers[:, k, :])  # [B,F]
    # 出射空气
    n_list.append(n_air)

    # 角度传播（用 Re(n) 做 Snell）
    theta_list = []
    theta0 = torch.full((B, F), float(theta0_deg) * torch.pi / 180.0, device=device)
    theta_list.append(theta0)
    for i in range(len(n_list)-1):
        n_i_real = n_list[i].real
        n_j_real = n_list[i+1].real
        sin_theta = (n_i_real / torch.clamp(n_j_real, min=1e-6)) * torch.sin(theta_list[-1])
        sin_theta = torch.clamp(sin_theta, -1.0, 1.0)
        theta_list.append(torch.asin(sin_theta))

    def r_coeff(n1, n2, th_i, th_t):
        return (n1 * torch.cos(th_i) - n2 * torch.cos(th_t)) / (n1 * torch.cos(th_i) + n2 * torch.cos(th_t))

    def t_coeff(n1, n2, th_i, th_t):
        return (2 * n1 * torch.cos(th_i)) / (n1 * torch.cos(th_i) + n2 * torch.cos(th_t))

    # 从底往上递推
    r_last = r_coeff(n_list[-2], n_list[-1], theta_list[-2], theta_list[-1])  # 最底层界面M↔M+1
    # 厚度/alpha 对应的是“有厚度的层”索引：t ∈ [1..M]，与界面 i↔i+1 的“下侧层”一致
    for m in reversed(range(0, M)):  # 对应界面 m ↔ m+1，且使用层 (m+1) 的 d/α/n/theta
        n_i, n_j = n_list[m], n_list[m+1]
        th_i, th_j = theta_list[m], theta_list[m+1]
        r_ij = r_coeff(n_i, n_j, th_i, th_j)
        t_ij = t_coeff(n_i, n_j, th_i, th_j)
        t_ji = t_coeff(n_j, n_i, th_j, th_i)
        r_ji = r_coeff(n_j, n_i, th_j, th_i)

        # 该界面下侧“有厚度层”的参数下标 = m （m=0 空气层；m>=1 样本层 #m）
        d_j = d_layers_pos[:, m].view(B, 1)         # [B,1]
        a_j = alpha_layers_pos[:, m].view(B, 1)     # [B,1]
        L_eff = d_j / torch.clamp(torch.cos(th_j), min=1e-6)  # [B,F]（广播）
        phi = (2 * torch.pi * freqs.view(1, -1) * n_j * L_eff) / c
        extr = torch.exp(-torch.clamp(a_j * L_eff, min=0.0))

        num = r_ij + (t_ij * t_ji * r_last * torch.exp(-2j * phi)) * extr
        den = 1.0 - r_ji * r_last * torch.exp(-2j * phi) * extr
        r_last = num / den

    H = r_last  # 频域反射系数
    if ref_signal is not None:
        Y_ref = torch.fft.rfft(ref_signal, n=out_len, dim=1)
        Y_total = Y_ref * H
    else:
        Y_total = H
    sim = torch.fft.irfft(Y_total, n=out_len, dim=1)
    return torch.nan_to_num(sim, nan=0.0, posinf=1e6, neginf=-1e6)


# ============== 辅助函数与安全 R² ==============

def mse(a, b):
    return torch.mean((a - b) ** 2)

def soft_argmax_1d(x: torch.Tensor, beta: float = 50.0, dim: int = -1):
    idxs = torch.arange(x.size(dim), device=x.device, dtype=x.dtype)
    w = torch.softmax(beta * x, dim=dim)
    return torch.sum(w * idxs, dim=dim)

def fractional_shift_linear(sig: torch.Tensor, shift: torch.Tensor):
    B, T = sig.shape
    base = torch.arange(T, device=sig.device).view(1, T).expand(B, -1).to(sig.dtype)
    left = torch.floor(base - shift.view(B, 1))
    right = left + 1.0
    wl = right - (base - shift.view(B, 1))
    wr = 1.0 - wl
    left_idx  = torch.clamp(left,  0, T - 1).long()
    right_idx = torch.clamp(right, 0, T - 1).long()
    return wl * sig.gather(1, left_idx) + wr * sig.gather(1, right_idx)

def soft_cut_and_shift_torch(time: torch.Tensor, amp: torch.Tensor,
                             left_need: int = 600, right_need: int = 899, temp: float = 50.0):
    B, T = amp.shape
    win_len = left_need + right_need + 1
    center = float(left_need)
    peak = soft_argmax_1d(amp.abs(), beta=temp, dim=1)
    shift = peak - center
    shifted = fractional_shift_linear(amp, shift)
    if T < win_len:
        shifted = F.pad(shifted, (0, win_len - T))
    return shifted[:, :win_len]

def r2_safe(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-3) -> float:
    diff = y_true - y_pred
    sse = torch.sum(diff * diff)
    y_center = y_true - y_true.mean()
    sst = torch.sum(y_center * y_center)
    sst = torch.clamp(sst, min=eps * y_true.numel())
    return (1.0 - sse / sst).item()

def softplus_inv(y: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    y = torch.clamp(y, min=1e-8)
    return torch.log(torch.expm1(beta * y)) / beta

# ============== 损失函数（逐层监督 + v2物理仿真 + 空气层α监督） ==============

def custom_loss_with_metrics(
    d_pred_n, d_true_n, alpha_pred_n, alpha_true_n, air_pred_n, air_true_n,   # === CHG: 新增空气层α
    recon, raw, time, dataset: THzDataset, batch_indices: torch.Tensor,
    lambda_phys: float = 1.0, beta_recon: float = 1e-3,
    w_d: float = 2.0, w_alpha: float = 1.0,
    use_phys: bool = True, device: str = 'cuda',
    sim_len:int = 3000, soft_tau: float = 50.0):

    raw1500 = align_to_len(raw, 1500)
    recon_seq = recon.squeeze(1)

    # 监督：逐层 + 空气层α
    loss_d        = mse(d_pred_n,     d_true_n)
    loss_alpha    = mse(alpha_pred_n, alpha_true_n)
    loss_air_alpha= mse(air_pred_n,   air_true_n)  # === NEW ===
    loss_recon    = mse(recon_seq,    raw1500)

    # 反标准化（样本层 + 空气层α）
    d_denorm       = d_pred_n     * dataset.d_std.to(device) + dataset.d_mean.to(device)                    # [B,L]
    alpha_denorm   = alpha_pred_n * dataset.alpha_std_flat.to(device) + dataset.alpha_mean_flat.to(device)  # [B,L]
    air_denorm     = air_pred_n * dataset.air_alpha_std.to(device) + dataset.air_alpha_mean.to(device)      # [B,1]

    d_phys_samples     = F.softplus(d_denorm)           # [B,L]
    alpha_phys_samples = F.softplus(alpha_denorm)       # [B,L]
    air_alpha_phys     = F.softplus(air_denorm)         # [B,1]  # === NEW ===

    if use_phys and lambda_phys > 0.0:
        idx = batch_indices
        # 空气层厚度固定（来自数据，但不参与监督），alpha 用网络预测值
        air_d     = F.softplus(dataset.air_d[idx].view(-1,1))         # [B,1]
        # air_alpha 来自预测（air_alpha_phys）
        d_layers_pos     = torch.cat([air_d,     d_phys_samples], dim=1)      # [B, 1+L]
        alpha_layers_pos = torch.cat([air_alpha_phys, alpha_phys_samples], dim=1)  # [B, 1+L]

        # 样本层介质常数（逐层）
        e_inf = dataset.e_inf_layers[idx]  # [B,L]
        tau   = dataset.tau_layers[idx]    # [B,L]
        e_s   = dataset.es_layers[idx]     # [B,L]

        sim_long = simulate_multilayer_time_torch_v2(
            time, d_layers_pos, alpha_layers_pos,
            e_inf, tau, e_s,
            out_len=sim_len, ref_signal=raw, theta0_deg=8.8
        )
        sim_cut  = soft_cut_and_shift_torch(time, sim_long, left_need=600, right_need=899, temp=soft_tau)
        loss_phys = mse(sim_cut, raw1500)
    else:
        loss_phys = torch.zeros((), device=device)

    # 总损失：alpha 项包含 样本层 + 空气层
    loss_total = w_d * loss_d + w_alpha * (loss_alpha + loss_air_alpha) + beta_recon * loss_recon + lambda_phys * loss_phys

    # r2_alpha 计算：将样本层 alpha 与空气层 alpha 合并
    r2_a = r2_safe(torch.cat([alpha_true_n, air_true_n], dim=1),
                   torch.cat([alpha_pred_n, air_pred_n], dim=1))

    metrics = {
        "loss_total": loss_total.item(),
        "loss_d": loss_d.item(),
        "loss_alpha": (loss_alpha + loss_air_alpha).item(),   # 统计里合并
        "loss_phys": loss_phys.item(),
        "loss_recon": loss_recon.item(),
        "r2_d": r2_safe(d_true_n, d_pred_n),
        "r2_alpha": r2_a,
    }
    return loss_total, metrics

# ============== 评估/绘图（保持你原有的 + 空气层alpha支持） ==============

def _np_r2(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    sse = np.sum((y_true - y_pred)**2)
    sst = np.sum((y_true - y_true.mean())**2) + 1e-12
    return 1.0 - sse/sst

def _plot_scatter(y_true, y_pred, title, out_base):
    fig = plt.figure(figsize=(4.2, 4.0))
    plt.scatter(y_true, y_pred, s=12, alpha=0.7)
    m = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(m, m, linestyle='--', linewidth=1.2)
    plt.xlabel("True"); plt.ylabel("Pred"); plt.title(title)
    save_fig_both(fig, out_base)

def _plot_residual(y_true, y_pred, title, out_base):
    res = y_pred - y_true
    fig = plt.figure(figsize=(4.8, 3.6))
    plt.axhline(0.0, color='k', linestyle='--', linewidth=1)
    plt.scatter(y_true, res, s=12, alpha=0.7)
    plt.xlabel("True"); plt.ylabel("Residual (Pred-True)"); plt.title(title)
    save_fig_both(fig, out_base)

def _plot_box(residuals_by_layer, labels, title, out_base):
    fig = plt.figure(figsize=(1.8*len(labels)+1.5, 4.0))
    plt.boxplot(residuals_by_layer, labels=labels, showfliers=True)
    plt.xlabel("Layer"); plt.ylabel("Residual"); plt.title(title)
    save_fig_both(fig, out_base)

def _plot_bar(values, labels, ylabel, title, out_base):
    fig = plt.figure(figsize=(1.8*len(labels)+1.5, 4.0))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels)
    plt.ylabel(ylabel); plt.title(title)
    save_fig_both(fig, out_base)

def _robust_range(a: np.ndarray) -> float:
    if a is None or a.size == 0:
        return 0.0
    lo, hi = np.nanpercentile(a, 1), np.nanpercentile(a, 99)
    return float(hi - lo + 1e-6)

def _normalize_by_p99(a: np.ndarray) -> np.ndarray:
    s = np.nanpercentile(np.abs(a), 99)
    if s <= 0 or np.isnan(s):
        s = 1.0
    return a / s

def _plot_three_signals(time, raw, recon, sim, title, out_base,
                        style: str = "stack", offset_value: float | None = None,
                        include_sim: bool = True):
    if not include_sim or sim is None:
        include_sim = False

    if style == "subplots":
        nrow = 3 if include_sim else 2
        fig, axes = plt.subplots(nrow, 1, figsize=(7.2, 5.0 if include_sim else 4.2), sharex=True)
        axes = np.atleast_1d(axes)
        axes[0].plot(time, raw);    axes[0].set_ylabel("Raw")
        axes[1].plot(time, recon);  axes[1].set_ylabel("Recon")
        if include_sim:
            axes[2].plot(time, sim); axes[2].set_ylabel("Physics")
        axes[-1].set_xlabel("Time"); fig.suptitle(title); fig.tight_layout()
        save_fig_both(fig, out_base)
        return

    if style == "normalize":
        raw_n   = _normalize_by_p99(raw)
        recon_n = _normalize_by_p99(recon)
        sim_n   = _normalize_by_p99(sim) if include_sim else None
        fig = plt.figure(figsize=(7.2, 3.8))
        plt.plot(time, raw_n,   label="Raw(1500)")
        plt.plot(time, recon_n, label="Decoder Recon")
        if include_sim:
            plt.plot(time, sim_n, label="Physics Sim (cut)")
        plt.xlabel("Time"); plt.ylabel("Norm Amp (÷p99)"); plt.title(title); plt.legend()
        save_fig_both(fig, out_base)
        return

    if offset_value is None:
        rngs = [_robust_range(raw), _robust_range(recon)]
        if include_sim:
            rngs.append(_robust_range(sim))
        offset_value = 1.2 * max(rngs)
        if not np.isfinite(offset_value) or offset_value <= 0:
            offset_value = 1.0

    fig = plt.figure(figsize=(7.2, 3.8))
    plt.axhline(0,           color='gray', linestyle='--', linewidth=0.8)
    plt.plot(time, raw,      label="Raw(1500)")
    plt.axhline(offset_value,   color='gray', linestyle='--', linewidth=0.8)
    plt.plot(time, recon + offset_value, label=f"Decoder Recon +{offset_value:.2g}")
    if include_sim:
        plt.axhline(2*offset_value, color='gray', linestyle='--', linewidth=0.8)
        plt.plot(time, sim + 2*offset_value, label=f"Physics Sim +{2*offset_value:.2g}")
        ylim_top = 2.0*offset_value + np.nanpercentile(sim, 99) + 0.05*offset_value
    else:
        ylim_top = offset_value + np.nanpercentile(recon, 99) + 0.05*offset_value

    ylim_bot = np.nanpercentile(raw, 1) - 0.1*offset_value
    plt.ylim([ylim_bot, ylim_top])

    plt.xlabel("Time"); plt.ylabel("Amplitude (stacked)")
    plt.title(title); plt.legend()
    save_fig_both(fig, out_base)

# ============== 评估 / 测试（含对比） ==============
def save_checkpoint(path, model: nn.Module, n_layers: int, alpha_mult: int, norm: dict, epoch: int):
    payload = {
        "model_state": model.state_dict(),
        "n_layers": n_layers,
        "alpha_mult": int(alpha_mult),
        "epoch": epoch,
        "alpha_mean": norm["alpha_mean"].cpu(),
        "alpha_std":  norm["alpha_std"].cpu(),
        "d_mean":     norm["d_mean"].cpu(),
        "d_std":      norm["d_std"].cpu(),
        # === NEW: 保存空气层 alpha 的均值方差 ===
        "air_alpha_mean": norm["air_alpha_mean"].cpu(),
        "air_alpha_std":  norm["air_alpha_std"].cpu(),
    }
    torch.save(payload, path)

def evaluate(model, loader, dataset_for_norm: THzDataset, device,
             lambda_phys=1.0, beta_recon=1e-3, use_phys=True,
             sim_len=3000, soft_tau=50.0, w_d=2.0, w_alpha=1.0):
    model.eval()
    agg = {"loss_total":0.0,"loss_d":0.0,"loss_alpha":0.0,"loss_phys":0.0,"loss_recon":0.0,"r2_d":0.0,"r2_alpha":0.0}
    n = 0
    for x, time, d_true_n, alpha_true_n, air_true_n, idx in loader:   # === CHG: 接收 air_true_n
        n += 1
        x = x.to(device); raw = x.squeeze(1)
        d_true_n = d_true_n.to(device); alpha_true_n = alpha_true_n.to(device); air_true_n = air_true_n.to(device)
        d_pred_n, alpha_pred_n, air_pred_n, recon = model(x)
        loss, m = custom_loss_with_metrics(
            d_pred_n, d_true_n, alpha_pred_n, alpha_true_n, air_pred_n, air_true_n,
            recon, raw, time, dataset_for_norm, idx,
            lambda_phys=lambda_phys, beta_recon=beta_recon, use_phys=use_phys,
            device=device, sim_len=sim_len, soft_tau=soft_tau,
            w_d=w_d, w_alpha=w_alpha
        )
        for k in agg: agg[k] += m[k]
    for k in agg: agg[k] /= max(n,1)
    return agg

def physics_refine_batch(
    time, raw1500, e_inf_layers, tau_layers, e_s_layers,
    air_d,                                  # [B,1]
    d_init_denorm, alpha_init_denorm, air_init_denorm,   # === CHG: 增加 air 初值（反标准化）
    steps:int=50, lr:float=0.05, lam_phys:float=1.0, lam_prior:float=0.1,
    sim_len:int=3000, soft_tau:float=50.0
):
    device = d_init_denorm.device
    B, L = d_init_denorm.shape

    # 优化变量：z_d, z_a, z_air（softplus^-1 以确保正性）
    z_d   = softplus_inv(d_init_denorm).detach().clone().requires_grad_(True)      # [B,L]
    z_a   = softplus_inv(alpha_init_denorm).detach().clone().requires_grad_(True)  # [B,L]
    z_air = softplus_inv(air_init_denorm).detach().clone().requires_grad_(True)    # [B,1]
    optimizer = torch.optim.Adam([z_d, z_a, z_air], lr=lr)
    loss_curve = []

    for _ in range(steps):
        optimizer.zero_grad()
        d_pos     = F.softplus(z_d)    # [B,L]
        alpha_pos = F.softplus(z_a)    # [B,L]
        air_pos   = F.softplus(z_air)  # [B,1]

        d_layers_pos     = torch.cat([F.softplus(air_d), d_pos], dim=1)           # [B,1+L]
        alpha_layers_pos = torch.cat([air_pos,           alpha_pos], dim=1)       # [B,1+L]

        sim_long = simulate_multilayer_time_torch_v2(
            time, d_layers_pos, alpha_layers_pos,
            e_inf_layers, tau_layers, e_s_layers,
            out_len=sim_len, ref_signal=raw1500, theta0_deg=8.8
        )
        sim_cut = soft_cut_and_shift_torch(time, sim_long, left_need=600, right_need=899, temp=soft_tau)

        loss_phys = mse(sim_cut, raw1500)
        loss_prior = mse(d_pos, d_init_denorm) + mse(alpha_pos, alpha_init_denorm) + mse(air_pos, air_init_denorm)  # === CHG：包含空气层先验
        loss = lam_phys * loss_phys + lam_prior * loss_prior
        loss.backward()
        optimizer.step()
        loss_curve.append(loss.detach().item())

    d_ref   = F.softplus(z_d).detach()
    a_ref   = F.softplus(z_a).detach()
    air_ref = F.softplus(z_air).detach()
    return d_ref, a_ref, air_ref, loss_curve

@torch.no_grad()
def test(file_path, ckpt_path, batch_size=32, device='cuda',
         save_csv='predictions.csv', save_sim: int = 1,
         lambda_phys=1.0, beta_recon=1e-3, use_phys=True,
         sim_len=3000, soft_tau=50.0, w_d: float = 2.0, w_alpha: float = 1.0,
         plot: bool = True, plot_dir: str = "figs", plot_samples: int = 12, seed: int = 42,
         plot_style: str = "stack", offset_value: float | None = None,
         compare: bool = False, compare_steps:int=50, compare_lr:float=0.05,
         compare_lam_phys:float=1.0, compare_lam_prior:float=0.1,
         compare_pdf:str="figs/phys_compare.pdf",
         train_loss_curve: list[float] | None = None, val_loss_curve: list[float] | None = None):

    ckpt = torch.load(ckpt_path, map_location=device)
    n_layers = ckpt["n_layers"]
    alpha_mult = int(ckpt.get("alpha_mult", 1))  # 兼容老的 ckpt
    norm = {
        "alpha_mean": ckpt["alpha_mean"].to(device),
        "alpha_std":  ckpt["alpha_std"].to(device),
        "d_mean":     ckpt["d_mean"].to(device),
        "d_std":      ckpt["d_std"].to(device),
        # === NEW ===
        "air_alpha_mean": ckpt["air_alpha_mean"].to(device),
        "air_alpha_std":  ckpt["air_alpha_std"].to(device),
    }

    dataset = THzDataset(file_path, n_layers, alpha_mult=alpha_mult, norm=norm, device=device, show_progress=True)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = THzNet(n_layers, alpha_mult=alpha_mult, backbone='resatt').to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    meters = {"loss_total":0.0,"loss_d":0.0,"loss_alpha":0.0,"loss_phys":0.0,"loss_recon":0.0,"r2_d":0.0,"r2_alpha":0.0}
    n_batches = 0

    # CSV 头：d1..dL + air_alpha + alpha{layer}
    alpha_cols = [f"alpha{i+1}" for i in range(n_layers)]
    header = ["index"] + [f"d{i+1}" for i in range(n_layers)] + ["air_alpha"] + alpha_cols
    rows = []

    d_true_all, d_pred_all = [], []
    a_true_all, a_pred_all = [], []
    air_true_all, air_pred_all = [], []  # === NEW ===

    # for 波形绘图
    raw1500_all, recon1500_all, sim1500_all = [], [], []
    time1500 = align_to_len(dataset.time, 1500).cpu().numpy()

    # 记录 Before/After 用于对比
    d_before_all, a_before_all, air_before_all = [], [], []   # === NEW ===
    d_after_all,  a_after_all,  air_after_all  = [], [], []

    for x, time, d_true_n, alpha_true_n, air_true_n, idx in loader:
        n_batches += 1
        x = x.to(device); raw = x.squeeze(1)
        d_true_n = d_true_n.to(device); alpha_true_n = alpha_true_n.to(device); air_true_n = air_true_n.to(device)

        d_pred_n, alpha_pred_n, air_pred_n, recon = model(x)
        loss, m = custom_loss_with_metrics(
            d_pred_n, d_true_n, alpha_pred_n, alpha_true_n, air_pred_n, air_true_n,
            recon, raw, time, dataset, idx,
            lambda_phys=lambda_phys, beta_recon=beta_recon, use_phys=use_phys,
            device=device, sim_len=sim_len, soft_tau=soft_tau,
            w_d=w_d, w_alpha=w_alpha
        )
        for k in meters: meters[k] += m[k]

        # 反标准化到物理空间
        d_denorm_true     = d_true_n     * dataset.d_std.to(device) + dataset.d_mean.to(device)                 # [B,L]
        alpha_denorm_true = alpha_true_n * dataset.alpha_std_flat.to(device) + dataset.alpha_mean_flat.to(device)  # [B,L]
        air_denorm_true   = air_true_n   * dataset.air_alpha_std.to(device) + dataset.air_alpha_mean.to(device)    # [B,1]

        d_denorm_pred     = d_pred_n     * dataset.d_std.to(device) + dataset.d_mean.to(device)                 # [B,L]
        alpha_denorm_pred = alpha_pred_n * dataset.alpha_std_flat.to(device) + dataset.alpha_mean_flat.to(device)  # [B,L]
        air_denorm_pred   = air_pred_n   * dataset.air_alpha_std.to(device) + dataset.air_alpha_mean.to(device)    # [B,1]

        # 保存 Before
        d_before_all.append(d_denorm_pred.detach().cpu().numpy())
        a_before_all.append(alpha_denorm_pred.detach().cpu().numpy())
        air_before_all.append(air_denorm_pred.detach().cpu().numpy())  # [B,1]

        if compare:
            # 推理期物理一致性微调（含空气层alpha）
            raw1500_batch = align_to_len(raw, 1500)
            e_inf = dataset.e_inf_layers[idx]; tau = dataset.tau_layers[idx]; e_s = dataset.es_layers[idx]
            air_d = F.softplus(dataset.air_d[idx].view(-1,1))
            torch.set_grad_enabled(True)
            d_ref, a_ref, air_ref, _ = physics_refine_batch(
                time, raw1500_batch, e_inf, tau, e_s,
                air_d,
                d_denorm_pred.detach(), alpha_denorm_pred.detach(), air_denorm_pred.detach(),
                steps=compare_steps, lr=compare_lr,
                lam_phys=compare_lam_phys, lam_prior=compare_lam_prior,
                sim_len=sim_len, soft_tau=soft_tau
            )
            torch.set_grad_enabled(False)
            d_after_all.append(d_ref.detach().cpu().numpy())
            a_after_all.append(a_ref.detach().cpu().numpy())
            air_after_all.append(air_ref.detach().cpu().numpy())

        # 汇总真值/预测（整体指标）
        d_true_all.append(d_denorm_true.detach().cpu().numpy())
        d_pred_all.append(d_denorm_pred.detach().cpu().numpy())
        a_true_all.append(alpha_denorm_true.detach().cpu().numpy())
        a_pred_all.append(alpha_denorm_pred.detach().cpu().numpy())
        air_true_all.append(air_denorm_true.detach().cpu().numpy())  # [B,1]
        air_pred_all.append(air_denorm_pred.detach().cpu().numpy())

        # 写 CSV
        for b, sid in enumerate(idx.tolist()):
            row = [sid] + d_denorm_pred[b].detach().cpu().numpy().tolist()
            row += [float(air_denorm_pred[b,0].detach().cpu().item())]  # air_alpha
            row += alpha_denorm_pred[b].detach().cpu().numpy().tolist()
            rows.append(row)

        # 1500点信号（原始/重建/仿真）
        raw1500  = align_to_len(raw, 1500)
        recon1500 = recon.squeeze(1)[..., :1500]
        if save_sim:
            # v2 物理仿真：空气层 α 用预测值
            d_phys_samples     = F.softplus(d_denorm_pred)         # [B,L]
            alpha_phys_samples = F.softplus(alpha_denorm_pred)     # [B,L]
            air_alpha_phys     = F.softplus(air_denorm_pred)       # [B,1]

            air_d = F.softplus(dataset.air_d[idx].view(-1, 1))     # [B,1]
            d_layers_pos     = torch.cat([air_d,          d_phys_samples], dim=1)     # [B, 1+L]
            alpha_layers_pos = torch.cat([air_alpha_phys, alpha_phys_samples], dim=1) # [B, 1+L]

            e_inf_layers = dataset.e_inf_layers[idx]  # [B,L]
            tau_layers   = dataset.tau_layers[idx]    # [B,L]
            e_s_layers   = dataset.es_layers[idx]     # [B,L]

            sim_long = simulate_multilayer_time_torch_v2(
                time, d_layers_pos, alpha_layers_pos,
                e_inf_layers, tau_layers, e_s_layers,
                out_len=sim_len, ref_signal=raw, theta0_deg=8.8
            )
            sim_cut = soft_cut_and_shift_torch(time, sim_long, left_need=600, right_need=899, temp=soft_tau)
        else:
            sim_cut = torch.zeros_like(raw1500)

        raw1500_all.append(raw1500.detach().cpu().numpy())
        recon1500_all.append(recon1500.detach().cpu().numpy())
        sim1500_all.append(sim_cut.detach().cpu().numpy())

    for k in meters: meters[k] /= max(n_batches,1)
    print("\n=== Test Metrics ===")
    print(f"Loss={meters['loss_total']:.6f} (d={meters['loss_d']:.6f}, alpha={meters['loss_alpha']:.6f}, "
          f"phys={meters['loss_phys']:.6f}, recon={meters['loss_recon']:.6f}) "
          f"R²_d={meters['r2_d']:.3f} R²_alpha(all)={meters['r2_alpha']:.3f}")

    # 保存预测结果CSV
    with open(save_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"[+] 保存预测到 {save_csv}")

    def percent_err(y, p):
        y = np.asarray(y); p = np.asarray(p)
        mask = np.abs(y) > 1e-8
        e = np.zeros_like(p, dtype=np.float64)
        e[mask] = (p[mask] - y[mask]) / y[mask] * 100.0
        e[~mask] = 0.0
        return e

    if compare:
        ensure_dir(os.path.dirname(compare_pdf) if os.path.dirname(compare_pdf) else ".")
        d_before = np.concatenate(d_before_all, axis=0)   # [N,L]
        a_before = np.concatenate(a_before_all, axis=0)   # [N,L]
        air_before = np.concatenate(air_before_all, axis=0).reshape(-1)  # [N]

        d_true   = np.concatenate(d_true_all, axis=0)     # [N,L]
        a_true   = np.concatenate(a_true_all, axis=0)     # [N,L]
        air_true = np.concatenate(air_true_all, axis=0).reshape(-1)      # [N]

        if len(d_after_all) > 0:
            d_after  = np.concatenate(d_after_all,  axis=0)  # [N,L]
            a_after  = np.concatenate(a_after_all,  axis=0)  # [N,L]
            air_after= np.concatenate(air_after_all, axis=0).reshape(-1)  # [N]
        else:
            d_after, a_after, air_after = d_before.copy(), a_before.copy(), air_before.copy()

        # 展平用于整体散点
        td_b, pd_b = d_true.reshape(-1), d_before.reshape(-1)
        ta_b, pa_b = a_true.reshape(-1), a_before.reshape(-1)
        td_a, pd_a = d_true.reshape(-1), d_after.reshape(-1)
        ta_a, pa_a = a_true.reshape(-1), a_after.reshape(-1)

        def r2_np(y, p):
            sse = np.sum((y - p)**2); sst = np.sum((y - np.mean(y))**2) + 1e-12
            return 1.0 - sse/sst
        def mape_np(y, p):
            y = np.asarray(y); p = np.asarray(p)
            mask = np.abs(y) > 1e-8
            if not np.any(mask): return np.nan
            return np.mean(np.abs((p[mask] - y[mask]) / y[mask])) * 100.0

        R2_b_d, R2_b_a = r2_np(td_b, pd_b), r2_np(ta_b, pa_b)
        R2_a_d, R2_a_a = r2_np(td_a, pd_a), r2_np(ta_a, pa_a)
        MAPE_b_d, MAPE_b_a = mape_np(td_b, pd_b), mape_np(ta_b, pa_b)
        MAPE_a_d, MAPE_a_a = mape_np(td_a, pd_a), mape_np(ta_a, pa_a)

        with PdfPages(compare_pdf) as pdf:
            # 1) 训练 & 验证损失曲线
            if train_loss_curve is not None and val_loss_curve is not None:
                epochs = range(1, len(train_loss_curve) + 1)
                fig = plt.figure(figsize=(6.4, 4.8))
                plt.plot(epochs, train_loss_curve, label='Train Loss')
                plt.plot(epochs, val_loss_curve, label='Val Loss')
                plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training and Validation Loss')
                plt.legend(); plt.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.splitext(compare_pdf)[0] + "_loss_curve.png", dpi=200)
                pdf.savefig(fig); plt.close(fig)

            # 2) 预测值 vs 真值 散点（d 和 alpha整体）
            fig, axes = plt.subplots(1, 2, figsize=(12,5))
            axes[0].scatter(td_b, pd_b, s=12, alpha=0.5, label='Before')
            axes[0].scatter(td_a, pd_a, s=12, alpha=0.7, label='After')
            lim_min = min(td_b.min(), pd_b.min(), td_a.min(), pd_a.min())
            lim_max = max(td_b.max(), pd_b.max(), td_a.max(), pd_a.max())
            axes[0].plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1.2)
            axes[0].set_title('Predicted vs True (d)')
            axes[0].set_xlabel('True'); axes[0].set_ylabel('Predicted'); axes[0].legend()

            axes[1].scatter(ta_b, pa_b, s=12, alpha=0.5, label='Before')
            axes[1].scatter(ta_a, pa_a, s=12, alpha=0.7, label='After')
            lim_min = min(ta_b.min(), pa_b.min(), ta_a.min(), pa_a.min())
            lim_max = max(ta_b.max(), pa_b.max(), ta_a.max(), pa_a.max())
            axes[1].plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1.2)
            axes[1].set_title('Predicted vs True (alpha - all samples)')
            axes[1].set_xlabel('True'); axes[1].set_ylabel('Predicted'); axes[1].legend()
            fig.tight_layout()
            fig.savefig(os.path.splitext(compare_pdf)[0] + "_scatter_all.png", dpi=200)
            pdf.savefig(fig); plt.close(fig)

            # 3) —— d：每层单独直方图（每张图一页）——
            bins = 40
            for i in range(n_layers):
                d_err_b = percent_err(d_true[:, i], d_before[:, i])
                d_err_a = percent_err(d_true[:, i], d_after[:,  i])
                fig = plt.figure(figsize=(6.0,5.0))
                plt.hist(d_err_b, bins=bins, alpha=0.5, label='Before')
                plt.hist(d_err_a, bins=bins, alpha=0.5, label='After')
                plt.title(f'Percent Error Histogram (d{i+1})')
                plt.xlabel('Percent Error (%)'); plt.ylabel('Count'); plt.legend()
                fig.tight_layout()
                fig.savefig(os.path.splitext(compare_pdf)[0] + f"_hist_d{i+1}.png", dpi=200)
                pdf.savefig(fig); plt.close(fig)

            # 4) —— 样本层 alpha：每层×每分量 单独直方图（每张图一页）——
            A = alpha_mult
            a_true_3d = a_true.reshape(-1, n_layers, A)  # [N,L,A]
            a_bef_3d  = a_before.reshape(-1, n_layers, A)
            a_aft_3d  = a_after.reshape(-1, n_layers, A)

            for i in range(n_layers):
                for k in range(A):
                    a_err_b = percent_err(a_true_3d[:, i, k], a_bef_3d[:, i, k])
                    a_err_a = percent_err(a_true_3d[:, i, k], a_aft_3d[:, i, k])
                    fig = plt.figure(figsize=(6.0, 5.0))
                    plt.hist(a_err_b, bins=bins, alpha=0.5, label='Before')
                    plt.hist(a_err_a, bins=bins, alpha=0.5, label='After')
                    plt.title(f'Percent Error Histogram (alpha L{i + 1} C{k + 1})')
                    plt.xlabel('Percent Error (%)'); plt.ylabel('Count'); plt.legend()
                    fig.tight_layout()
                    fig.savefig(os.path.splitext(compare_pdf)[0] + f"_hist_alpha_L{i + 1}_C{k + 1}.png", dpi=200)
                    pdf.savefig(fig); plt.close(fig)

            # 5) —— 空气层 alpha：单独直方图 ——  # === NEW ===
            a_err_b = percent_err(air_true, air_before)
            a_err_a = percent_err(air_true, air_after)
            fig = plt.figure(figsize=(6.0, 5.0))
            plt.hist(a_err_b, bins=bins, alpha=0.5, label='Before')
            plt.hist(a_err_a, bins=bins, alpha=0.5, label='After')
            plt.title('Percent Error Histogram (alpha Air Layer)')
            plt.xlabel('Percent Error (%)'); plt.ylabel('Count'); plt.legend()
            fig.tight_layout()
            fig.savefig(os.path.splitext(compare_pdf)[0] + "_hist_alpha_air.png", dpi=200)
            pdf.savefig(fig); plt.close(fig)

            # 6) 汇总柱状图（整体）
            fig, axes = plt.subplots(1, 2, figsize=(12,5))
            cats = ['d', 'alpha-all']; x = np.arange(len(cats)); width = 0.35
            axes[0].bar(x - width/2, [R2_b_d, R2_b_a], width, label='Before')
            axes[0].bar(x + width/2, [R2_a_d, R2_a_a], width, label='After')
            axes[0].set_title('R² (higher is better)')
            axes[0].set_xticks(x); axes[0].set_xticklabels(cats)
            axes[0].set_ylabel('R²'); axes[0].legend()

            axes[1].bar(x - width/2, [MAPE_b_d, MAPE_b_a], width, label='Before')
            axes[1].bar(x + width/2, [MAPE_a_d, MAPE_a_a], width, label='After')
            axes[1].set_title('MAPE (%) (lower is better)')
            axes[1].set_xticks(x); axes[1].set_xticklabels(cats)
            axes[1].set_ylabel('MAPE (%)'); axes[1].legend()
            fig.tight_layout()
            fig.savefig(os.path.splitext(compare_pdf)[0] + "_metrics.png", dpi=200)
            pdf.savefig(fig); plt.close(fig)

            # 7) 文本页
            fig = plt.figure(figsize=(7.5, 4.5))
            plt.axis('off')
            txt = (
                f"Metrics (Before):\n"
                f"  Val Loss={meters.get('loss_total', np.nan):.6f}  "
                f"d={meters.get('loss_d', np.nan):.6f}  alpha(all)={meters.get('loss_alpha', np.nan):.6f}  "
                f"recon={meters.get('loss_recon', np.nan):.6f}  phys={meters.get('loss_phys', np.nan):.6f}\n"
                f"  R²(d)={R2_b_d:.4f}  R²(alpha-all)={R2_b_a:.4f}  "
                f"MAPE(d)={MAPE_b_d:.2f}%  MAPE(alpha-all)={MAPE_b_a:.2f}%\n\n"
                f"Metrics (After):\n"
                f"  R²(d)={R2_a_d:.4f}  R²(alpha-all)={R2_a_a:.4f}  "
                f"MAPE(d)={MAPE_a_d:.2f}%  MAPE(alpha-all)={MAPE_a_a:.2f}%\n"
            )
            plt.text(0.02, 0.98, txt, va='top', ha='left', fontsize=10)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        print(f"[+] 对比 PDF 已保存：{compare_pdf}")

def _save_loss_curve_png(train_losses, val_losses, out_png_path):
    epochs = range(1, len(train_losses)+1)
    fig = plt.figure(figsize=(6.4,4.8))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve')
    plt.legend(); plt.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=200)
    plt.close(fig)

@torch.no_grad()
def _infer_collect(file_path, ckpt_path, device='cuda', batch_size=32,
                   lambda_phys=1.0, beta_recon=1e-3, use_phys=True,
                   sim_len=3000, soft_tau=50.0, w_d=2.0, w_alpha=1.0,
                   save_csv_path=None):
    ckpt = torch.load(ckpt_path, map_location=device)
    n_layers = ckpt["n_layers"]
    alpha_mult = int(ckpt.get("alpha_mult", 1))
    norm = {
        "alpha_mean": ckpt["alpha_mean"].to(device),
        "alpha_std":  ckpt["alpha_std"].to(device),
        "d_mean":     ckpt["d_mean"].to(device),
        "d_std":      ckpt["d_std"].to(device),
        "air_alpha_mean": ckpt["air_alpha_mean"].to(device),   # === NEW ===
        "air_alpha_std":  ckpt["air_alpha_std"].to(device),
    }
    dataset = THzDataset(file_path, n_layers, alpha_mult=alpha_mult, norm=norm, device=device, show_progress=True)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = THzNet(n_layers, alpha_mult=alpha_mult, backbone='resatt').to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    d_true_all, d_pred_all = [], []
    a_true_all, a_pred_all = [], []
    air_true_all, air_pred_all = [], []
    meters = {"loss_total":0.0,"loss_d":0.0,"loss_alpha":0.0,"loss_phys":0.0,"loss_recon":0.0,"r2_d":0.0,"r2_alpha":0.0}
    n_batches = 0

    if save_csv_path is not None:
        alpha_cols = [f"alpha{i+1}" for i in range(n_layers)]
        header = ["index"] + [f"d{i+1}" for i in range(n_layers)] + ["air_alpha"] + alpha_cols
        rows = []

    for x, time, d_true_n, alpha_true_n, air_true_n, idx in loader:
        n_batches += 1
        x = x.to(device); raw = x.squeeze(1)
        d_true_n = d_true_n.to(device); alpha_true_n = alpha_true_n.to(device); air_true_n = air_true_n.to(device)

        d_pred_n, alpha_pred_n, air_pred_n, recon = model(x)
        loss, m = custom_loss_with_metrics(
            d_pred_n, d_true_n, alpha_pred_n, alpha_true_n, air_pred_n, air_true_n,
            recon, raw, time, dataset, idx,
            lambda_phys=lambda_phys, beta_recon=beta_recon, use_phys=use_phys,
            device=device, sim_len=sim_len, soft_tau=soft_tau,
            w_d=w_d, w_alpha=w_alpha
        )
        for k in meters: meters[k] += m[k]

        d_denorm_true     = d_true_n     * dataset.d_std.to(device) + dataset.d_mean.to(device)
        alpha_denorm_true = alpha_true_n * dataset.alpha_std_flat.to(device) + dataset.alpha_mean_flat.to(device)
        air_denorm_true   = air_true_n   * dataset.air_alpha_std.to(device) + dataset.air_alpha_mean.to(device)

        d_denorm_pred     = d_pred_n     * dataset.d_std.to(device) + dataset.d_mean.to(device)
        alpha_denorm_pred = alpha_pred_n * dataset.alpha_std_flat.to(device) + dataset.alpha_mean_flat.to(device)
        air_denorm_pred   = air_pred_n   * dataset.air_alpha_std.to(device) + dataset.air_alpha_mean.to(device)

        d_true_all.append(d_denorm_true.detach().cpu().numpy())
        d_pred_all.append(d_denorm_pred.detach().cpu().numpy())
        a_true_all.append(alpha_denorm_true.detach().cpu().numpy())
        a_pred_all.append(alpha_denorm_pred.detach().cpu().numpy())
        air_true_all.append(air_denorm_true.detach().cpu().numpy())
        air_pred_all.append(air_denorm_pred.detach().cpu().numpy())

        if save_csv_path is not None:
            for b, sid in enumerate(idx.tolist()):
                row = [sid] + d_denorm_pred[b].detach().cpu().numpy().tolist()
                row += [float(air_denorm_pred[b,0].detach().cpu().item())]
                row += alpha_denorm_pred[b].detach().cpu().numpy().tolist()
                rows.append(row)

    for k in meters: meters[k] /= max(n_batches,1)

    d_true = np.concatenate(d_true_all, axis=0).reshape(-1)
    d_pred = np.concatenate(d_pred_all, axis=0).reshape(-1)
    a_true_flat = np.concatenate(a_true_all, axis=0)  # [N, L]
    a_pred_flat = np.concatenate(a_pred_all, axis=0)
    air_true = np.concatenate(air_true_all, axis=0).reshape(-1)   # [N]
    air_pred = np.concatenate(air_pred_all, axis=0).reshape(-1)

    # “alpha-all” = 样本层 alpha 展平 + 空气层 alpha
    alpha_all_true = np.concatenate([a_true_flat.reshape(-1), air_true.reshape(-1)], axis=0)
    alpha_all_pred = np.concatenate([a_pred_flat.reshape(-1), air_pred.reshape(-1)], axis=0)

    def mape_np(y, p):
        y = np.asarray(y); p = np.asarray(p)
        mask = np.abs(y) > 1e-8
        if not np.any(mask):
            return np.nan
        return np.mean(np.abs((p[mask] - y[mask]) / y[mask])) * 100.0

    metrics = {
        "R2_d": _np_r2(d_true, d_pred),
        "R2_alpha": _np_r2(alpha_all_true, alpha_all_pred),  # === CHG：包含空气层
        "MAPE_d": mape_np(d_true, d_pred),
        "MAPE_alpha": mape_np(alpha_all_true, alpha_all_pred),
        "loss_total": meters["loss_total"],
        "loss_d": meters["loss_d"],
        "loss_alpha": meters["loss_alpha"],
        "loss_recon": meters["loss_recon"],
        "loss_phys": meters["loss_phys"],
    }

    # 额外信息：保留组件（这里 A=1）
    info = {
        "L": n_layers, "A": 1,
        "alpha_true_comps": [a_true_flat.reshape(-1)], "alpha_pred_comps": [a_pred_flat.reshape(-1)],
        "air_true": air_true.reshape(-1), "air_pred": air_pred.reshape(-1)   # === NEW ===
    }
    if save_csv_path is not None:
        with open(save_csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
        print(f"[+] 保存预测到 {save_csv_path}")

    return d_true, d_pred, alpha_all_true, alpha_all_pred, metrics, info
def _make_dual_compare_pdf(pdf_path,
                           # 曲线
                           train_no_phys, val_no_phys, train_with_phys, val_with_phys,
                           # 预测展开（整体）
                           d_true_A, d_pred_A, a_true_A, a_pred_A, metrics_A, tag_A,
                           d_true_B, d_pred_B, a_true_B, a_pred_B, metrics_B, tag_B,
                           # 额外：alpha 分量（按分量展开的备用数据）
                           alpha_true_comps_A, alpha_pred_comps_A,
                           alpha_true_comps_B, alpha_pred_comps_B,
                           alpha_mult:int,
                           n_layers:int,
                           # === NEW: 空气层单独数组（可选）
                           air_true_A=None, air_pred_A=None,
                           air_true_B=None, air_pred_B=None):
    ensure_dir(os.path.dirname(pdf_path) if os.path.dirname(pdf_path) else ".")
    with PdfPages(pdf_path) as pdf:
        L, A = n_layers, alpha_mult
        NA_d = len(d_true_A) // L
        NB_d = len(d_true_B) // L
        NA_a = len(a_true_A) // (L*A + 1)
        NB_a = len(a_true_B) // (L*A + 1)

        d_true_A_2d = np.asarray(d_true_A).reshape(NA_d, L)
        d_pred_A_2d = np.asarray(d_pred_A).reshape(NA_d, L)
        d_true_B_2d = np.asarray(d_true_B).reshape(NB_d, L)
        d_pred_B_2d = np.asarray(d_pred_B).reshape(NB_d, L)

        # 1) 双模型损失曲线
        fig = plt.figure(figsize=(7.0,5.2))
        epA = range(1, len(train_no_phys)+1)
        epB = range(1, len(train_with_phys)+1)
        plt.plot(epA, train_no_phys, label=f'{tag_A} Train')
        plt.plot(epA, val_no_phys,   label=f'{tag_A} Val')
        plt.plot(epB, train_with_phys, label=f'{tag_B} Train')
        plt.plot(epB, val_with_phys,   label=f'{tag_B} Val')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training / Validation Loss (Dual)')
        plt.legend(); plt.grid(alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


        def _r2_np(y, p):
            y = np.asarray(y); p = np.asarray(p)
            sse = np.sum((y - p) ** 2)
            sst = np.sum((y - np.mean(y)) ** 2) + 1e-12
            return 1.0 - sse / sst

        def scatter_dual(yA_t, yA_p, yB_t, yB_p, title, tag_A, tag_B):
            yA_t = np.asarray(yA_t).reshape(-1)
            yA_p = np.asarray(yA_p).reshape(-1)
            yB_t = np.asarray(yB_t).reshape(-1)
            yB_p = np.asarray(yB_p).reshape(-1)

            fig = plt.figure(figsize=(5.2, 5.2))
            plt.scatter(yA_t, yA_p, s=12, alpha=0.6, label=tag_A)
            plt.scatter(yB_t, yB_p, s=12, alpha=0.6, label=tag_B)

            lim_min = np.nanmin([yA_t.min(), yA_p.min(), yB_t.min(), yB_p.min()])
            lim_max = np.nanmax([yA_t.max(), yA_p.max(), yB_t.max(), yB_p.max()])
            plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1.2)

            r2A = _r2_np(yA_t, yA_p)
            r2B = _r2_np(yB_t, yB_p)
            plt.title(f"{title}\nR² {tag_A}={r2A:.3f} | {tag_B}={r2B:.3f}")
            plt.xlabel("True"); plt.ylabel("Predicted"); plt.legend()
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


        for i in range(L):
            yA_t = d_true_A_2d[:, i]; yA_p = d_pred_A_2d[:, i]
            yB_t = d_true_B_2d[:, i]; yB_p = d_pred_B_2d[:, i]
            scatter_dual(yA_t, yA_p, yB_t, yB_p, f"d{i+1}", tag_A, tag_B)


        for i in range(L):
            for k in range(alpha_mult):
                yA_t = np.asarray(alpha_true_comps_A[k]).reshape(-1, L)[:, i]
                yA_p = np.asarray(alpha_pred_comps_A[k]).reshape(-1, L)[:, i]
                yB_t = np.asarray(alpha_true_comps_B[k]).reshape(-1, L)[:, i]
                yB_p = np.asarray(alpha_pred_comps_B[k]).reshape(-1, L)[:, i]
                scatter_dual(yA_t, yA_p, yB_t, yB_p, f"alpha L{i+1} C{k+1}", tag_A, tag_B)


        if air_true_A is not None and air_pred_A is not None and air_true_B is not None and air_pred_B is not None:
            scatter_dual(
                np.asarray(air_true_A).reshape(-1),
                np.asarray(air_pred_A).reshape(-1),
                np.asarray(air_true_B).reshape(-1),
                np.asarray(air_pred_B).reshape(-1),
                "alpha Air Layer",
                tag_A, tag_B
            )

        # 4) —— 样本层 alpha：每层×每分量 单独直方图（双模型对比）
        def percent_err(y, p):
            y = np.asarray(y); p = np.asarray(p)
            mask = np.abs(y) > 1e-8
            e = np.zeros_like(p, dtype=np.float64)
            e[mask] = (p[mask] - y[mask]) / y[mask] * 100.0
            e[~mask] = 0.0
            return e

        bins = 40
        for i in range(L):
            for k in range(alpha_mult):
                a_true_A_il = np.asarray(alpha_true_comps_A[k]).reshape(-1, L)[:, i]
                a_pred_A_il = np.asarray(alpha_pred_comps_A[k]).reshape(-1, L)[:, i]
                a_true_B_il = np.asarray(alpha_true_comps_B[k]).reshape(-1, L)[:, i]
                a_pred_B_il = np.asarray(alpha_pred_comps_B[k]).reshape(-1, L)[:, i]

                a_err_A = percent_err(a_true_A_il, a_pred_A_il)
                a_err_B = percent_err(a_true_B_il, a_pred_B_il)

                fig = plt.figure(figsize=(6.0, 5.0))
                plt.hist(a_err_A, bins=bins, alpha=0.5, label=tag_A)
                plt.hist(a_err_B, bins=bins, alpha=0.5, label=tag_B)
                plt.title(f'Percent Error Histogram (alpha L{i + 1} C{k + 1})')
                plt.xlabel('Percent Error (%)'); plt.ylabel('Count'); plt.legend()
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 3) —— d：每层单独直方图
        bins = 40
        for i in range(L):
            d_err_A = percent_err(d_true_A_2d[:, i], d_pred_A_2d[:, i])
            d_err_B = percent_err(d_true_B_2d[:, i], d_pred_B_2d[:, i])
            fig = plt.figure(figsize=(6.0,5.0))
            plt.hist(d_err_A, bins=bins, alpha=0.5, label=tag_A)
            plt.hist(d_err_B, bins=bins, alpha=0.5, label=tag_B)
            plt.title(f'Percent Error Histogram (d{i+1})')
            plt.xlabel('Percent Error (%)'); plt.ylabel('Count'); plt.legend()
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 4) —— 空气层 alpha：单独直方图（若提供）
        if air_true_A is not None and air_pred_A is not None and air_true_B is not None and air_pred_B is not None:
            a_err_A = percent_err(np.asarray(air_true_A).reshape(-1), np.asarray(air_pred_A).reshape(-1))
            a_err_B = percent_err(np.asarray(air_true_B).reshape(-1), np.asarray(air_pred_B).reshape(-1))
            fig = plt.figure(figsize=(6.0,5.0))
            plt.hist(a_err_A, bins=bins, alpha=0.5, label=tag_A)
            plt.hist(a_err_B, bins=bins, alpha=0.5, label=tag_B)
            plt.title('Percent Error Histogram (alpha Air Layer)')
            plt.xlabel('Percent Error (%)'); plt.ylabel('Count'); plt.legend()
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ========= NEW: 每个输出量的 R² 与 MAPE 单图（同图比较 A/B） =========
        def _mape_np(y, p):
            y = np.asarray(y); p = np.asarray(p)
            mask = np.abs(y) > 1e-8
            if not np.any(mask): return np.nan
            return np.mean(np.abs((p[mask] - y[mask]) / y[mask])) * 100.0

        # ---- d_i 的 R² & MAPE（每层各一张 R² 图 + 一张 MAPE 图）----
        for i in range(L):
            yA_t = d_true_A_2d[:, i]; yA_p = d_pred_A_2d[:, i]
            yB_t = d_true_B_2d[:, i]; yB_p = d_pred_B_2d[:, i]
            r2A = _r2_np(yA_t, yA_p); r2B = _r2_np(yB_t, yB_p)
            mA  = _mape_np(yA_t, yA_p); mB  = _mape_np(yB_t, yB_p)

            # R²
            fig = plt.figure(figsize=(4.8,4.2))
            plt.bar([0,1], [r2A, r2B], tick_label=[tag_A, tag_B])
            plt.ylabel('R²'); plt.title(f'R² per Output: d{i+1}')
            fig.tight_layout()
            png_path = os.path.splitext(pdf_path)[0] + f"_r2_d{i+1}.png"
            fig.savefig(png_path, dpi=200); pdf.savefig(fig); plt.close(fig)

            # MAPE
            fig = plt.figure(figsize=(4.8,4.2))
            plt.bar([0,1], [mA, mB], tick_label=[tag_A, tag_B])
            plt.ylabel('MAPE (%)'); plt.title(f'MAPE per Output: d{i+1}')
            fig.tight_layout()
            png_path = os.path.splitext(pdf_path)[0] + f"_mape_d{i+1}.png"
            fig.savefig(png_path, dpi=200); pdf.savefig(fig); plt.close(fig)

        # ---- alpha_{i,k} 的 R² & MAPE（每层每分量各两张图）----
        for i in range(L):
            for k in range(alpha_mult):
                yA_t = np.asarray(alpha_true_comps_A[k]).reshape(-1, L)[:, i]
                yA_p = np.asarray(alpha_pred_comps_A[k]).reshape(-1, L)[:, i]
                yB_t = np.asarray(alpha_true_comps_B[k]).reshape(-1, L)[:, i]
                yB_p = np.asarray(alpha_pred_comps_B[k]).reshape(-1, L)[:, i]
                r2A = _r2_np(yA_t, yA_p); r2B = _r2_np(yB_t, yB_p)
                mA  = _mape_np(yA_t, yA_p); mB  = _mape_np(yB_t, yB_p)

                # R²
                fig = plt.figure(figsize=(4.8,4.2))
                plt.bar([0,1], [r2A, r2B], tick_label=[tag_A, tag_B])
                plt.ylabel('R²'); plt.title(f'R² per Output: alpha L{i+1} C{k+1}')
                fig.tight_layout()
                png_path = os.path.splitext(pdf_path)[0] + f"_r2_alpha_L{i+1}_C{k+1}.png"
                fig.savefig(png_path, dpi=200); pdf.savefig(fig); plt.close(fig)

                # MAPE
                fig = plt.figure(figsize=(4.8,4.2))
                plt.bar([0,1], [mA, mB], tick_label=[tag_A, tag_B])
                plt.ylabel('MAPE (%)'); plt.title(f'MAPE per Output: alpha L{i+1} C{k+1}')
                fig.tight_layout()
                png_path = os.path.splitext(pdf_path)[0] + f"_mape_alpha_L{i+1}_C{k+1}.png"
                fig.savefig(png_path, dpi=200); pdf.savefig(fig); plt.close(fig)

        # ---- 空气层 alpha 的 R² & MAPE（若提供）----
        if air_true_A is not None and air_pred_A is not None and air_true_B is not None and air_pred_B is not None:
            yA_t = np.asarray(air_true_A).reshape(-1); yA_p = np.asarray(air_pred_A).reshape(-1)
            yB_t = np.asarray(air_true_B).reshape(-1); yB_p = np.asarray(air_pred_B).reshape(-1)
            r2A = _r2_np(yA_t, yA_p); r2B = _r2_np(yB_t, yB_p)
            mA  = _mape_np(yA_t, yA_p); mB  = _mape_np(yB_t, yB_p)

            # R²
            fig = plt.figure(figsize=(4.8,4.2))
            plt.bar([0,1], [r2A, r2B], tick_label=[tag_A, tag_B])
            plt.ylabel('R²'); plt.title('R² per Output: alpha Air')
            fig.tight_layout()
            png_path = os.path.splitext(pdf_path)[0] + f"_r2_alpha_air.png"
            fig.savefig(png_path, dpi=200); pdf.savefig(fig); plt.close(fig)

            # MAPE
            fig = plt.figure(figsize=(4.8,4.2))
            plt.bar([0,1], [mA, mB], tick_label=[tag_A, tag_B])
            plt.ylabel('MAPE (%)'); plt.title('MAPE per Output: alpha Air')
            fig.tight_layout()
            png_path = os.path.splitext(pdf_path)[0] + f"_mape_alpha_air.png"
            fig.savefig(png_path, dpi=200); pdf.savefig(fig); plt.close(fig)
        # ========= /NEW =========

        # 5) R² & MAPE 汇总柱状图（整体）
        fig, axes = plt.subplots(1, 2, figsize=(12,5))
        cats = ['d', 'alpha-all']; x = np.arange(len(cats)); width = 0.35
        R2_A = [metrics_A['R2_d'], metrics_A['R2_alpha']]
        R2_B = [metrics_B['R2_d'], metrics_B['R2_alpha']]
        axes[0].bar(x - width/2, R2_A, width, label=tag_A)
        axes[0].bar(x + width/2, R2_B, width, label=tag_B)
        axes[0].set_title('R² (higher is better)')
        axes[0].set_xticks(x); axes[0].set_xticklabels(cats)
        axes[0].set_ylabel('R²'); axes[0].legend()

        MAPE_A = [metrics_A['MAPE_d'], metrics_A['MAPE_alpha']]
        MAPE_B = [metrics_B['MAPE_d'], metrics_B['MAPE_alpha']]
        axes[1].bar(x - width/2, MAPE_A, width, label=tag_A)
        axes[1].bar(x + width/2, MAPE_B, width, label=tag_B)
        axes[1].set_title('MAPE (%) (lower is better)')
        axes[1].set_xticks(x); axes[1].set_xticklabels(cats)
        axes[1].set_ylabel('MAPE (%)'); axes[1].legend()
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 6) 文本页
        fig = plt.figure(figsize=(7.5, 4.5))
        plt.axis('off')
        txt = (
            f"{tag_A} Metrics:\n"
            f"  Val Loss={metrics_A.get('loss_total', np.nan):.6f}  "
            f"d={metrics_A.get('loss_d', np.nan):.6f}  alpha(all)={metrics_A.get('loss_alpha', np.nan):.6f}  "
            f"recon={metrics_A.get('loss_recon', np.nan):.6f}  phys={metrics_A.get('loss_phys', np.nan):.6f}\n"
            f"  R²(d)={metrics_A.get('R2_d', np.nan):.4f}  R²(alpha-all)={metrics_A.get('R2_alpha', np.nan):.4f}  "
            f"MAPE(d)={metrics_A.get('MAPE_d', np.nan):.2f}%  MAPE(alpha-all)={metrics_A.get('MAPE_alpha', np.nan):.2f}%\n\n"
            f"{tag_B} Metrics:\n"
            f"  Val Loss={metrics_B.get('loss_total', np.nan):.6f}  "
            f"d={metrics_B.get('loss_d', np.nan):.6f}  alpha(all)={metrics_B.get('loss_alpha', np.nan):.6f}  "
            f"recon={metrics_B.get('loss_recon', np.nan):.6f}  phys={metrics_B.get('loss_phys', np.nan):.6f}\n"
            f"  R²(d)={metrics_B.get('R2_d', np.nan):.4f}  R²(alpha-all)={metrics_B.get('R2_alpha', np.nan):.4f}  "
            f"MAPE(d)={metrics_B.get('MAPE_d', np.nan):.2f}%  MAPE(alpha-all)={metrics_B.get('MAPE_alpha', np.nan):.2f}%\n"
        )
        plt.text(0.02, 0.98, txt, va='top', ha='left', fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    print(f"[+] 双跑总对比 PDF 已保存：{pdf_path}")


# ============== 训练（单模型）与 双模型训练（traindual） ==============

def train(file_path, n_layers=1, alpha_mult=1, epochs=50, batch_size=16, lr=1e-3, device='cuda',
          val_ratio=0.1, lambda_phys=1.0, beta_recon=1e-3, out_dir='ckpts_single',
          sim_len=3000, soft_tau=50.0, backbone='resatt',
          w_d: float = 2.0, w_alpha: float = 1.0, progress: bool = True, use_phys=True):
    """最小化单模型训练入口，补齐 CLI 的 'train' 分支"""
    ensure_dir(out_dir)
    figs_dir = os.path.join(out_dir, "figs"); ensure_dir(figs_dir)

    dataset = THzDataset(file_path, n_layers, alpha_mult=alpha_mult, device=device, show_progress=progress)
    print(f">>> 数据集大小 N={len(dataset)}, L={n_layers}, A={alpha_mult}")
    N = len(dataset)
    val_len = max(1, int(N * val_ratio)); train_len = N - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    norm = {
        "alpha_mean": dataset.alpha_mean_flat.squeeze(0).detach(),
        "alpha_std":  dataset.alpha_std_flat.squeeze(0).detach(),
        "d_mean":     dataset.d_mean.squeeze(0).detach(),
        "d_std":      dataset.d_std.squeeze(0).detach(),
        "air_alpha_mean": dataset.air_alpha_mean.detach(),
        "air_alpha_std":  dataset.air_alpha_std.detach(),
    }


    _ = THzDataset(file_path, n_layers, alpha_mult=alpha_mult, norm=norm, device=device, show_progress=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, drop_last=False)

    model = THzNet(n_layers, alpha_mult=alpha_mult, backbone=backbone).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    tr, va = [], []; best, best_ep = float('inf'), 0
    for ep in range(1, epochs+1):
        print(f"\n===== [Single] Epoch {ep}/{epochs} ({backbone}) =====")
        model.train(); sumLoss, cnt = 0.0, 0
        for bi, (x, time, d_true_n, alpha_true_n, air_true_n, idx) in enumerate(train_loader, 1):
            x = x.to(device); raw = x.squeeze(1)
            d_true_n = d_true_n.to(device); alpha_true_n = alpha_true_n.to(device); air_true_n = air_true_n.to(device)

            opt.zero_grad()
            d_pred_n, alpha_pred_n, air_pred_n, recon = model(x)
            loss, m = custom_loss_with_metrics(
                d_pred_n, d_true_n, alpha_pred_n, alpha_true_n, air_pred_n, air_true_n,
                recon, raw, time, dataset, idx,
                lambda_phys=lambda_phys if use_phys else 0.0, beta_recon=beta_recon, use_phys=use_phys,
                device=device, sim_len=sim_len, soft_tau=soft_tau,
                w_d=w_d, w_alpha=w_alpha
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            sumLoss += loss.item() * x.size(0); cnt += x.size(0)

            print(f"[Epoch {ep} Batch {bi}/{len(train_loader)}] "
                  f"Loss={m['loss_total']:.6f} (d={m['loss_d']:.6f}, a={m['loss_alpha']:.6f}, "
                  f"phys={m['loss_phys']:.6f}, recon={m['loss_recon']:.6f})")

        tr.append(sumLoss / max(cnt,1))
        print(f"[Train] Loss_total={tr[-1]:.6f}")

        vm = evaluate(model, val_loader, dataset, device,
                      lambda_phys=lambda_phys if use_phys else 0.0, beta_recon=beta_recon, use_phys=use_phys,
                      sim_len=sim_len, soft_tau=soft_tau, w_d=w_d, w_alpha=w_alpha)
        va.append(vm['loss_total'])
        print(f"[Val] Loss={vm['loss_total']:.6f} (d={vm['loss_d']:.6f}, a={vm['loss_alpha']:.6f}, "
              f"phys={vm['loss_phys']:.6f}, recon={vm['loss_recon']:.6f}) R²_d={vm['r2_d']:.3f} R²_a={vm['r2_alpha']:.3f}")

        # 保存 last/best
        save_checkpoint(os.path.join(out_dir, f"last_n{n_layers}.pth"), model, n_layers, alpha_mult, norm, ep)
        if vm['loss_total'] < best:
            best, best_ep = vm['loss_total'], ep
            save_checkpoint(os.path.join(out_dir, f"best_n{n_layers}.pth"), model, n_layers, alpha_mult, norm, ep)
            print(f"[*] Saved BEST (val loss {best:.6f} at epoch {best_ep})")

    _save_loss_curve_png(tr, va, os.path.join(figs_dir, f"loss_curve_single_n{n_layers}.png"))
    print(f"=== Single Training finished. Best {best:.6f}@{best_ep} ===")

def train_dual(file_path, n_layers=1, alpha_mult=1, epochs=50, batch_size=16, lr=1e-3, device='cuda',
               val_ratio=0.1, lambda_phys=1.0, beta_recon=1e-3, out_dir='ckpts_dual',
               sim_len=3000, soft_tau=50.0, backbone='resatt',
               w_d: float = 2.0, w_alpha: float = 1.0, progress: bool = True):
    ensure_dir(out_dir)
    subA = os.path.join(out_dir, "no_phys")
    subB = os.path.join(out_dir, "with_phys")
    ensure_dir(subA); ensure_dir(subB)
    figs_dir = os.path.join(out_dir, "figs")
    ensure_dir(figs_dir)

    # —— 数据与划分（共享） ——
    full_dataset = THzDataset(file_path, n_layers, alpha_mult=alpha_mult, device=device, show_progress=progress)
    print(f">>> 数据集大小 N={len(full_dataset)}, L={n_layers}, A={alpha_mult}")
    N = len(full_dataset)
    val_len = max(1, int(N * val_ratio)); train_len = N - val_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len])

    norm = {
        "alpha_mean": full_dataset.alpha_mean_flat.squeeze(0).detach(),
        "alpha_std":  full_dataset.alpha_std_flat.squeeze(0).detach(),
        "d_mean":     full_dataset.d_mean.squeeze(0).detach(),
        "d_std":      full_dataset.d_std.squeeze(0).detach(),
        "air_alpha_mean": full_dataset.air_alpha_mean.detach(),
        "air_alpha_std":  full_dataset.air_alpha_std.detach(),
    }
    _ = THzDataset(file_path, n_layers, alpha_mult=alpha_mult, norm=norm, device=device, show_progress=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, drop_last=False)

    # —— 两个模型与优化器 ——
    model_A = THzNet(n_layers, alpha_mult=alpha_mult, backbone=backbone).to(device)  # no_phys
    model_B = THzNet(n_layers, alpha_mult=alpha_mult, backbone=backbone).to(device)  # with_phys
    opt_A = optim.Adam(model_A.parameters(), lr=lr)
    opt_B = optim.Adam(model_B.parameters(), lr=lr)

    # 曲线与 best
    trA, vaA = [], []; bestA, bestA_ep = float('inf'), 0
    trB, vaB = [], []; bestB, bestB_ep = float('inf'), 0

    for ep in range(1, epochs+1):
        print(f"\n===== [Dual] Epoch {ep}/{epochs} ({backbone}) =====")
        model_A.train(); model_B.train()
        sumA, cntA = 0.0, 0
        sumB, cntB = 0.0, 0

        for bi, (x, time, d_true_n, alpha_true_n, air_true_n, idx) in enumerate(train_loader, 1):
            x = x.to(device); raw = x.squeeze(1)
            d_true_n = d_true_n.to(device); alpha_true_n = alpha_true_n.to(device); air_true_n = air_true_n.to(device)

            # --- A: no_phys ---
            opt_A.zero_grad()
            dA, aA, airA, rA = model_A(x)
            lossA, mA = custom_loss_with_metrics(
                dA, d_true_n, aA, alpha_true_n, airA, air_true_n,
                rA, raw, time, full_dataset, idx,
                lambda_phys=0.0, beta_recon=beta_recon, use_phys=False,
                device=device, sim_len=sim_len, soft_tau=soft_tau, w_d=w_d, w_alpha=w_alpha
            )
            lossA.backward()
            torch.nn.utils.clip_grad_norm_(model_A.parameters(), max_norm=5.0)
            opt_A.step()
            sumA += lossA.item() * x.size(0); cntA += x.size(0)

            # --- B: with_phys ---
            opt_B.zero_grad()
            dB, aB, airB, rB = model_B(x)
            lossB, mB = custom_loss_with_metrics(
                dB, d_true_n, aB, alpha_true_n, airB, air_true_n,
                rB, raw, time, full_dataset, idx,
                lambda_phys=lambda_phys, beta_recon=beta_recon, use_phys=True,
                device=device, sim_len=sim_len, soft_tau=soft_tau, w_d=w_d, w_alpha=w_alpha
            )
            lossB.backward()
            torch.nn.utils.clip_grad_norm_(model_B.parameters(), max_norm=5.0)
            opt_B.step()
            sumB += lossB.item() * x.size(0); cntB += x.size(0)

            print(f"[Epoch {ep} Batch {bi}/{len(train_loader)}] "
                  f"A(no_phys) Loss={mA['loss_total']:.6f} (d={mA['loss_d']:.6f}, a={mA['loss_alpha']:.6f}, "
                  f"recon={mA['loss_recon']:.6f})  |  "
                  f"B(with_phys) Loss={mB['loss_total']:.6f} (d={mB['loss_d']:.6f}, a={mB['loss_alpha']:.6f}, "
                  f"phys={mB['loss_phys']:.6f}, recon={mB['loss_recon']:.6f})")

        trA.append(sumA / max(cntA,1))
        trB.append(sumB / max(cntB,1))
        print(f"[Train] A(no_phys) Loss_total={trA[-1]:.6f}  |  B(with_phys) Loss_total={trB[-1]:.6f}")

        # 验证
        vmA = evaluate(model_A, val_loader, full_dataset, device,
                       lambda_phys=0.0, beta_recon=beta_recon, use_phys=False,
                       sim_len=sim_len, soft_tau=soft_tau, w_d=w_d, w_alpha=w_alpha)
        vmB = evaluate(model_B, val_loader, full_dataset, device,
                       lambda_phys=lambda_phys, beta_recon=beta_recon, use_phys=True,
                       sim_len=sim_len, soft_tau=soft_tau, w_d=w_d, w_alpha=w_alpha)
        vaA.append(vmA['loss_total']); vaB.append(vmB['loss_total'])
        print(f"[Val]  A Loss={vmA['loss_total']:.6f} (d={vmA['loss_d']:.6f}, a={vmA['loss_alpha']:.6f}, "
              f"recon={vmA['loss_recon']:.6f}) R²_d={vmA['r2_d']:.3f} R²_a={vmA['r2_alpha']:.3f}")
        print(f"[Val]  B Loss={vmB['loss_total']:.6f} (d={vmB['loss_d']:.6f}, a={vmB['loss_alpha']:.6f}, "
              f"phys={vmB['loss_phys']:.6f}, recon={vmB['loss_recon']:.6f}) R²_d={vmB['r2_d']:.3f} R²_a={vmB['r2_alpha']:.3f}")

        # 保存 last
        save_checkpoint(os.path.join(subA, f"last_n{n_layers}.pth"), model_A, n_layers, alpha_mult, norm, ep)
        save_checkpoint(os.path.join(subB, f"last_n{n_layers}.pth"), model_B, n_layers, alpha_mult, norm, ep)

        # 保存 best
        if vmA['loss_total'] < bestA:
            bestA, bestA_ep = vmA['loss_total'], ep
            save_checkpoint(os.path.join(subA, f"best_n{n_layers}.pth"), model_A, n_layers, alpha_mult, norm, ep)
            print(f"[*] A Saved BEST (val loss {bestA:.6f} at epoch {bestA_ep})")
        if vmB['loss_total'] < bestB:
            bestB, bestB_ep = vmB['loss_total'], ep
            save_checkpoint(os.path.join(subB, f"best_n{n_layers}.pth"), model_B, n_layers, alpha_mult, norm, ep)
            print(f"[*] B Saved BEST (val loss {bestB:.6f} at epoch {bestB_ep})")

    print(f"=== Dual Training finished. Best A {bestA:.6f}@{bestA_ep} | Best B {bestB:.6f}@{bestB_ep} ===")

    # 导出各自损失曲线 PNG
    _save_loss_curve_png(trA, vaA, os.path.join(figs_dir, f"loss_curve_no_phys_n{n_layers}.png"))
    _save_loss_curve_png(trB, vaB, os.path.join(figs_dir, f"loss_curve_with_phys_n{n_layers}.png"))

    # 统一测试 + 生成总对比 PDF
    bestA_path = os.path.join(subA, f"best_n{n_layers}.pth")
    bestB_path = os.path.join(subB, f"best_n{n_layers}.pth")

    d_t_A, d_p_A, a_t_A_all, a_p_A_all, metA, infoA = _infer_collect(
        file_path, bestA_path, device=device, batch_size=batch_size,
        lambda_phys=0.0, beta_recon=beta_recon, use_phys=False,
        sim_len=sim_len, soft_tau=soft_tau, w_d=w_d, w_alpha=w_alpha,
        save_csv_path=os.path.join(subA, f"predictions_no_phys.csv")
    )
    d_t_B, d_p_B, a_t_B_all, a_p_B_all, metB, infoB = _infer_collect(
        file_path, bestB_path, device=device, batch_size=batch_size,
        lambda_phys=lambda_phys, beta_recon=beta_recon, use_phys=True,
        sim_len=sim_len, soft_tau=soft_tau, w_d=w_d, w_alpha=w_alpha,
        save_csv_path=os.path.join(subB, f"predictions_with_phys.csv")
    )

    # alpha 分量（样本层展开）+ 空气层（单列）
    comps_t_A, comps_p_A = infoA["alpha_true_comps"], infoA["alpha_pred_comps"]
    comps_t_B, comps_p_B = infoB["alpha_true_comps"], infoB["alpha_pred_comps"]
    A = infoA["A"]  # 与 infoB 一致
    air_t_A, air_p_A = infoA["air_true"], infoA["air_pred"]
    air_t_B, air_p_B = infoB["air_true"], infoB["air_pred"]

    pdf_out = os.path.join(figs_dir, f"dual_compare_n{n_layers}.pdf")
    _make_dual_compare_pdf(
        pdf_out,
        trA, vaA, trB, vaB,
        d_t_A, d_p_A, a_t_A_all, a_p_A_all, metA, "No Physics",
        d_t_B, d_p_B, a_t_B_all, a_p_B_all, metB, "With Physics",
        comps_t_A, comps_p_A, comps_t_B, comps_p_B, A,
        n_layers,
        air_true_A=air_t_A, air_pred_A=air_p_A, air_true_B=air_t_B, air_pred_B=air_p_B  # === NEW ===
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',  type=str, choices=['train','test','traindual'], default='traindual')
    parser.add_argument('--file',  type=str, default='1layers_signal.xlsx')
    parser.add_argument('--n',     type=int, choices=[1,2,3], default=1)
    parser.add_argument('--alpha-mult', type=int, choices=[1,2], default=1)
    parser.add_argument('--epochs',type=int, default=50)
    parser.add_argument('--bs',    type=int, default=256)
    parser.add_argument('--lr',    type=float, default=1e-3)
    parser.add_argument('--val',   type=float, default=0.1)
    parser.add_argument('--lambda_phys', type=float, default=2)
    parser.add_argument('--beta_recon',  type=float, default=1e-3)
    parser.add_argument('--out',   type=str, default='ckpts')
    parser.add_argument('--ckpt',  type=str, default='ckpts_dual/with_phys/best_n1.pth')
    parser.add_argument('--save-sim', type=int, default=1)
    parser.add_argument('--phys',  type=int, default=1, help='1=开物理损失; 0=关')
    parser.add_argument('--sim-len', type=int, default=3000)
    parser.add_argument('--soft-tau', type=float, default=50.0)
    parser.add_argument('--backbone', type=str, choices=['resnet','resatt'], default='resatt')
    parser.add_argument('--w_d', type=float, default=2.0, help='d的监督权重')
    parser.add_argument('--w_alpha', type=float, default=1.0, help='alpha的监督权重')
    parser.add_argument('--progress', type=int, default=1)
    parser.add_argument('--plot', type=int, default=1)
    parser.add_argument('--plot-dir', type=str, default='figs11', help='图表输出目录')
    parser.add_argument('--plot-samples', type=int, default=12, help='信号可视化抽样样本数')
    parser.add_argument('--seed', type=int, default=42, help='抽样随机种子')
    parser.add_argument('--plot-style', type=str, choices=['stack','normalize','subplots'],
                        default='stack')
    parser.add_argument('--offset', type=float, default=None,
                        help='stack 模式的纵向错位幅度')
    parser.add_argument('--compare', type=int, default=1, help='1=生成 Before/After 对比 PDF（单跑test用）')
    parser.add_argument('--compare-steps', type=int, default=50, help='微调步数')
    parser.add_argument('--compare-lr', type=float, default=0.05, help='微调学习率')
    parser.add_argument('--compare-lam-phys', type=float, default=0.7, help='物理一致性损失权重')
    parser.add_argument('--compare-lam-prior', type=float, default=0.1, help='先验约束权重')
    parser.add_argument('--compare-pdf', type=str, default='figs/phys_compare.pdf')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_phys = bool(args.phys)

    if args.mode == 'train':
        train(args.file, n_layers=args.n, alpha_mult=args.alpha_mult, epochs=args.epochs,
              batch_size=args.bs, lr=args.lr, device=device,
              val_ratio=args.val, lambda_phys=args.lambda_phys,
              beta_recon=args.beta_recon, out_dir=args.out,
              use_phys=use_phys, sim_len=args.sim_len, soft_tau=args.soft_tau,
              backbone=args.backbone, w_d=args.w_d, w_alpha=args.w_alpha,
              progress=bool(args.progress))

    elif args.mode == 'test':
        ckpt = torch.load(args.ckpt, map_location='cpu')
        ckpt_alpha_mult = int(ckpt.get("alpha_mult", args.alpha_mult))
        test(args.file, ckpt_path=args.ckpt, batch_size=args.bs,
             device=device, save_csv='predictions.csv',
             lambda_phys=args.lambda_phys, beta_recon=args.beta_recon,
             use_phys=use_phys, sim_len=args.sim_len, soft_tau=args.soft_tau,
             save_sim=args.save_sim, w_d=args.w_d, w_alpha=args.w_alpha,
             plot=bool(args.plot), plot_dir=args.plot_dir,
             plot_samples=args.plot_samples, seed=args.seed,
             plot_style=args.plot_style, offset_value=args.offset,
             compare=bool(args.compare), compare_steps=args.compare_steps,
             compare_lr=args.compare_lr, compare_lam_phys=args.compare_lam_phys,
             compare_lam_prior=args.compare_lam_prior, compare_pdf=args.compare_pdf)

    else:  # traindual
        train_dual(args.file, n_layers=args.n, alpha_mult=args.alpha_mult, epochs=args.epochs,
                   batch_size=args.bs, lr=args.lr, device=device,
                   val_ratio=args.val, lambda_phys=args.lambda_phys,
                   beta_recon=args.beta_recon, out_dir=args.out if args.out!='ckpts' else 'ckpts_dual',
                   sim_len=args.sim_len, soft_tau=args.soft_tau, backbone=args.backbone,
                   w_d=args.w_d, w_alpha=args.w_alpha, progress=bool(args.progress))
