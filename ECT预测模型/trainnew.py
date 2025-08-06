import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from scipy import stats
from read import process_and_merge  # 生成并返回候选特征频率矩阵或列表
from datetime import datetime
import joblib  # 模型保存工具

# 导入用户自定义模型
try:
    from custom_model import CustomModel
except ImportError:
    CustomModel = None
    print("警告：未找到 custom_model.py 中的 CustomModel，请在 custom_model.py 中实现。")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    缺失值填充与异常值处理
    """
    df = df.copy()
    df.fillna(df.mean(), inplace=True)
    for col in df.columns[2:]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        low = Q1 - 1.5 * IQR
        high = Q3 + 1.5 * IQR
        df[col] = np.where((df[col] < low) | (df[col] > high), df[col].mean(), df[col])
    return df


def build_model(model_name: str, alpha: float = 1.0):
    name = model_name.lower()
    if name == 'lasso':
        return Lasso(alpha=alpha)
    elif name == 'svr':
        return SVR(kernel='rbf')
    elif name == 'randomforest':
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif name == 'custommodel':
        return CustomModel()
    else:
        return Ridge(alpha=alpha)


def evaluate_features(feature_subset: list, model_name: str = 'Ridge', alpha: float = 1.0) -> tuple:
    """
    留一法交叉验证评估给定特征子集和模型
    返回：平均 r 值, 平均 p 值
    """
    file_paths = [f"./data/训练集{i}.xlsx" for i in range(1, 6)]
    cumulative_r = 0.0
    cumulative_p = 0.0

    for path in file_paths:
        df = pd.read_excel(path)
        feats = df.columns[df.columns.get_loc('TRT(min)'): df.columns.get_loc('O2_N2_SW_positivePeaks') + 1]
        data = df[['Person', 'PANSS(%)'] + list(feats)]
        data = preprocess_data(data)

        X_full = data.iloc[:, 2:].values
        X = X_full[:, feature_subset]
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        y = data['PANSS(%)'].values

        X_scaled = StandardScaler().fit_transform(X)

        loo = LeaveOneOut()
        y_true, y_pred = [], []
        for tr_idx, te_idx in loo.split(X_scaled):
            X_tr, X_te = X_scaled[tr_idx], X_scaled[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            model = build_model(model_name, alpha)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te.reshape(1, -1))[0]
            y_true.append(y_te[0])
            y_pred.append(pred)

        r_val, p_val = stats.pearsonr(y_true, y_pred)
        cumulative_r += r_val
        cumulative_p += p_val

    return cumulative_r / 5, cumulative_p / 5


def find_best_features_for_model(model_name: str) -> tuple:
    """
    逐步前向选择最优特征子集（针对指定模型）
    返回：最优特征索引列表, 对应平均 r 值
    """
    raw_indices = process_and_merge()
    if isinstance(raw_indices, np.ndarray):
        all_indices = np.unique(raw_indices.flatten()).tolist()
    else:
        all_indices = list(raw_indices)
    all_indices = [int(i) for i in all_indices]

    scores = {}
    # 单特征评估
    for idx in all_indices:
        r, _ = evaluate_features([idx], model_name)
        scores[(idx,)] = r
        print(f"[{model_name}] 特征 [{idx}] 单特征平均 r: {r:.4f}")

    best_tuple = max(scores, key=scores.get)
    best_subset = list(best_tuple)
    best_r = scores[best_tuple]
    remaining = [i for i in all_indices if i not in best_subset]

    # 逐步向前选择
    while True:
        improved = False
        candidate, candidate_r = None, best_r
        for idx in remaining:
            subset = best_subset + [idx]
            r, _ = evaluate_features(subset, model_name)
            print(f"[{model_name}] 评估子集 {subset} 平均 r: {r:.4f}")
            if r > candidate_r:
                candidate, candidate_r = idx, r
        if candidate is not None:
            best_subset.append(candidate)
            remaining.remove(candidate)
            best_r = candidate_r
            improved = True
            print(f"[{model_name}] 更新子集: {best_subset}, 新平均 r: {best_r:.4f}")
        if not improved:
            break

    print(f"[{model_name}] 最终最优特征子集: {best_subset}, 平均 r: {best_r:.4f}")
    return best_subset, best_r


def main():
    model_list = ['custommodel']  # custommodel RIHDGE
    log_file = 'stacking_base_model_log.txt'

    for m in model_list:
        try:
            best_feats, best_r = find_best_features_for_model(m)
            _, best_p = evaluate_features(best_feats, m)

            # 全数据拟合并保存模型
            df_all = pd.concat([pd.read_excel(f"./data/训练集{i}.xlsx") for i in range(1, 6)], ignore_index=True)
            feats = df_all.columns[df_all.columns.get_loc('TRT(min)'): df_all.columns.get_loc('O2_N2_SW_positivePeaks') + 1]
            data = df_all[['Person', 'PANSS(%)'] + list(feats)]
            data = preprocess_data(data)

            X_full = data.iloc[:, 2:].values
            X = X_full[:, best_feats]
            y = data['PANSS(%)'].values
            X_scaled = StandardScaler().fit_transform(X)

            model = build_model(m)
            model.fit(X_scaled, y)

            # 保存模型
            model_path = f'model_{m}.pkl'
            joblib.dump(model, model_path)

            # 保存结果文件
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_file = f"results_{m}_{ts}.txt"
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(f"模型: {m}\n")
                f.write(f"最优特征子集: {best_feats}\n")
                f.write(f"平均 r 值: {best_r:.4f}\n")
                f.write(f"平均 p 值: {best_p:.4f}\n")

            # 追加日志
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"模型: {m}\n")
                log.write(f"最优特征子集: {best_feats}\n")
                log.write(f"平均 r 值: {best_r:.4f}\n")
                log.write(f"平均 p 值: {best_p:.4f}\n")
                log.write(f"模型已保存: {model_path}\n\n")

            print(f"[{m}] 结果与模型已保存：{out_file}, {model_path}\n")

        except Exception as e:
            print(f"模型 {m} 处理出错: {e}\n")
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"模型 {m} 处理出错: {e}\n\n")


if __name__ == '__main__':
    main()
