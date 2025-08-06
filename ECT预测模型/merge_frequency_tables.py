import pandas as pd
import numpy as np
from collections import Counter
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler



def process_file(file_path, model_type='ElasticNet'):
    """
    处理单个文件，LOOCV + 指定模型，返回二值化的频率表 DataFrame
    """
    data = pd.read_excel(file_path)

    # 预处理：填充缺失、剔除异常
    def preprocess_data(df):
        df = df.copy()
        df.fillna(df.mean(), inplace=True)
        for col in df.columns[2:]:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df[col] = np.where((df[col] < lb) | (df[col] > ub), df[col].mean(), df[col])
        return df

    # 选取特征列
    feat_cols = data.columns[
        data.columns.get_loc('TRT(min)'):
        data.columns.get_loc('O2_N2_SW_positivePeaks') + 1
    ]
    data = data[['Person', 'PANSS(%)'] + list(feat_cols)]
    data = preprocess_data(data)

    X = data.iloc[:, 2:].values
    y = data['PANSS(%)'].values
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    X_scaled = StandardScaler().fit_transform(X)

    loo = LeaveOneOut()
    counter = Counter()

    for train_idx, _ in loo.split(X_scaled):
        X_tr, y_tr = X_scaled[train_idx], y[train_idx]

        # 实例化模型
        if model_type == 'Ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'Lasso':
            model = Lasso(alpha=0.01)
        elif model_type == 'ElasticNet':
            model = ElasticNet(alpha=0.01, l1_ratio=0.8, max_iter=10000)
        elif model_type == 'SVR':
            model = SVR(kernel='linear', C=1.0)
        elif model_type == 'RandomForest':
            model = RandomForestRegressor(n_estimators=100)

        model.fit(X_tr, y_tr)

        # 抽取非零系数或特征重要性 > 0 的特征
        if hasattr(model, 'coef_'):
            sel = np.where(model.coef_ != 0)[0]
        elif hasattr(model, 'feature_importances_'):
            sel = np.where(model.feature_importances_ > 0)[0]
        else:
            sel = []

        counter.update(sel)

    # 二值化：每个特征在本文件里出现过就算 1
    df_freq = pd.DataFrame({
        'Feature_Index': np.arange(X_scaled.shape[1]),
        'Frequency': [1 if counter.get(i, 0) > 0 else 0
                      for i in range(X_scaled.shape[1])]
    })
    return df_freq

def merge_frequency_tables(files, model_type='ElasticNet', threshold=15):
    """
    合并多文件的二值化频率表，保存 Excel，
    返回 Total_Frequency >= threshold 的特征索引列表
    并打印筛选数量
    """
    tables = [process_file(fp, model_type) for fp in files]
    merged = pd.concat(tables, axis=1)

    # 找到所有的 Frequency 列
    freq_cols = [c for c in merged.columns if c.startswith('Frequency')]
    merged['Total_Frequency'] = merged[freq_cols].sum(axis=1)

    # 保存结果
    out = 'feature_selection_frequency_table_merged.xlsx'
    merged.to_excel(out, index=False)
    print(f"[{model_type}] 合并后的频率表已保存到 {out}")

    # 筛选
    sel = merged[merged['Total_Frequency'] >= threshold]
    idxs = sel['Feature_Index'].values

    # 打印统计信息
    print(f"阈值 = {threshold}，共筛选出 {len(idxs)} 个特征。")
    return idxs

if __name__ == '__main__':
    files = [
        r"C:\Users\kimber\PycharmProjects\预测模型完整版\data\训练集1.xlsx",
        r"C:\Users\kimber\PycharmProjects\预测模型完整版\data\训练集2.xlsx",
        r"C:\Users\kimber\PycharmProjects\预测模型完整版\data\训练集3.xlsx",
        r"C:\Users\kimber\PycharmProjects\预测模型完整版\data\训练集4.xlsx",
        r"C:\Users\kimber\PycharmProjects\预测模型完整版\data\训练集5.xlsx"
    ]

    model_list = ['ElasticNet']  # 可换成其他模型
    for m in model_list:
        try:
            idxs = merge_frequency_tables(files, model_type=m, threshold=15)

            print(f"模型 {m} 选中特征索引：", idxs[:,0])
        except Exception as e:
            print(f"模型 {m} 处理出错: {e}")
