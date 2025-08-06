#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ridge 模型超参数 alpha 调优脚本
1. 合并 5 折训练集
2. 预处理缺失值和异常值
3. 使用 LeaveOneOut + GridSearchCV 搜索最佳 alpha
4. 自定义 Pearson 相关系数做为 scoring
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import make_scorer
from scipy import stats

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """缺失值填充与异常值处理"""
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

def pearson_r(y_true, y_pred):
    """自定义 Pearson 相关系数评分函数"""
    # 对于常数序列 pearsonr 会报错，这里捕获并返回 0
    try:
        return stats.pearsonr(y_true, y_pred)[0]
    except Exception:
        return 0.0

def load_and_prepare():
    """加载并合并 5 折训练集，返回标准化后的 X_all, y_all"""
    file_paths = [f"./data/训练集{i}.xlsx" for i in range(1, 6)]
    dfs = []
    for path in file_paths:
        df = pd.read_excel(path)
        # 假设特征列从 'TRT(min)' 到 'O2_N2_SW_positivePeaks'
        feats = df.columns[df.columns.get_loc('TRT(min)'): df.columns.get_loc('O2_N2_SW_positivePeaks') + 1]
        sub = df[['Person', 'PANSS(%)'] + list(feats)]
        dfs.append(sub)
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = preprocess_data(df_all)

    X = df_all.iloc[:, 2:].values
    y = df_all['PANSS(%)'].values
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, y

def tune_ridge_alpha(X, y):
    """使用 LOOCV + GridSearchCV 调优 Ridge 的 alpha"""
    # 候选 alpha 范围
    alphas = np.logspace(-10, 10, 13)
    param_grid = {'alpha': alphas}
    ridge = Ridge()

    loo = LeaveOneOut()
    scorer = make_scorer(pearson_r)

    grid = GridSearchCV(
        estimator=ridge,
        param_grid=param_grid,
        cv=loo,
        scoring=scorer,
        n_jobs=-1,        # 并行加速
        verbose=1
    )
    grid.fit(X, y)
    return grid.best_params_['alpha'], grid.best_score_

if __name__ == '__main__':
    print("加载并准备数据…")
    X_all, y_all = load_and_prepare()

    print("开始调参 Ridge.alpha …")
    best_alpha, best_score = tune_ridge_alpha(X_all, y_all)

    print(f"\n调参完成！最佳 alpha = {best_alpha}")
    print(f"对应的平均 Pearson r (LOO‐CV) = {best_score:.4f}")
