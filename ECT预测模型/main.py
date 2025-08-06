import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 根据你的环境也可以使用 'TkAgg'
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from read import process_and_merge
from test import test


# 预处理数据函数
def preprocess_data(df):
    df.fillna(df.mean(), inplace=True)  # 填充缺失值
    for column in df.columns[2:]:  # 假设前两列是非特征列
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), df[column].mean(), df[column])
    return df


def find_best_features():
    # 从 process_and_merge 获取所有候选特征索引（二维，形状为 [n_features, n_folds]）
    all_selected_indices = process_and_merge()
    # 提取第 1 列（或者所有列相同，因此取第一列）作为整数索引列表
    all_selected_indices = np.array(all_selected_indices)
    all_selected_indices = all_selected_indices[:, 0].astype(int).tolist()

    # 训练集和测试集文件路径列表
    file_paths_train = [f"./data/训练集{i + 1}.xlsx" for i in range(5)]
    file_paths_test = [f"./data/测试集{i + 1}.xlsx" for i in range(5)]

    r_value_dict = {}

    # 计算每个单独特征的平均 r 值
    for feature in all_selected_indices:
        feature_subset = [feature]
        r_value_zong = 0.0

        for j in range(5):
            # 读取并预处理训练集和测试集
            train_data = pd.read_excel(file_paths_train[j])
            test_data = pd.read_excel(file_paths_test[j])

            train_data = train_data[['Person', 'PANSS(%)'] + list(
                train_data.columns[train_data.columns.get_loc('TRT(min)'):
                                   train_data.columns.get_loc('O2_N2_SW_positivePeaks') + 1]
            )]
            test_data = test_data[['Person', 'PANSS(%)'] + list(
                test_data.columns[test_data.columns.get_loc('TRT(min)'):
                                  test_data.columns.get_loc('O2_N2_SW_positivePeaks') + 1]
            )]

            train_data = preprocess_data(train_data)
            test_data = preprocess_data(test_data)

            X_train = train_data.iloc[:, 2:].values
            y_train = train_data['PANSS(%)'].values
            X_test = test_data.iloc[:, 2:].values
            y_test = test_data['PANSS(%)'].values

            # 只选择当前特征子集
            X_train_subset = X_train[:, feature_subset]
            X_test_subset = X_test[:, feature_subset]

            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_subset)
            X_test_scaled = scaler.transform(X_test_subset)

            y_true_list = []
            y_pred_list = []
            loo = LeaveOneOut()

            for train_index, val_index in loo.split(X_train_scaled):
                X_tr, X_val = X_train_scaled[train_index], X_train_scaled[val_index]
                y_tr, y_val = y_train[train_index], y_train[val_index]

                svr = SVR(kernel='linear', C=1.0, epsilon=0.1, max_iter=10000)
                svr.fit(X_tr, y_tr)
                y_pred = svr.predict(X_val.reshape(1, -1))

                y_true_list.append(y_val[0])
                y_pred_list.append(y_pred[0])

            # 计算 Pearson 相关系数
            r_value, _ = stats.pearsonr(y_true_list, y_pred_list)
            r_value_zong += r_value

        # 平均 r 值
        r_value_zong /= 5.0
        r_value_dict[(feature,)] = r_value_zong

    # 找到最佳单特征子集
    best_feature = max(r_value_dict, key=r_value_dict.get)[0]
    best_r = r_value_dict[(best_feature,)]

    # 逐步向子集中加入其他特征
    best_subset = [best_feature]
    remaining = [f for f in all_selected_indices if f not in best_subset]
    best_r_zong = best_r
    updated = True

    while updated and remaining:
        updated = False
        temp_best = None
        for feature in remaining:
            current = best_subset + [feature]
            r_sum = 0.0

            for j in range(5):
                train_data = pd.read_excel(file_paths_train[j])
                test_data = pd.read_excel(file_paths_test[j])

                train_data = train_data[['Person', 'PANSS(%)'] + list(
                    train_data.columns[train_data.columns.get_loc('TRT(min)'):
                                       train_data.columns.get_loc('O2_N2_SW_positivePeaks') + 1]
                )]
                test_data = test_data[['Person', 'PANSS(%)'] + list(
                    test_data.columns[test_data.columns.get_loc('TRT(min)'):
                                      test_data.columns.get_loc('O2_N2_SW_positivePeaks') + 1]
                )]
                train_data = preprocess_data(train_data)
                test_data = preprocess_data(test_data)

                X_train = train_data.iloc[:, 2:].values[:, current]
                y_train = train_data['PANSS(%)'].values
                X_train_scaled = StandardScaler().fit_transform(X_train)

                y_true_list = []
                y_pred_list = []
                for idx_tr, idx_val in LeaveOneOut().split(X_train_scaled):
                    model = SVR(kernel='linear', C=1.0, epsilon=0.1, max_iter=10000)
                    model.fit(X_train_scaled[idx_tr], y_train[idx_tr])
                    y_pred = model.predict(X_train_scaled[idx_val].reshape(1, -1))
                    y_true_list.append(y_train[idx_val][0])
                    y_pred_list.append(y_pred[0])

                r_val, _ = stats.pearsonr(y_true_list, y_pred_list)
                r_sum += r_val

            r_avg = r_sum / 5.0
            if r_avg > best_r_zong:
                best_r_zong = r_avg
                temp_best = feature
                updated = True

        if updated and temp_best is not None:
            best_subset.append(temp_best)
            remaining.remove(temp_best)

    return best_subset, best_r_zong


def main():
    best_features, best_r = find_best_features()
    print("最优特征子集:", best_features)
    print("最终平均 r 值:", best_r)
    total_r, total_p = test(best_features)
    print(f"Total r_value: {total_r}, p_value: {total_p}")


if __name__ == '__main__':
    main()
