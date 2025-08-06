import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 或者使用 'TkAgg'，根据你的环境选择
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
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

    # 选择最重要的特征索引
    all_selected_indices = process_and_merge()
    # 读取数据文件路径
    file_paths_train = [
        f"./data/训练集1.xlsx",
        f"./data/训练集2.xlsx",
        f"./data/训练集3.xlsx",
        f"./data/训练集4.xlsx",
        f"./data/训练集5.xlsx"
    ]

    file_paths_test = [
        f"./data/测试集1.xlsx",
        f"./data/测试集2.xlsx",
        f"./data/测试集3.xlsx",
        f"./data/测试集4.xlsx",
        f"./data/测试集5.xlsx"
    ]

    r_value_dict = {}

    # 计算所有单独特征的r_value_zong
    for feature in all_selected_indices:
        feature_subset = [feature]  # 当前特征子集
        print(feature_subset)
        r_value_zong = 0

        for j in range(5):  # 遍历五个训练集和测试集
            # 读取数据集
            train_data = pd.read_excel(file_paths_train[j])
            test_data = pd.read_excel(file_paths_test[j])

            # 选择特征和目标变量
            feature_columns = train_data.columns[
                              train_data.columns.get_loc('TRT(min)'):train_data.columns.get_loc('O2_N2_SW_positivePeaks') + 1]

            train_data = train_data[['Person', 'PANSS(%)'] + list(feature_columns)]
            train_data = preprocess_data(train_data)

            # 选择特征和目标变量
            feature_columns1 = test_data.columns[
                               test_data.columns.get_loc('TRT(min)'):test_data.columns.get_loc('O2_N2_SW_positivePeaks') + 1]

            test_data = test_data[['Person', 'PANSS(%)'] + list(feature_columns1)]
            test_data = preprocess_data(test_data)

            # 提取特征和标签
            X_train = train_data.iloc[:, 2:].values  # T0 特征
            y_train = train_data['PANSS(%)'].values  # T0 FSNS
            X_test = test_data.iloc[:, 2:].values  # T0 特征
            y_test = test_data['PANSS(%)'].values  # T0 FSNS

            # 使用选择的特征子集
            X_train_subset = X_train[:, feature_subset]
            X_test_subset = X_test[:, feature_subset]

            # 标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_subset)
            X_test_scaled = scaler.transform(X_test_subset)

            id_list = []  # 用于存储被试编号
            y_true_list = []
            y_pred_list = []
            mse_list = []
            mae_list = []
            r2_list = []
            y_true_list = []
            y_pred_list = []
            id_list = []  # 用于存储被试编号
            extra_mse = []
            extra_r2 = []
            extra_mae = []
            extra_r_value = []
            extra_p_value = []
            loo = LeaveOneOut()

            for train_index, test_index in loo.split(X_train_scaled):
                X_train_loo, X_test_loo = X_train_scaled[train_index], X_train_scaled[test_index]
                y_train_loo, y_test_loo = y_train[train_index], y_train[test_index]

                # 存储被试编号
                id_list.append(train_data['Person'].values[test_index][0])

                # 使用Ridge回归模型
                ridge_model = Ridge(alpha=1.0)
                ridge_model.fit(X_train_loo, y_train_loo)

                # 进行预测
                y_pred = ridge_model.predict(X_test_loo.reshape(1, -1))  # reshape为2D数组
                y_true_list.append(y_test_loo[0])
                y_pred_list.append(y_pred[0])

                # 记录性能指标
                mse_list.append(mean_squared_error(y_test_loo, y_pred))
                mae_list.append(mean_absolute_error(y_test_loo, y_pred))

                # 计算 R² 仅在样本数大于 1 的情况下
                if len(y_test_loo) > 1:
                    r2 = r2_score(y_test_loo, y_pred)
                    r2_list.append(r2)
                else:
                    r2_list.append(np.nan)  # 或者可以添加0，视情况而定

            # 计算平均性能指标
            mean_mse = np.mean(mse_list)
            mean_r2 = np.nanmean(r2_list) if np.any(~np.isnan(r2_list)) else 0  # 检查是否有有效值
            mean_mae = np.mean(mae_list)

            # 计算相关系数和p值
            r_value, p_value = stats.pearsonr(y_true_list, y_pred_list)
            r_value_zong += r_value

        r_value_zong = r_value_zong / 5
        print(r_value_zong)
        r_value_dict[tuple(feature_subset)] = r_value_zong
        print(r_value_dict)

    # 选择r_value_zong最高的特征作为第一个子集
    best_feature_subset = [max(r_value_dict, key=r_value_dict.get)]
    r_value_zong_max = r_value_dict[best_feature_subset[0]]
    print(best_feature_subset)
    print(r_value_zong_max)

    # 展平 best_feature_subset 中的元组，提取出整数索引
    best_feature_subset_flat = [f[0] if isinstance(f, tuple) else f for f in best_feature_subset]

    # 删除 best_feature_subset 中已包含的特征
    remaining_features = [f for f in all_selected_indices if f not in best_feature_subset_flat]
    best_r_value_zong = r_value_zong_max
    best_next_feature = None
    # 开始逐步加入其他特征，计算r_value_zong
    while remaining_features:
        for feature in remaining_features:
            # 新的特征子集
            current_subset = best_feature_subset_flat + [feature]

            # 计算当前特征子集的r_value_zong
            r_value_zong = 0
            for j in range(5):  # 遍历五个训练集和测试集
                # 数据预处理、特征选择等步骤保持不变
                train_data = pd.read_excel(file_paths_train[j])
                test_data = pd.read_excel(file_paths_test[j])

                # 选择特征和目标变量
                feature_columns = train_data.columns[
                                  train_data.columns.get_loc('TRT(min)'):train_data.columns.get_loc(
                                      'O2_N2_SW_positivePeaks') + 1]

                train_data = train_data[['Person', 'PANSS(%)'] + list(feature_columns)]
                train_data = preprocess_data(train_data)

                # 选择特征和目标变量
                feature_columns1 = test_data.columns[
                                   test_data.columns.get_loc('TRT(min)'):test_data.columns.get_loc(
                                       'O2_N2_SW_positivePeaks') + 1]

                test_data = test_data[['Person', 'PANSS(%)'] + list(feature_columns1)]
                test_data = preprocess_data(test_data)

                # 提取特征和标签
                X_train = train_data.iloc[:, 2:].values  # T0 特征
                y_train = train_data['PANSS(%)'].values  # T0 FSNS
                X_test = test_data.iloc[:, 2:].values  # T0 特征
                y_test = test_data['PANSS(%)'].values  # T0 FSNS

                # 使用选择的特征子集
                X_train_subset = X_train[:, current_subset]
                X_test_subset = X_test[:, current_subset]

                # 标准化特征
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_subset)
                X_test_scaled = scaler.transform(X_test_subset)

                id_list = []  # 用于存储被试编号
                y_true_list = []
                y_pred_list = []
                loo = LeaveOneOut()

                for train_index, test_index in loo.split(X_train_scaled):
                    X_train_loo, X_test_loo = X_train_scaled[train_index], X_train_scaled[test_index]
                    y_train_loo, y_test_loo = y_train[train_index], y_train[test_index]

                    # 存储被试编号
                    id_list.append(train_data['Person'].values[test_index][0])

                    # 使用Ridge回归模型
                    ridge_model = Ridge(alpha=1.0)
                    ridge_model.fit(X_train_loo, y_train_loo)

                    # 进行预测
                    y_pred = ridge_model.predict(X_test_loo.reshape(1, -1))  # reshape为2D数组
                    y_true_list.append(y_test_loo[0])
                    y_pred_list.append(y_pred[0])

                    # 记录性能指标
                    mse_list.append(mean_squared_error(y_test_loo, y_pred))
                    mae_list.append(mean_absolute_error(y_test_loo, y_pred))

                    # 计算 R² 仅在样本数大于 1 的情况下
                    if len(y_test_loo) > 1:
                        r2 = r2_score(y_test_loo, y_pred)
                        r2_list.append(r2)
                    else:
                        r2_list.append(np.nan)  # 或者可以添加0，视情况而定

                # 计算平均性能指标
                mean_mse = np.mean(mse_list)
                mean_r2 = np.nanmean(r2_list) if np.any(~np.isnan(r2_list)) else 0  # 检查是否有有效值
                mean_mae = np.mean(mae_list)

                # 计算相关系数和p值
                r_value, p_value = stats.pearsonr(y_true_list, y_pred_list)
                r_value_zong += r_value

            r_value_zong = r_value_zong / 5
            print(f"特征子集: {current_subset}, r_value_zong: {r_value_zong}")

            if r_value_zong > best_r_value_zong:
                best_r_value_zong = r_value_zong
                best_next_feature = feature

        if best_next_feature is not None:
            if best_next_feature in remaining_features:
                best_feature_subset_flat.append(best_next_feature)
                remaining_features.remove(best_next_feature)
                print(f"更新特征子集: {best_feature_subset_flat}, r_value_zong: {best_r_value_zong}")
            else:
                print(f"特征 {best_next_feature} 不在 remaining_features 中，跳出循环")
                break  # 跳出循环

    # 最终最优特征子集
    print("最优特征子集:", best_feature_subset_flat)
    print("最终r_value_zong:", best_r_value_zong)

    return best_feature_subset_flat, best_r_value_zong

def main():

    best_feature_subset, best_r_value_zong = find_best_features()
    print("最优特征子集:", best_feature_subset)
    print("最终r_value_zong:", best_r_value_zong)
    total_r_value, total_p_value = test(best_feature_subset)
    print(f'Total r_value: {total_r_value}, p_value: {total_p_value}')

if __name__ == '__main__':
    main()  # 调用main函数