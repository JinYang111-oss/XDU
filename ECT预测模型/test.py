import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
matplotlib.use('Agg')  # 或者使用 'TkAgg'，根据你的环境选择
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import LeaveOneOut, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from collections import Counter
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def test(feature_subset):
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
    r_value_zong = 0
    extra_p_value_zong = 0
    # 存储所有循环的结果
    results_df = pd.DataFrame(columns=['subject_id', 'y_true', 'y_pred', 'r_value', 'p_value'])

    for j in range(5):  # 遍历五个训练集和测试集
        # 读取数据集
        data = pd.read_excel(file_paths_train[j])
        data1 = pd.read_excel(file_paths_test[j])

        # 选择特征列
        feature_columns = data.columns[
                          data.columns.get_loc('TRT(min)'):data.columns.get_loc('O2_N2_SW_positivePeaks') + 1]
        data = data[['Person', 'PANSS(%)'] + list(feature_columns)]
        feature_columns1 = data1.columns[
                           data1.columns.get_loc('TRT(min)'):data1.columns.get_loc('O2_N2_SW_positivePeaks') + 1]
        data1 = data1[['Person', 'PANSS(%)'] + list(feature_columns1)]



        data = preprocess_data(data)

        data1 = preprocess_data(data1)

        # 4. 特征选择
        X = data.iloc[:, 2:].values  # T0 特征
        y = data['PANSS(%)'].values  # T0 FSNS

        X_1 = data1.iloc[:, 2:].values  # T0 特征
        y_1 = data1['PANSS(%)'].values  # T0 FSNS

        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_test = scaler.fit_transform(X_1)

        # LOOCV 特征选择
        loo = LeaveOneOut()
        X_top_subset = X_scaled[:, feature_subset]


        # 生成不同特征子集的组合

        mse_list = []
        mae_list = []
        r2_list = []
        y_true_list = []
        y_pred_list = []
        id_list = []  # 用于存储被试编号


        for train_index, test_index in loo.split(X_top_subset):
            X_train_loo, X_test_loo = X_top_subset[train_index], X_top_subset[test_index]
            y_train_loo, y_test_loo = y[train_index], y[test_index]

            # 存储被试编号
            id_list.append(data['Person'].values[test_index][0])

            svr_model = SVR(kernel='linear', C=1.0, epsilon=0.1, max_iter=10000)
            svr_model.fit(X_train_loo, y_train_loo)

            # 进行预测
            y_pred = svr_model.predict(X_test_loo.reshape(1, -1))  # reshape为2D数组
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

        X_top_subset = np.array(feature_subset, dtype=int)
        # 使用整数型索引来选择特征
        X_subset = X_scaled[:, X_top_subset]
        svr_model = SVR(kernel='linear', C=1.0, epsilon=0.1, max_iter=10000)
        svr_model.fit(X_subset, y)

        X_extra_top_features1 = X_test[:, X_top_subset]
        y_extra_pred = svr_model.predict(X_extra_top_features1)

        # 计算额外验证集的相关系数和p值
        extra_r_value, extra_p_value_current = stats.pearsonr(y_1, y_extra_pred)
        # 将每个循环的实际值和预测值存入表格
        for i in range(len(y_1)):
            temp_df = pd.DataFrame({
                'subject_id': [data['Person'].values[i]],
                'y_true': [y_1[i]],
                'y_pred': [y_extra_pred[i]],
                'r_value': [extra_r_value],
                'p_value': [extra_p_value_current]
            })

            # 使用 pd.concat() 代替 append
            results_df = pd.concat([results_df, temp_df], ignore_index=True)

        # 打印当前循环的结果
        print(f'Round {j + 1} - r_value: {extra_r_value}, p_value: {extra_p_value_current}')

    # 计算所有循环的总 r 值和 p 值
    total_r_value, total_p_value = stats.pearsonr(results_df['y_true'], results_df['y_pred'])
    # 保存结果到文件
    results_df.to_excel('svr_predictions_results.xlsx', index=False)


    return total_r_value, total_p_value





# if __name__ == '__main__':
#     feature_subset = [268, 593, 145, 92, 786, 27, 515, 469, 133, 364, 906, 2, 604, 272]
#     total_r_value, total_p_value = test(feature_subset)
#     print(f'Total r_value: {total_r_value}, p_value: {total_p_value}')
