import os
import pandas as pd
from sklearn.model_selection import KFold
from merge_frequency_tables import merge_frequency_tables


def process_and_merge():
    # 读取动态生成的 Excel 文件
    df = pd.read_excel(f"./ECTdata.xlsx")

    # 获取被试编号
    subject_ids = df['Person'].unique()

    # 创建 KFold 对象，分成 5 折
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 用于存储每个折次的被试编号
    folds = {i: [] for i in range(5)}

    # 根据被试编号分配折次
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(subject_ids)):
        test_subjects = subject_ids[test_idx]

        # 将这些被试编号对应的数据分配到相应的折次
        for subj_id in test_subjects:
            # 获取对应被试编号的数据
            subj_data = df[df['Person'] == subj_id]
            folds[fold_idx].append(subj_data)

    # 创建保存文件的文件夹
    output_dir = f"data"
    os.makedirs(output_dir, exist_ok=True)

    # 循环五次，将每一折的数据保存为训练集和测试集
    for fold_idx in range(5):
        # 获取当前折次的测试集
        test_subjects = folds[fold_idx]
        test_data = pd.concat(test_subjects)

        # 获取剩余的训练集
        train_data = pd.concat([pd.concat(folds[i]) for i in range(5) if i != fold_idx])

        # 保存训练集和测试集为 Excel 文件
        train_file = os.path.join(output_dir, f"训练集{fold_idx + 1}.xlsx")
        test_file = os.path.join(output_dir, f"测试集{fold_idx + 1}.xlsx")

        # 保存为 Excel 文件
        train_data.to_excel(train_file, index=False)
        test_data.to_excel(test_file, index=False)


    # 读取所有的训练集文件路径
    files = [
        f"./data/训练集1.xlsx",
        f"./data/训练集2.xlsx",
        f"./data/训练集3.xlsx",
        f"./data/训练集4.xlsx",
        f"./data/训练集5.xlsx"
    ]


    # 合并并保存频率表
    first_column_values = merge_frequency_tables(files)
    print(first_column_values)
    return (first_column_values)


if __name__ == "__main__":
    process_and_merge()
