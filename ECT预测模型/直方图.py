import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 读取 Excel 文件
file_path = 'D:\\研究生\\研0\\ECT精分预测\\数据\\精分\\小崔师姐\\置换检验结果\\output.xlsx'
df = pd.read_excel(file_path)

# 获取第一列的数据
column_name = df.columns[0]
data = df[column_name]

# 定义区间范围（间隔为0.05）
bins = np.arange(0, 1.05, 0.05)

# 计算置换检验的 P 值
threshold = 0.87
count_greater = np.sum(data > threshold)
p_value = count_greater / len(data)

# 绘制直方图
plt.figure(figsize=(8, 6))
plt.hist(data, bins=bins, edgecolor='black', color='#69b3a2', alpha=0.7)

# 在 0.8702 处画一条垂直线
plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Experiment Result (r={threshold:.2f})\nP-value={p_value:.3f}')

# 添加文本标注
plt.text(threshold, plt.ylim()[1] * 0.9, f'r={threshold:.2f}', color='red', ha='right', fontsize=10, weight='bold')

# 设置标题和标签
plt.title(f'Distribution of {column_name}', fontsize=14)
plt.xlabel(column_name, fontsize=12)
plt.ylabel('Count', fontsize=12)

# 添加图例
plt.legend()

# 保存图片到文件
save_path = f'D:\\研究生\\研0\\ECT精分预测\\数据\\精分\\小崔师姐\\置换检验结果\\{column_name}_distribution.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()  # 关闭图形，释放内存

print(f"图片已保存至: {save_path}")
