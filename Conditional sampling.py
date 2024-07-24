import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import mean_squared_error, mean_absolute_error


def get_known_indices(extreme_noisy_keypoints_x, extreme_noisy_keypoints_y,threshold=256):
    known_indices = []
    for i, (x, y) in enumerate(zip(extreme_noisy_keypoints_x, extreme_noisy_keypoints_y)):
        if 0 <= x <= threshold and 0 <= y <= threshold:
            known_indices.append(i)
    return known_indices


def evaluate_keypoints(row, known_indices, mean_vector, cov_matrix):
    # 直接使用已经解析的 keypoints_x 和 keypoints_y
    keypoints_x = np.array(row['keypoints_x'])
    keypoints_y = np.array(row['keypoints_y'])

    # 获取当前行的extreme_noisy_keypoints_x 和extreme_noisy_keypoints_y
    extreme_noisy_keypoints_x = np.array(row['extreme_noisy_keypoints_x'])
    extreme_noisy_keypoints_y = np.array(row['extreme_noisy_keypoints_y'])

    # 提取已知位置的坐标
    known_values_x = np.array([extreme_noisy_keypoints_x[i] for i in known_indices])
    known_values_y = np.array([extreme_noisy_keypoints_y[i] for i in known_indices])
    known_values = np.hstack([known_values_x, known_values_y])

    if len(known_indices) == 18:
        # 如果所有关键点都已知，直接返回 extreme_noisy_keypoints
        restored_keypoints_x = extreme_noisy_keypoints_x
        restored_keypoints_y = extreme_noisy_keypoints_y
    else:
        # 选择未知的索引
        unknown_indices = list(set(range(18)) - set(known_indices))

        # 将已知和未知的索引扩展到 x 和 y 的范围
        known_indices_full = known_indices + [i + 18 for i in known_indices]
        unknown_indices_full = unknown_indices + [i + 18 for i in unknown_indices]

        # 分离已知和未知的均值向量和协方差矩阵
        mean_known = mean_vector[known_indices_full]
        mean_unknown = mean_vector[unknown_indices_full]

        cov_known_known = cov_matrix[np.ix_(known_indices_full, known_indices_full)]
        cov_known_unknown = cov_matrix[np.ix_(known_indices_full, unknown_indices_full)]
        cov_unknown_known = cov_matrix[np.ix_(unknown_indices_full, known_indices_full)]
        cov_unknown_unknown = cov_matrix[np.ix_(unknown_indices_full, unknown_indices_full)]

        # 计算条件均值和协方差矩阵
        conditional_mean = mean_unknown + cov_unknown_known @ np.linalg.inv(cov_known_known) @ (
                    known_values - mean_known)
        conditional_cov = cov_unknown_unknown - cov_unknown_known @ np.linalg.inv(cov_known_known) @ cov_known_unknown

        # 进行条件采样
        sampled_values = multivariate_normal(conditional_mean, conditional_cov).rvs()

        # 组合已知值和采样值恢复完整的骨骼点坐标
        restored_keypoints_x = np.zeros(18)
        restored_keypoints_y = np.zeros(18)
        for i, idx in enumerate(known_indices):
            restored_keypoints_x[idx] = known_values_x[i]
            restored_keypoints_y[idx] = known_values_y[i]
        for i, idx in enumerate(unknown_indices):
            restored_keypoints_x[idx] = sampled_values[i]
            restored_keypoints_y[idx] = sampled_values[i + len(unknown_indices)]

    # 计算误差
    original_keypoints_x = keypoints_x
    original_keypoints_y = keypoints_y

    # 计算均方根误差 (RMSE)
    rmse_x = np.sqrt(mean_squared_error(original_keypoints_x, restored_keypoints_x))
    rmse_y = np.sqrt(mean_squared_error(original_keypoints_y, restored_keypoints_y))

    # 计算平均绝对误差 (MAE)
    mae_x = mean_absolute_error(original_keypoints_x, restored_keypoints_x)
    mae_y = mean_absolute_error(original_keypoints_y, restored_keypoints_y)

    # 计算总的 RMSE 和 MAE
    total_rmse = np.sqrt((rmse_x ** 2 + rmse_y ** 2) / 2)
    total_mae = (mae_x + mae_y) / 2

    return restored_keypoints_x, restored_keypoints_y, total_rmse, total_mae

# 示例文件路径和阈值
input_file_path = 'fasion-resize-annotation-test.csv'
output_file_path = 'fasion-resize-annotation-test.csv'
threshold = 255

# 读取CSV文件
df = pd.read_csv(input_file_path)

# 解析extreme_noisy_keypoints_x和extreme_noisy_keypoints_y
df['extreme_noisy_keypoints_x'] = df['extreme_noisy_keypoints_x'].apply(
    lambda x: list(map(float, x.strip('[]').split(','))))
df['extreme_noisy_keypoints_y'] = df['extreme_noisy_keypoints_y'].apply(
    lambda x: list(map(float, x.strip('[]').split(','))))

# 解析 keypoints_x 和 keypoints_y
df['keypoints_x'] = df['keypoints_x'].apply(lambda x: list(map(float, x.strip('[]').split(','))))
df['keypoints_y'] = df['keypoints_y'].apply(lambda x: list(map(float, x.strip('[]').split(','))))

# 将数据按关键点位置分组
keypoints_x = np.array(df['keypoints_x'].tolist())
keypoints_y = np.array(df['keypoints_y'].tolist())

# 构建包含所有关键点位置的完整数据矩阵
all_keypoints = np.hstack([keypoints_x, keypoints_y])

# 计算总体均值向量和协方差矩阵
mean_vector = np.mean(all_keypoints, axis=0)
cov_matrix = np.cov(all_keypoints, rowvar=False)

# 初始化误差的累加器
total_rmse_list = []
total_mae_list = []

# 遍历每一行数据
for index, row in df.iterrows():
    extreme_noisy_keypoints_x = row['extreme_noisy_keypoints_x']
    extreme_noisy_keypoints_y = row['extreme_noisy_keypoints_y']

    # 获取known_indices
    known_indices = get_known_indices(extreme_noisy_keypoints_x, extreme_noisy_keypoints_y, threshold)

    # 调用evaluate_keypoints对这一行进行采样
    restored_keypoints_x, restored_keypoints_y, total_rmse, total_mae = evaluate_keypoints(row, known_indices, mean_vector, cov_matrix)

    # 保存恢复的关键点坐标
    df.at[index, 'restored_keypoints_x'] = str(restored_keypoints_x.tolist())
    df.at[index, 'restored_keypoints_y'] = str(restored_keypoints_y.tolist())

    # 保存当前行的 RMSE 和 MAE
    total_rmse_list.append(total_rmse)
    total_mae_list.append(total_mae)

# 计算所有行的平均 RMSE 和 MAE
average_rmse = np.mean(total_rmse_list)
average_mae = np.mean(total_mae_list)

print(f"平均 RMSE: {average_rmse}")
print(f"平均 MAE: {average_mae}")

# 保存更新后的注释文件
df.to_csv(output_file_path, index=False)