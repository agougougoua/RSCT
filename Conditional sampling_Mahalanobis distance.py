import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_known_indices(extreme_noisy_keypoints_x, extreme_noisy_keypoints_y, mean_vector, cov_matrix, threshold=0.5):
    known_indices = []
    all_keypoints = np.hstack([extreme_noisy_keypoints_x, extreme_noisy_keypoints_y])
    for i in range(18):
        point = np.array([extreme_noisy_keypoints_x[i], extreme_noisy_keypoints_y[i]])
        mean = mean_vector[i], mean_vector[i + 18]
        cov = cov_matrix[i:i+2, i:i+2]
        mahalanobis_distance = np.sqrt((point - mean).T @ np.linalg.inv(cov) @ (point - mean))
        if mahalanobis_distance < threshold:
            known_indices.append(i)
    return known_indices

def evaluate_keypoints(row, known_indices, mean_vector, cov_matrix):
    keypoints_x = np.array(row['keypoints_x'])
    keypoints_y = np.array(row['keypoints_y'])
    extreme_noisy_keypoints_x = np.array(row['extreme_noisy_keypoints_x'])
    extreme_noisy_keypoints_y = np.array(row['extreme_noisy_keypoints_y'])

    known_values_x = np.array([extreme_noisy_keypoints_x[i] for i in known_indices])
    known_values_y = np.array([extreme_noisy_keypoints_y[i] for i in known_indices])
    known_values = np.hstack([known_values_x, known_values_y])

    if len(known_indices) == 18:
        restored_keypoints_x = extreme_noisy_keypoints_x
        restored_keypoints_y = extreme_noisy_keypoints_y
    else:
        unknown_indices = list(set(range(18)) - set(known_indices))
        known_indices_full = known_indices + [i + 18 for i in known_indices]
        unknown_indices_full = unknown_indices + [i + 18 for i in unknown_indices]

        mean_known = mean_vector[known_indices_full]
        mean_unknown = mean_vector[unknown_indices_full]

        cov_known_known = cov_matrix[np.ix_(known_indices_full, known_indices_full)]
        cov_known_unknown = cov_matrix[np.ix_(known_indices_full, unknown_indices_full)]
        cov_unknown_known = cov_matrix[np.ix_(unknown_indices_full, known_indices_full)]
        cov_unknown_unknown = cov_matrix[np.ix_(unknown_indices_full, unknown_indices_full)]

        conditional_mean = mean_unknown + cov_unknown_known @ np.linalg.inv(cov_known_known) @ (known_values - mean_known)
        conditional_cov = cov_unknown_unknown - cov_unknown_known @ np.linalg.inv(cov_known_known) @ cov_known_unknown

        sampled_values = multivariate_normal(conditional_mean, conditional_cov).rvs()

        restored_keypoints_x = np.zeros(18)
        restored_keypoints_y = np.zeros(18)
        for i, idx in enumerate(known_indices):
            restored_keypoints_x[idx] = known_values_x[i]
            restored_keypoints_y[idx] = known_values_y[i]
        for i, idx in enumerate(unknown_indices):
            restored_keypoints_x[idx] = sampled_values[i]
            restored_keypoints_y[idx] = sampled_values[i + len(unknown_indices)]

    original_keypoints_x = keypoints_x
    original_keypoints_y = keypoints_y

    rmse_x = np.sqrt(mean_squared_error(original_keypoints_x, restored_keypoints_x))
    rmse_y = np.sqrt(mean_squared_error(original_keypoints_y, restored_keypoints_y))

    mae_x = mean_absolute_error(original_keypoints_x, restored_keypoints_x)
    mae_y = mean_absolute_error(original_keypoints_y, restored_keypoints_y)

    total_rmse = np.sqrt((rmse_x ** 2 + rmse_y ** 2) / 2)
    total_mae = (mae_x + mae_y) / 2

    return restored_keypoints_x, restored_keypoints_y, total_rmse, total_mae


input_file_path = 'fasion-resize-annotation-test.csv'
output_file_path = 'fasion-resize-annotation-test.csv'
threshold = 3.03

df = pd.read_csv(input_file_path)

df['extreme_noisy_keypoints_x'] = df['extreme_noisy_keypoints_x'].apply(lambda x: list(map(float, x.strip('[]').split(','))))
df['extreme_noisy_keypoints_y'] = df['extreme_noisy_keypoints_y'].apply(lambda x: list(map(float, x.strip('[]').split(','))))

df['keypoints_x'] = df['keypoints_x'].apply(lambda x: list(map(float, x.strip('[]').split(','))))
df['keypoints_y'] = df['keypoints_y'].apply(lambda x: list(map(float, x.strip('[]').split(','))))

keypoints_x = np.array(df['keypoints_x'].tolist())
keypoints_y = np.array(df['keypoints_y'].tolist())

all_keypoints = np.hstack([keypoints_x, keypoints_y])

mean_vector = np.mean(all_keypoints, axis=0)
cov_matrix = np.cov(all_keypoints, rowvar=False)

total_rmse_list = []
total_mae_list = []

for index, row in df.iterrows():
    extreme_noisy_keypoints_x = row['extreme_noisy_keypoints_x']
    extreme_noisy_keypoints_y = row['extreme_noisy_keypoints_y']

    known_indices = get_known_indices(extreme_noisy_keypoints_x, extreme_noisy_keypoints_y, mean_vector, cov_matrix, threshold)
    print(known_indices)

    restored_keypoints_x, restored_keypoints_y, total_rmse, total_mae = evaluate_keypoints(row, known_indices, mean_vector, cov_matrix)

    df.at[index, 'restored_keypoints_x'] = str(restored_keypoints_x.tolist())
    df.at[index, 'restored_keypoints_y'] = str(restored_keypoints_y.tolist())

    total_rmse_list.append(total_rmse)
    total_mae_list.append(total_mae)

average_rmse = np.mean(total_rmse_list)
average_mae = np.mean(total_mae_list)

print(f"平均 RMSE: {average_rmse}")
print(f"平均 MAE: {average_mae}")

df.to_csv(output_file_path, index=False)
