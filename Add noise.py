import pandas as pd
import numpy as np
def process_data(file_path, output_file_path, target_snr_db=20, beta=0.5):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 将dB转换为线性值

    target_snr_linear = 10 ** (target_snr_db / 10)


    keypoint_importance = {
        0: 1,  # 鼻子
        1: 1,  # 脖子
        2: 1,  # 右肩
        3: 1,  # 右肘
        4: 1,  # 右手腕
        5: 1,  # 左肩
        6: 1,  # 左肘
        7: 1,  # 左手腕
        8: 1,  # 右臀
        9: 1,  # 右膝
        10: 1,  # 右脚踝
        11: 1,  # 左臀
        12: 1,  # 左膝
        13: 1,  # 左脚踝
        14: 1,  # 右眼
        15: 1,  # 左眼
        16: 1,  # 右耳
        17: 1 } # 左耳
    def calculate_average_signal_power(x_coords, y_coords):
        x_coords = np.array(eval(x_coords))
        y_coords = np.array(eval(y_coords))
        signal_power_x = np.mean(x_coords ** 2)
        signal_power_y = np.mean(y_coords ** 2)
        return (signal_power_x + signal_power_y) / 2

    # 根据重要性级别计算噪声功率
    def calculate_noise_power(signal_power, importance, target_snr_linear, num_keypoints):
        lambd = keypoint_importance[importance]
        noise_power = signal_power / (target_snr_linear * lambd)
        return noise_power

    # 为不同的关键点添加不同程度的高斯噪声
    def add_noise_to_keypoints(x_coords, y_coords, num_keypoints, avg_signal_power):
        x_coords = np.array(eval(x_coords))
        y_coords = np.array(eval(y_coords))
        noisy_x_coords = []
        noisy_y_coords = []
        for i in range(num_keypoints):
            signal_power = (x_coords[i] ** 2 + y_coords[i] ** 2) / 2
            noise_power = calculate_noise_power(signal_power, i, target_snr_linear, num_keypoints)
            noise_x = np.random.normal(0, np.sqrt(noise_power))
            noise_y = np.random.normal(0, np.sqrt(noise_power))
            noisy_x_coords.append(x_coords[i] + noise_x)
            noisy_y_coords.append(y_coords[i] + noise_y)
        return noisy_x_coords, noisy_y_coords

    # 计算所有点的信号功率平均值
    num_keypoints = 18
    avg_signal_power = df.apply(lambda row: calculate_average_signal_power(row['keypoints_x'], row['keypoints_y']), axis=1).mean()

    # 应用不同程度的噪声到关键点
    noisy_keypoints = df.apply(lambda row: add_noise_to_keypoints(row['keypoints_x'], row['keypoints_y'], num_keypoints, avg_signal_power), axis=1)
    df['extreme_noisy_keypoints_x'], df['extreme_noisy_keypoints_y'] = zip(*noisy_keypoints)

    # 保存带有噪音的关键点数据
    df.to_csv(output_file_path, index=False)

    print(f"带有噪音的关键点数据已保存到 {output_file_path}")

    return output_file_path


file_path = '\\fasion-resize-annotation-test.csv'
output_path='\\fasion-resize-annotation-test.csv'

# for snr in range(10, -11, -2):
#     for beta in np.arange(1.0, -0.2, -0.2):
#         print(f"beta: {beta}, snr: {snr}")
snr = -10
process_data(file_path, output_path, snr)
