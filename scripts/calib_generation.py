import numpy as np
import argparse

def invert_transform_matrix(matrix_lines):
    """将T_cam_imu格式的4x4矩阵反转为T_imu_cam"""
    matrix = []
    for line in matrix_lines:
        matrix.extend(eval(line.strip().strip('- ')))
    T = np.array(matrix).reshape((4, 4))
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv.tolist()

def process_yaml(input_path, output_path):
    """处理YAML文件：转换T_cam_imu为T_imu_cam，并删除T_cn_cnm1"""
    with open(input_path, 'r') as f:
        lines = f.readlines()

    new_lines = ['%YAML:1.0\n\n']  # 添加开头标记
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'T_cam_imu:' in line:
            new_lines.append('  T_imu_cam:\n')
            matrix_lines = lines[i+1:i+5]
            T_inv = invert_transform_matrix(matrix_lines)
            for row in T_inv:
                new_lines.append('    - [' + ', '.join(f'{v:.16f}' for v in row) + ']\n')
            i += 5  # 跳过原矩阵
        elif 'T_cn_cnm1:' in line:
            i += 5  # 跳过整块
        else:
            new_lines.append(line)
            i += 1

    with open(output_path, 'w') as f:
        f.writelines(new_lines)
    print(f"✅ 转换完成，已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="将 YAML 文件中的 T_cam_imu 矩阵转换为 T_imu_cam 并删除 T_cn_cnm1")
    parser.add_argument('--input', '-i', type=str, required=True, help='输入 YAML 文件路径')
    parser.add_argument('--output', '-o', type=str, required=True, help='输出 YAML 文件路径')

    args = parser.parse_args()

    process_yaml(args.input, args.output)

if __name__ == "__main__":
    main()