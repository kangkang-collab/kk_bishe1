import os
import numpy as np
import torch

data_root = "/home/itslab/kk/SparseDrive_first/data/nuscenesmini/"
lidar_dir = os.path.join(data_root, "samples/LIDAR_TOP")

# 列出前 5 个 bin 文件
lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".bin")])[:5]

for i, f in enumerate(lidar_files):
    file_path = os.path.join(lidar_dir, f)
    print(f"\n[{i}] lidar_path: {file_path}")

    if not os.path.exists(file_path):
        print("  ❌ 文件不存在")
        continue

    # 读取 bin 文件
    try:
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)
        print("  ✅ points.shape:", points.shape)

        # 转为 torch.Tensor
        points_tensor = torch.from_numpy(points)
        print("  ✅ torch tensor shape:", points_tensor.shape)
    except Exception as e:
        print("  ❌ 读取或 reshape 出错:", e)
