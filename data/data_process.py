import os
import numpy as np
import random
import open3d as o3d

# =========================
# 路径配置
# =========================
input_dir = "/chenyiming/projects/PoinTr/data/strawberry_pointcloud_2048"
save_root = "/chenyiming/projects/PoinTr/data/strawberry"

train_ratio = 0.8
seed = 42

# =========================
# HPR 单视角
# =========================
def hidden_point_removal(pc, camera=[0, 0, 2], radius=100):
    """
    pc: (N,3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    _, pt_map = pcd.hidden_point_removal(camera, radius)
    visible = np.asarray(pcd.points)[pt_map]

    return visible


# =========================
# 统一归一化（关键！！）
# =========================
def normalize_pair(gt, partial):
    center = np.mean(gt, axis=0)
    scale = np.max(np.linalg.norm(gt - center, axis=1))

    gt = (gt - center) / scale
    partial = (partial - center) / scale

    return gt, partial


# =========================
# 读取 ply
# =========================
def load_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points).astype(np.float32)


# =========================
# 采样函数
# =========================
def random_sample(pc, n):
    N = pc.shape[0]

    if N == n:
        return pc
    elif N > n:
        idx = np.random.choice(N, n, replace=False)
        return pc[idx]
    else:
        idx = np.random.choice(N, n - N, replace=True)
        return np.concatenate([pc, pc[idx]], axis=0)


# =========================
# 获取文件列表
# =========================
names = [f"strawberry{str(i).zfill(3)}" for i in range(1, 401)]

random.seed(seed)
random.shuffle(names)

split_idx = int(len(names) * train_ratio)
train_names = names[:split_idx]
val_names = names[split_idx:]

print(f"Train: {len(train_names)}, Val: {len(val_names)}")


# =========================
# 创建目录
# =========================
os.makedirs(os.path.join(save_root, "train"), exist_ok=True)
os.makedirs(os.path.join(save_root, "val"), exist_ok=True)


# =========================
# 主处理函数
# =========================
def process_split(name_list, split):
    for idx, name in enumerate(name_list):
        ply_path = os.path.join(input_dir, f"{name}.ply")

        # ===== 读取完整点云 =====
        gt = load_ply(ply_path)

        # ===== HPR生成partial =====
        partial = hidden_point_removal(gt)

        # ===== 采样 =====
        partial = random_sample(partial, 1024)
        gt = random_sample(gt, 2048)

        # ===== 统一归一化（关键）=====
        gt, partial = normalize_pair(gt, partial)

        # ===== 保存 =====
        data = {
            "partial": partial.astype(np.float32),
            "gt": gt.astype(np.float32)
        }

        save_path = os.path.join(save_root, split, f"{idx:06d}.npy")
        np.save(save_path, data)

        if idx % 50 == 0:
            print(f"[{split}] {idx}/{len(name_list)}")

# =========================
# 执行
# =========================
process_split(train_names, "train")
process_split(val_names, "val")

print("✅ 数据处理完成（PoinTr标准）")