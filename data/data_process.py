import os
import numpy as np
import random
import open3d as o3d

input_dir = "/your/path/PoinTr/data/Complete_pointcloud_downsample"
save_root = "/your/path/PoinTr/data/strawberry"

train_ratio = 0.8
seed = 42

def hidden_point_removal(pc, camera=[0, 0, 2], radius=100):
    """
    pc: (N,3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    _, pt_map = pcd.hidden_point_removal(camera, radius)
    visible = np.asarray(pcd.points)[pt_map]

    return visible


def normalize_pair(gt, partial):
    center = np.mean(gt, axis=0)
    scale = np.max(np.linalg.norm(gt - center, axis=1))

    gt = (gt - center) / scale
    partial = (partial - center) / scale

    return gt, partial


def load_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points).astype(np.float32)


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


names = [f"strawberry{str(i).zfill(3)}" for i in range(1, 401)]

random.seed(seed)
random.shuffle(names)

split_idx = int(len(names) * train_ratio)
train_names = names[:split_idx]
val_names = names[split_idx:]

print(f"Train: {len(train_names)}, Val: {len(val_names)}")


os.makedirs(os.path.join(save_root, "train"), exist_ok=True)
os.makedirs(os.path.join(save_root, "val"), exist_ok=True)


def process_split(name_list, split):
    for idx, name in enumerate(name_list):
        ply_path = os.path.join(input_dir, f"{name}.ply")

        gt = load_ply(ply_path)

        partial = hidden_point_removal(gt)

        partial = random_sample(partial, 1024)
        gt = random_sample(gt, 2048)

        gt, partial = normalize_pair(gt, partial)

        data = {
            "partial": partial.astype(np.float32),
            "gt": gt.astype(np.float32)
        }

        save_path = os.path.join(save_root, split, f"{idx:06d}.npy")
        np.save(save_path, data)

        if idx % 50 == 0:
            print(f"[{split}] {idx}/{len(name_list)}")

process_split(train_names, "train")
process_split(val_names, "val")

print("date process successful！")
