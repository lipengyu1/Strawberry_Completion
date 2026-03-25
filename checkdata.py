import numpy as np
import open3d as o3d
import argparse
import os
import matplotlib.pyplot as plt


def load_npy(path):
    data = np.load(path, allow_pickle=True).item()
    return data["partial"], data["gt"]


def create_pcd(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd


def save_ply(partial, gt, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    pcd_partial = create_pcd(partial, [1, 0, 0])
    pcd_gt = create_pcd(gt, [0, 1, 0])

    o3d.io.write_point_cloud(os.path.join(save_dir, "partial.ply"), pcd_partial)
    o3d.io.write_point_cloud(os.path.join(save_dir, "gt.ply"), pcd_gt)

    print(f"✅ PLY 已保存到: {save_dir}")



def save_png(partial, gt, save_path):
    gt_shifted = gt + np.array([1.5, 0, 0])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(partial[:,0], partial[:,1], partial[:,2], s=1)
    ax.scatter(gt_shifted[:,0], gt_shifted[:,1], gt_shifted[:,2], s=1)

    ax.set_axis_off()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ PNG saved: {save_path}")


def print_stats(partial, gt):
    print("\n===== 数据检查 =====")
    print("Partial shape:", partial.shape)
    print("GT shape:", gt.shape)

    print("Partial range:", partial.min(), partial.max())
    print("GT range:", gt.min(), gt.max())

    print("Partial center:", np.mean(partial, axis=0))
    print("GT center:", np.mean(gt, axis=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", type=str, required=True)
    parser.add_argument("--out", type=str, default="./vis_result")

    args = parser.parse_args()

    partial, gt = load_npy(args.npy)

    print_stats(partial, gt)

    os.makedirs(args.out, exist_ok=True)

    # 保存PLY（可用MeshLab/CloudCompare看）
    save_ply(partial, gt, args.out)

    # 保存PNG（服务器无GUI也能用）
    save_png(partial, gt, os.path.join(args.out, "vis.png"))