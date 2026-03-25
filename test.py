import os
import torch
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

# ====================== 路径配置 ======================
CKPT_PATH = "/chenyiming/projects/PoinTr/experiments/PoinTr_strawberry/PCN_models/strawberry_pointr/ckpt-best.pth"
CONFIG_PATH = "cfgs/PCN_models/PoinTr_strawberry.yaml"
EXP_NAME = "strawberry_pointr"

SAVE_DIR = f"experiments/{EXP_NAME}/visualization_ply"

MAX_SAMPLES = None   # 可改成 20 测试前20个
# =====================================================


# ====================== 导入项目模块 ======================
from utils.logger import get_root_logger
from utils.config import get_config
from tools import builder
from models import build_model_from_cfg
# =======================================================


# ====================== 保存PLY ======================
def save_ply(points, color, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    color = np.array(color)
    color = np.tile(color, (points.shape[0], 1))
    color = (color * 255).astype(np.uint8)

    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points, color):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

    print(f"✅ 保存: {filename}")
# =======================================================


# ====================== 加载模型（关键修复） ======================
def load_model(model, ckpt_path):
    print(f"🔄 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'base_model' in ckpt:
        state_dict = ckpt['base_model']
    else:
        state_dict = ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    print("✅ 权重加载完成")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
# =======================================================


def main():
    # ====================== 构造 args（修复 parser 问题） ======================
    args = argparse.Namespace()
    args.config = CONFIG_PATH
    args.exp_name = EXP_NAME
    args.ckpts = CKPT_PATH
    args.num_workers = 4  

    args.resume = False
    args.distributed = False
    args.local_rank = 0
    args.seed = 0
    args.log_name = "test"
    args.use_gpu = True
    args.test = True

    # experiment 路径（必须）
    config_path = Path(args.config)
    args.experiment_path = os.path.join(
        './experiments',
        config_path.stem,
        config_path.parent.stem,
        args.exp_name
    )
    os.makedirs(args.experiment_path, exist_ok=True)

    logger = get_root_logger(
        log_file=os.path.join(args.experiment_path, 'test.log'),
        name=args.log_name
    )

    logger.info("📦 加载配置...")
    config = get_config(args, logger=logger)

    # ====================== 构建数据集 ======================
    logger.info("📦 构建数据集...")

    (_, test_loader) = builder.dataset_builder(args, config.dataset.test)

    logger.info(f"测试集大小: {len(test_loader.dataset)}")

    # ====================== 构建模型 ======================
    logger.info("🧠 构建模型...")
    model = build_model_from_cfg(config.model).cuda()
    model.eval()

    load_model(model, CKPT_PATH)

    print("\n🚀 开始测试并导出PLY...\n")

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):

            if MAX_SAMPLES and idx >= MAX_SAMPLES:
                break

            # ===== 兼容两种dataset格式 =====
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                taxonomy_ids, model_ids, data = batch
                partial, gt = data[0].cuda(), data[1].cuda()
                model_id = str(model_ids[0])
            else:
                partial, gt = batch[0].cuda(), batch[1].cuda()
                model_id = str(idx)

            # ===== 前向 =====
            ret = model(partial)

            # ===== 统一解析输出（关键修复）=====
            if isinstance(ret, dict):
                pred = ret.get('fine', None)
                if pred is None:
                    pred = ret.get('output', ret.get('coarse'))

            elif isinstance(ret, (list, tuple)):
                # ⭐ PoinTr常见： (coarse, fine)
                if len(ret) == 2:
                    pred = ret[1]   # fine
                elif len(ret) >= 1:
                    pred = ret[-1]  # 最后一个通常是 fine
                else:
                    raise ValueError("模型返回空 tuple")

            else:
                pred = ret

            # ===== 转numpy =====
            partial_np = partial.squeeze(0).cpu().numpy()
            pred_np = pred.squeeze(0).cpu().numpy()
            gt_np = gt.squeeze(0).cpu().numpy()

            name = f"{idx:04d}_{model_id}"

            # ===== 保存 =====
            save_ply(partial_np, [1, 0, 0], os.path.join(SAVE_DIR, f"{name}_input.ply"))
            save_ply(pred_np,    [0, 0, 1], os.path.join(SAVE_DIR, f"{name}_pred.ply"))
            save_ply(gt_np,      [0, 1, 0], os.path.join(SAVE_DIR, f"{name}_gt.ply"))

            if idx % 10 == 0:
                print(f"已处理 {idx}/{len(test_loader)}")

    print("\n🎉 完成！PLY已保存到：")
    print(os.path.abspath(SAVE_DIR))


if __name__ == "__main__":
    main()