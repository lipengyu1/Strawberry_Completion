import torch
import os
import numpy as np
from tqdm import tqdm

from models import build_model_from_cfg
from tools import builder
from utils.config import get_config
from utils import parser
from utils.logger import get_root_logger

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


def test_strawberry(args, config):
    logger = get_root_logger(name='test')

    # ===== 构建数据集 =====
    (_, test_dataloader) = builder.dataset_builder(args, config.dataset.test)

    # ===== 构建模型 =====
    model = build_model_from_cfg(config.model)

    checkpoint = torch.load(args.ckpts)
    model.load_state_dict(checkpoint['model'], strict=False)

    model = model.cuda()
    model.eval()

    chamfer_l1 = ChamferDistanceL1().cuda()
    chamfer_l2 = ChamferDistanceL2().cuda()

    total_l1, total_l2 = 0.0, 0.0
    total_f = 0.0
    count = 0

    print("\n===== Testing Strawberry Dataset =====")

    with torch.no_grad():
        for batch in tqdm(test_dataloader):

            # 兼容你的 dataset 返回格式
            if len(batch) == 3:
                _, _, data = batch
                partial = data[0].cuda()
                gt = data[1].cuda()
            else:
                partial, gt = batch
                partial = partial.cuda()
                gt = gt.cuda()

            # ===== forward =====
            pred = model(partial)

            if isinstance(pred, tuple):
                pred = pred[-1]

            # ===== Chamfer =====
            l1 = chamfer_l1(pred, gt).item()
            l2 = chamfer_l2(pred, gt).item()

            total_l1 += l1
            total_l2 += l2

            count += 1

    avg_l1 = total_l1 / count * 1e3
    avg_l2 = total_l2 / count * 1e4

    print("\n===== FINAL RESULT =====")
    print(f"L1_CD (1e-3): {avg_l1:.4f}")
    print(f"L2_CD (1e-4): {avg_l2:.4f}")


def main():
    args = parser.get_args()
    config = get_config(args)

    test_strawberry(args, config)


if __name__ == '__main__':
    main()