# -*- coding: utf-8 -*-

import logging
import open3d
import torch
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

# ===== 尝试加载 EMD（可选）=====
try:
    from extensions.emd import emd_module as emd
    EMD_AVAILABLE = True
except:
    EMD_AVAILABLE = False


class Metrics(object):

    # ===== 基础指标（一定可用）=====
    ITEMS = [
        {
            'name': 'F-Score',
            'enabled': True,
            'eval_func': 'cls._get_f_score',
            'is_greater_better': True,
            'init_value': 0
        },
        {
            'name': 'CDL1',
            'enabled': True,
            'eval_func': 'cls._get_chamfer_distancel1',
            'eval_object': ChamferDistanceL1(ignore_zeros=True),
            'is_greater_better': False,
            'init_value': 32767
        },
        {
            'name': 'CDL2',
            'enabled': True,
            'eval_func': 'cls._get_chamfer_distancel2',
            'eval_object': ChamferDistanceL2(ignore_zeros=True),
            'is_greater_better': False,
            'init_value': 32767
        }
    ]

    # ===== 动态添加 EMD（如果可用）=====
    if EMD_AVAILABLE:
        ITEMS.append({
            'name': 'EMDistance',
            'enabled': True,
            'eval_func': 'cls._get_emd_distance',
            'eval_object': emd.emdModule(),
            'is_greater_better': False,
            'init_value': 32767
        })


    # ===============================
    # 主接口
    # ===============================
    @classmethod
    def get(cls, pred, gt, require_emd=False):
        _items = cls.items()
        _values = [0] * len(_items)

        for i, item in enumerate(_items):

            # ===== 如果不需要 EMD，跳过 =====
            if item['name'] == 'EMDistance' and not require_emd:
                _values[i] = torch.tensor(0.).to(gt.device)
                continue

            eval_func = eval(item['eval_func'])
            _values[i] = eval_func(pred, gt)

        return _values


    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i['enabled']]


    @classmethod
    def names(cls):
        return [i['name'] for i in cls.items()]


    # ===============================
    # F-score
    # ===============================
    @classmethod
    def _get_f_score(cls, pred, gt, th=0.01):

        b = pred.size(0)
        device = pred.device

        if b != 1:
            f_score_list = []
            for idx in range(b):
                f_score_list.append(
                    cls._get_f_score(pred[idx:idx+1], gt[idx:idx+1], th)
                )
            return sum(f_score_list) / len(f_score_list)

        pred = cls._get_open3d_ptcloud(pred)
        gt = cls._get_open3d_ptcloud(gt)

        dist1 = pred.compute_point_cloud_distance(gt)
        dist2 = gt.compute_point_cloud_distance(pred)

        recall = float(sum(d < th for d in dist2)) / float(len(dist2))
        precision = float(sum(d < th for d in dist1)) / float(len(dist1))

        result = 2 * recall * precision / (recall + precision) if recall + precision else 0.

        return torch.tensor(result).to(device)


    @classmethod
    def _get_open3d_ptcloud(cls, tensor):
        tensor = tensor.squeeze().cpu().numpy()
        ptcloud = open3d.geometry.PointCloud()
        ptcloud.points = open3d.utility.Vector3dVector(tensor)
        return ptcloud


    # ===============================
    # Chamfer Distance
    # ===============================
    @classmethod
    def _get_chamfer_distancel1(cls, pred, gt):
        chamfer = cls.ITEMS[1]['eval_object']
        return chamfer(pred, gt) * 1000


    @classmethod
    def _get_chamfer_distancel2(cls, pred, gt):
        chamfer = cls.ITEMS[2]['eval_object']
        return chamfer(pred, gt) * 1000


    # ===============================
    # EMD（仅在可用时）
    # ===============================
    @classmethod
    def _get_emd_distance(cls, pred, gt, eps=0.005, iterations=100):
        if not EMD_AVAILABLE:
            return torch.tensor(0.).to(gt.device)

        emd_loss = cls.ITEMS[-1]['eval_object']
        dist, _ = emd_loss(pred, gt, eps, iterations)
        emd_out = torch.mean(torch.sqrt(dist))

        return emd_out * 1000


    # ===============================
    # 封装
    # ===============================
    def __init__(self, metric_name, values):

        self._items = Metrics.items()
        self.metric_name = metric_name

        if isinstance(values, list):
            self._values = values
        elif isinstance(values, dict):
            self._values = [item['init_value'] for item in self._items]

            metric_indexes = {item['name']: idx for idx, item in enumerate(self._items)}

            for k, v in values.items():
                if k in metric_indexes:
                    self._values[metric_indexes[k]] = v
                else:
                    logging.warning(f'Ignore Metric {k}')
        else:
            raise Exception('Unsupported value type')


    def state_dict(self):
        return {item['name']: self._values[i] for i, item in enumerate(self._items)}


    def __repr__(self):
        return str(self.state_dict())


    def better_than(self, other):
        if other is None:
            return True

        for i, item in enumerate(self._items):
            if item['name'] == self.metric_name:
                if item['is_greater_better']:
                    return self._values[i] > other._values[i]
                else:
                    return self._values[i] < other._values[i]

        raise Exception('Invalid metric name')