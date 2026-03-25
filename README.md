# PoinTr Environment Setup for Strawberry Point Cloud Completion

This document describes how to set up the environment for running **PoinTr** on a strawberry point cloud completion task.

---

## 📌 Prerequisites

- Linux (Ubuntu recommended)
- CUDA 11.5 compatible GPU
- Conda (Anaconda or Miniconda)

---

## ⚙️ Environment Setup

### 1. Create and Activate Conda Environment

```bash
conda create -n pntr python=3.7 -y
conda activate pntr
```
### 2. Install PyTorch with CUDA

```bash
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 torchaudio==0.11.0
```

### 3.Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4.Install Chamfer Distance

```bash
cd /PoinTr/chamfer_dist
python setup.py install
```

### 5.Install PointNet++ Ops

```bash
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

### 6.Install Gridding Module

```bash
cd /PoinTr/extensions/gridding
python setup.py install
```

### 7.Install KNN CUDA

```bash
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
## 📂 Dataset Structure

```
The strawberry point cloud dataset should be organized as follows:
PointTr/
├── data/
│      └── strawberry/
│              ├── train/
│              │      ├── 000000.npy
│              │      └── b00001.npy
│              └── val/
│                      ├── 000000.npy
│                      ├── 000001.npy
│                      └── ...
└── ...
```

## 🙏 Acknowledgements

This project is based on the excellent work of **PoinTr**. We thank the authors for making their code publicly available.

- PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers  
  https://github.com/yuxumin/PoinTr

## 📚 References

If you use this project, please consider citing the original PoinTr work:

```bibtex
@inproceedings{pointr,
  title={PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers},
  author={Yu, Xin and others},
  booktitle={ICCV},
  year={2021}
}
```