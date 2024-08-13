# 3D重建作业项目

## 项目概述
本项目是CMU SU24课程的3D重建作业。我们的目标是实现一个基本的3D重建系统，能够从多视角图像中重建3D模型。

## 环境配置
- OpenCV
- Scikit Image
- NumPy
- Matplotlib

## 安装指南

1. 安装依赖
   ```
   pip install -r requirements.txt
   ```

## 项目结构
```
.
├── README.md
├── data
│   ├── im1.png
│   ├── im2.png
│   ├── intrinsics.npz
│   ├── pnp.npz
│   ├── some_corresp.npz
│   └── temple_coords.npz
├── python
│   ├── helper.py
│   ├── my_test.py
│   ├── project_cad.py
│   ├── submission.py
│   ├── test_depth.py
│   ├── test_params.py
│   ├── test_pose.py
│   ├── test_rectify.py
│   └── test_temple_coords.py
├── requirements.txt
└── zip.sh
```

**注意**：不包含 .gitignore 和 git 元数据文件。

## TODO List

- [ ] Sparse Reconstruction
    - [x] Eight Point Algorithm
    - [x] Find Epipolar Correspondences
    - [x] Compute the Essential Matrix
    - [ ] Implement Triangulation
    - [ ] Write test script in `test_temple_coords.py` that uses `data/temple_coords.npz`
- [ ] Dense Reconstruction
    - [ ] Image Rectification
    - [ ] Disparity Map
    - [ ] Depth Map
    - [ ] Pose Estimation **(Extra Credit)**
        - [ ] Camera Matrix Estimation
        - [ ] Camera Parameter Estimation
        - [ ] Project a CAD model to the image