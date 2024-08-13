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
├── proj-b.pdf
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

### 说明

**注意**：部分文件详见 pdf 说明

1. `python` 文件夹就是所有我们所需的源代码。
2. `submission.py` 里面包含所有我们需要实现的函数
3. `my_test.py` 是我自己写的 script，可以测试 Eight Point Algorithm 和 Find Epipolar Correspondences 的正确性

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