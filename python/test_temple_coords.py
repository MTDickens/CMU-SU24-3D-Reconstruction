import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import cv2 as cv

# 1. Load the two temple images and the points from data/some_corresp.npz

#load images
I1 = cv.cvtColor(cv.imread('../data/im1.png'), cv.COLOR_BGR2GRAY).astype(np.float32)
I2 = cv.cvtColor(cv.imread('../data/im2.png'), cv.COLOR_BGR2GRAY).astype(np.float32)

# intrinsics = np.load('../data/intrinsics.npz')
# K1, K2 = intrinsics['K1'], intrinsics['K2']

corresp = np.load('../data/some_corresp.npz')
pts1, pts2 = corresp['pts1'], corresp['pts2']

# 2. Run eight_point to compute F

M = np.max((np.max(pts1), np.max(pts2)))
F = sub.eight_point(pts1, pts2, M)
# hlp.displayEpipolarF(I1, I2, F)
# hlp.epipolarMatchGUI(I1, I2, F)

# 3. Load points in image 1 from data/temple_coords.npz
temple_coords = np.load('../data/temple_coords.npz')
temple_pts1 = temple_coords['pts1']

# 4. Run epipolar_correspondences to get points in image 2
temple_pts2 = sub.epipolar_correspondences(I1, I2, F, temple_pts1)

# 5. Compute the camera projection matrix P1
intrinsics = np.load('../data/intrinsics.npz')
#获取相机内参
K1, K2 = intrinsics['K1'], intrinsics['K2']
#相机1的外参（单位矩阵，假设没有旋转和平移）
M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P1 = K1 @ M1
#计算本质矩阵
E = sub.essential_matrix(F, K1, K2)

# 6. Use camera2 to get 4 camera projection matrices P2
M2s = hlp.camera2(E)

# 7. Run triangulate using the projection matrices
P2s = np.zeros((3, 4, 4))
for i in range(4):
    P2s[:, :, i] = K2 @ M2s[:, :, i]


pts3d0 = sub.triangulate(P1, temple_pts1, P2s[:, :, 0], temple_pts2)
pts3d1 = sub.triangulate(P1, temple_pts1, P2s[:, :, 1], temple_pts2)
pts3d2 = sub.triangulate(P1, temple_pts1, P2s[:, :, 2], temple_pts2)
pts3d3 = sub.triangulate(P1, temple_pts1, P2s[:, :, 3], temple_pts2)


# 8. Figure out the correct P2
M2,pts3d = sub.find_correct_P2(M1, temple_pts1, M2s, temple_pts2, K1, K2)
print("M2:))))))))))))))))))")
print(M2)
print("pts3d:))))))))))))))))))")
print(pts3d)

# 9. Scatter plot the correct 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pts3d_cameraP1[0], pts3d_cameraP1[1], pts3d_cameraP1[2], c='r', marker='o')
# ax.scatter(pts3d_cameraP2[0], pts3d_cameraP2[1], pts3d_cameraP2[2], c='b', marker='o')
ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], c='r', marker='o')
# ax.scatter(pts3d0[:, 0], pts3d0[:, 1], pts3d0[:, 2], c='r', marker='o')
# ax.scatter(pts3d1[:, 0], pts3d1[:, 1], pts3d1[:, 2], c='r', marker='o')
# ax.scatter(pts3d2[:, 0], pts3d2[:, 1], pts3d2[:, 2], c='r', marker='o')
# ax.scatter(pts3d3[:, 0], pts3d3[:, 1], pts3d3[:, 2], c='b', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
