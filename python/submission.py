"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import cv2

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    # Normalize the points
    pts1 = pts1 / M
    pts2 = pts2 / M
    
    # Create matrix A from correspondences
    N = pts1.shape[0]

    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]
    A = np.column_stack((x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones(N)))
    
    # Perform SVD on A
    _, S, Vt = np.linalg.svd(A)
    
    # The solution is the last column of V corresponding to the smallest singular value
    F = Vt[np.abs(S).argmin()].reshape(3, 3)
    
    # Enforce rank-2 constraint on F
    U, S, Vt = np.linalg.svd(F)
    S[S.argmin()] = 0  # Set the smallest singular value to 0
    F = U @ np.diag(S) @ Vt
    
    # Denormalize the fundamental matrix
    T = np.diag((1.0/M, 1.0/M, 1))
    F = T.T @ F @ T

    return F






"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1, window_size=10):
    N = pts1.shape[0]
    pts2 = np.zeros_like(pts1)
    half_window = window_size // 2

    for i in range(N):
        x, y = pts1[i]
        v = [[x], [y], [1]]
        l = F @ v # l = (a, b, c), then the line is ax + by + c = 0
                # Iterate through potential points on the epipolar line within the image boundaries

        # Normalize the line to avoid numerical issues
        a, b, c = l.flatten()
        norm_factor = np.sqrt(a**2 + b**2)
        a, b, c = a / norm_factor, b / norm_factor, c / norm_factor

        smallest_diff = float('inf')
        best_pt = None

        # Search through all the pixels on the line
        for x2 in range(half_window, im2.shape[1] - half_window):
            y2 = int(-(a * x2 + c) / b)  # solve for y2 using the line equation

            # Make sure the windows can stay inside the image
            if y2 < half_window or y2 >= im2.shape[0] - half_window:
                continue

            # Extract windows
            window1 = im1[y-half_window:y+half_window+1, x-half_window:x+half_window+1]
            window2 = im2[y2-half_window:y2+half_window+1, x2-half_window:x2+half_window+1]

            # Calculate similarity (e.g., SSD or NCC)
            diff = distance(window1, window2)

            if diff < smallest_diff:
                smallest_diff = diff
                best_pt = (x2, y2)

        pts2[i] = best_pt
    
    return pts2

def distance(window1, window2):
    # Simple euclidean sum 
    return np.sum(np.abs(window1 - window2))


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    return K2.T @ F @ K1


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    # num_points : 3D点的数量
    num_points = pts1.shape[0]
    pts3d = np.zeros((num_points, 3))  # 初始化3D点矩阵

    for i in range(num_points):
        # 获取图像1和图像2中对应的2D点
        u1, v1 = pts1[i]
        u2, v2 = pts2[i]

        # 构建矩阵A
        A = np.array([
            u1 * P1[2] - P1[0],
            v1 * P1[2] - P1[1],
            u2 * P2[2] - P2[0],
            v2 * P2[2] - P2[1]
        ])

        # 对A进行SVD分解
        _, _, V = np.linalg.svd(A)

        # 取V的最后一列（对应于最小奇异值）并归一化
        X = V[-1]
        X /= X[3]  # 齐次坐标归一化

        # 保存3D点
        pts3d[i] = X[:3]

    return pts3d


def pos_num(M, pts3d):
    '''
    计算3D点的深度，并计算在相机前面的点的数量
    深度为正的点在相机前面，深度为负的点在相机后面
    最好大多数点都在相机前面
    需要用到相机的外显参数矩阵P
    '''
    pts3d_camera = M @ np.hstack((pts3d, np.ones((pts3d.shape[0], 1)))).T
    num_positive_depths = np.sum(pts3d_camera[2] > 0)
    return num_positive_depths
    

def find_correct_P2(M1, pts1, M2s, pts2, K1, K2):  
    '''
    这里的M1和M2s都是相机的外显参数矩阵
    K1和K2是相机的内参矩阵
    pts1和pts2是对应的2D点
    分别计算4个P2的深度的点数，找到大多数点在相机前面的P2
    返回的值是正确的M2(即第二个相机的外显参数矩阵)
    '''
    P1 = K1 @ M1  # 相机1的投影矩阵
    P2s = np.zeros((3, 4, 4))  # 4个相机2的投影矩阵
    M2 = None
    num_points = pts1.shape[0]
    max_positive_depths = 0
    correct_pts3d = None
    for i in range(4):
        # print(i)
        # print(M2s[:, :, i])
        # matrix_3x3 = M2s[:, :3 , i]

        # # 计算 3x3 矩阵的行列式
        # determinant = np.linalg.det(matrix_3x3)
        # print(determinant)

        P2s[:, :, i] = K2 @ M2s[:, :, i]  # 相机2的投影矩阵
        # print(P1)
        # print(P2s[:, :, i])

        pts3d = triangulate(P1, pts1, P2s[:, :, i], pts2)
        num_positive_depths = pos_num(M2s[:, :, i], pts3d) + pos_num(M1, pts3d)
        # print("i: ", i)
        # print("num_positive_depths: ", num_positive_depths)
        if num_positive_depths > max_positive_depths:
            max_positive_depths = num_positive_depths
            M2 = P2s[:, :, i]
            correct_pts3d = pts3d
        
    return M2, correct_pts3d


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    pass


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
