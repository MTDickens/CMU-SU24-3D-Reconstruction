import cv2 as cv
import numpy as np
import helper as hlp
import submission as sub
import numpy.linalg as la
import skimage.color as col
import matplotlib.pyplot as plt

# 1. Load the images and the parameters

I1 = cv.cvtColor(cv.imread('../data/im1.png'), cv.COLOR_BGR2GRAY).astype(np.float32)
I2 = cv.cvtColor(cv.imread('../data/im2.png'), cv.COLOR_BGR2GRAY).astype(np.float32)

intrinsics = np.load('../data/intrinsics.npz')
K1, K2 = intrinsics['K1'], intrinsics['K2']

corresp = np.load('../data/some_corresp.npz')
pts1, pts2 = corresp['pts1'], corresp['pts2']
# pts1, pts2 = hlp._projtrans(M1, pts1), hlp._projtrans(M2, pts2)

M = np.max((np.max(pts1), np.max(pts2)))
F = sub.eight_point(pts1, pts2, M)
# hlp.displayEpipolarF(I1, I2, F)
hlp.epipolarMatchGUI(I1, I2, F)