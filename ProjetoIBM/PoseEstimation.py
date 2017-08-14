import cv2
import numpy as np
import glob


def estimaPose(mtx, dist):
    with np.load('Images/Calibragem/caliInfo.npz') as X:
        mtx, dist, _, _= [X[i] for i in ('mtx','dist', 'rvecs', 'tvecs')]

