import cv2
import numpy as np
import scipy
from matplotlib import pyplot as plt

img=cv2.imread('Monumento/Images/monument.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


sift = cv2.xfeatures2d.SIFT_create()
kp=sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp,img)

cv2.imwrite('Images/sift_keypoints.jpg',img)



