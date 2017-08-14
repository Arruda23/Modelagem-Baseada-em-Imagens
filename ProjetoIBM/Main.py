import numpy as np
import cv2
import glob
import matplotlib as plt
from PoseEstimation import estimaPose

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('Images/Calibragem/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2=cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners2, ret)

        cv2.imwrite('k.jpg', img)

        cv2.imshow('img', img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

#remover distorção
ret, matrix, distortion, rotationVectors, translationVectors = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None, None)
imgDistortion=cv2.imread('Images/Calibragrem/img1.jpg')
h,w= img.shape[:2]
newcameramtx, regionOfInterest= cv2.getOptimalNewCameraMatrix(matrix,distortion,(w,h),1,(w,h))
#undistort
mapx,mapy=cv2.initUndistortRectifyMap(matrix,distortion,None,newcameramtx,(w,h),5)
dst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
#cortar imagem
x,y,w,h=regionOfInterest
dst=dst[y:y+h,x:x+w]
cv2.imwrite('Images/Calibragem/calibrateResult.png',dst)
print("CALIBRAGEM EXECUTADA COM SUCESSO")

#guarda info da calibragem
np.savez('Images/Calibragem/caliInfo.npz',matrix,distortion)

#with np.load('Images/Calibragem/caliInfo.npz') as X:
#    matrix, distortion, _, _ = [X[i] for i in ('matrix', 'distortion', 'rotationVectors', 'translationVectors')]









#reprojetar o erro
mean_error=0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rotationVectors[i], translationVectors[i], matrix, distortion)
    error=cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    total_error =+ mean_error

print ("total error:", mean_error/len(objpoints))


