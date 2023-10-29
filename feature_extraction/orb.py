import sys
import imutils
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

fname = sys.argv[1]
img = cv.imread(fname, cv.IMREAD_GRAYSCALE)

# Initiate ORB detector
orb = cv.ORB_create()


# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

def drawKeypoints(img):
	kp, des = orb.detectAndCompute(img, None)
	return cv.drawKeypoints(img, kp[0:100], None, color=(0,255,0), flags=0)

def findCommonCoordinates(keypoint_arr):
	# 

def drawCommonKeypoints(img,kp):
	# 


img2 = np.copy(img)
rot_p_30 = imutils.rotate(img, angle=30)
rot_n_30 = imutils.rotate(img, angle=-30)
rot_p_15 = imutils.rotate(img, angle=15)
rot_n_15 = imutils.rotate(img, angle=-15)

fig, axarr = plt.subplots(1,5)
axarr[0].imshow(drawKeypoints(img2))
axarr[1].imshow(drawKeypoints(rot_p_30))
axarr[2].imshow(drawKeypoints(rot_n_30))
axarr[3].imshow(drawKeypoints(rot_p_15))
axarr[4].imshow(drawKeypoints(rot_n_15))

plt.show()

