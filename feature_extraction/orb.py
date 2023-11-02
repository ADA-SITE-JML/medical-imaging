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
#img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)


def drawPoints(img, points):
	img_new = np.stack((img,)*3, axis=-1)
	for coord in points:
		img_new = cv.circle(img_new,(int(coord[0]),int(coord[1])),5, (255,0,0),2)
	return img_new

# taken from these posts (combination):
# https://stackoverflow.com/questions/30327659/how-can-i-remap-a-point-after-an-image-rotation
# https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py
def getTransformedPixel(points, angle, scale = 1.0):
	(h, w) = img.shape[:2]
	(cX, cY) = (w / 2, h / 2)

	# add ones
	ones = np.ones(shape=(len(points), 1))
	points_ones = np.hstack([points, ones])

	M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])

	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))
	# adjust the rotation matrix to take into account translation
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY

	# transform points
	return M.dot(points_ones.T).T

img2 = np.copy(img)

rot_p_30 = imutils.rotate_bound(img, angle=30)
rot_n_30 = imutils.rotate_bound(img, angle=-30)
rot_p_15 = imutils.rotate_bound(img, angle=15)
rot_n_15 = imutils.rotate_bound(img, angle=-15)
rot_p_45 = imutils.rotate_bound(img, angle=45)

kp, des = orb.detectAndCompute(img, None)

# Get all the coordinates and convert them to int
coords = [(int(k.pt[0]),int(k.pt[1])) for k in kp][:20]

fig, axarr = plt.subplots(2,3)

axarr[0,0].imshow(drawPoints(img2,coords))

trPixels = getTransformedPixel(coords, 30, scale = 1.0)
axarr[0,1].imshow(drawPoints(rot_p_30, trPixels))

trPixels = getTransformedPixel(coords, -30, scale = 1.0)
axarr[0,2].imshow(drawPoints(rot_n_30, trPixels))

trPixels = getTransformedPixel(coords, 15, scale = 1.0)
axarr[1,0].imshow(drawPoints(rot_p_15, trPixels))

trPixels = getTransformedPixel(coords, -15, scale = 1.0)
axarr[1,1].imshow(drawPoints(rot_n_15, trPixels))

trPixels = getTransformedPixel(coords, 45, scale = 1.0)
axarr[1,2].imshow(drawPoints(rot_p_45, trPixels))

plt.show()
