import sys
import imutils
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def normalize(img):
	norm_img = img - np.min(img)
	norm_img = norm_img / np.max(norm_img)

def crop_image(img,th=0):
	mask = img > th
	return img[np.ix_(mask.any(1),mask.any(0))]

# doubles the image's width and height by adding black margins
def add_margins(img):
	(h, w) = img.shape[:2]
	mx = max(h,w)*2
	new_img = np.zeros((mx,mx), np.uint8)

	hh = (mx-h)//2
	wh = (mx-w)//2

	new_img[hh:hh+h,wh:wh+w] = img
	return new_img


def drawPoints(img, points):
	col = (255,0,0)
	thk = 1

	img_new = np.stack((img,)*3, axis=-1)

	for coord in points:
		point = (int(coord[0]),int(coord[1]))

		img_new = cv.circle(img_new,point,5, col,thk)
	return img_new


def rotateAndDrawPoints(img, angle):
	col = (255,0,0)
	thk = 1

	img_new = imutils.rotate_bound(img, angle = angle)
#	img_new = imutils.rotate_bound(img_new, angle = -angle)

	img_new = np.stack((img_new,)*3, axis=-1)
	kp = orb.detect(img_new, None)

	return cv.drawKeypoints(img_new, kp, None, color=(255,0,0), flags=0)


def drawCommonPoints(img, points):
	img_new = np.stack((img,)*3, axis=-1)

	for coord in points:
		point = (int(coord[0]),int(coord[1]))
		col = (255,0,0)
		thk = 1
		if point in matchDict:
			col = (0,255,0)
			thk = 3
			img_new = cv.circle(img_new,point,5, col,thk)
	return img_new


# Credits: taken from these posts (combination):
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


def getImageWithTransformedKeypoints(img, angle, original_coords):
	rot_img = imutils.rotate_bound(img, angle = angle)

	# Get all the coordinates and convert them to int
	trPixels = getTransformedPixel(original_coords, angle, scale = 1.0)

	return drawPoints(rot_img, trPixels)


# returns the disctionary of nearby points (key) with the match count (value)
def matchPoints(sorted_list1, sorted_list2, maxDistance = 50):
	global matchDict

	idx1 = 1
	idx2 = 1

	while (idx1 < len(sorted_list1)) and (idx2 < len(sorted_list2)):
		if cityblock(sorted_list1[idx1], sorted_list2[idx2]) < maxDistance:
			if sorted_list1[idx1] in matchDict:
				matchDict[sorted_list1[idx1]] += 1
			else:
				matchDict[sorted_list1[idx1]] = 1
			idx2 += 1
		else:
			idx1 += 1
	return matchDict

def getImages(img, coords, max_images):
	img_arr = np.zeros((max_images,40,40))

	for i in range(max_images):
		coord = coords[i]
		img_arr[i,:,:] = img[coord[0]-20:coord[0]+20,coord[1]-20:coord[1]+20]
	return img_arr


fname = sys.argv[1]
im_init = cv.imread(fname, cv.IMREAD_GRAYSCALE)

#---------------------------------------------------------------
#img = normalize(img_)
img_crp = crop_image(im_init,2)
img_margin = add_margins(img_crp)


# Screen 1: Demonstration of the keypoint transformation
fig, axarr = plt.subplots(1,3)
fig.suptitle('1. Initial preparation')
axarr[0].imshow(im_init,cmap='gray')
axarr[1].imshow(img_crp,cmap='gray')
axarr[2].imshow(img_margin,cmap='gray')
plt.show()

#---------------------------------------------------------------
img = img_margin

# Initiate ORB detector
orb = cv.ORB_create()

matchDict = {}

kp = orb.detect(img, None)

# Get all the coordinates and convert them to int
coords = [(int(k.pt[0]),int(k.pt[1])) for k in kp]

# Screen 2: Demonstration of the keypoint transformation
fig, axarr = plt.subplots(2,3)
fig.suptitle('2. Demonstration of the keypoint transformation')
axarr[0,0].imshow(drawPoints(img,coords))
axarr[0,1].imshow(getImageWithTransformedKeypoints(img, 30, coords))
axarr[0,2].imshow(getImageWithTransformedKeypoints(img, -30, coords))
axarr[1,0].imshow(getImageWithTransformedKeypoints(img, 15, coords))
axarr[1,1].imshow(getImageWithTransformedKeypoints(img, -15, coords))
axarr[1,2].imshow(getImageWithTransformedKeypoints(img, 45, coords))
plt.show()

#---------------------------------------------------------------

# Screen 3: Keypoints in various images
fig, axarr = plt.subplots(2,3)
fig.suptitle('3. Keypoints in various images')
axarr[0,0].imshow(drawPoints(img,coords))
axarr[0,1].imshow(rotateAndDrawPoints(img, 30))
axarr[0,2].imshow(rotateAndDrawPoints(img, -30))
axarr[1,0].imshow(rotateAndDrawPoints(img, 15))
axarr[1,1].imshow(rotateAndDrawPoints(img, -15))
axarr[1,2].imshow(rotateAndDrawPoints(img, 45))
plt.show()

#---------------------------------------------------------------

# Screen 4: Showing the 20 common keypoints and the 20x20 regions around them
coords_sorted = sorted(coords)
# md = matchPoints(coords2,trPixels)
# print(md)
fig, axarr = plt.subplots(4,5)
fig.suptitle('4. 20 common keypoints and the 20x20 regions around them')
img_arr  = getImages(img,coords,20)
for i in range(4):
	for j in range(5):
		axarr[i,j].imshow(img_arr[i*5+j],cmap='gray')
plt.show()
