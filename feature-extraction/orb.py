import sys
import imutils
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import math;

def normalize(img):
	norm_img = img - np.min(img)
	norm_img = norm_img / np.max(norm_img)

def crop_image(img, th=0):
	mask = img > th
	return img[np.ix_(mask.any(1),mask.any(0))]

# Doubles the image's width and height by adding black margins
def add_margins(img):
	(h, w) = img.shape[:2]
	# Find the biggest dimension and double it
	#mx = max(h,w)*2
	# Using the diagonal is a correct way to do that
	mx = math.ceil(2 * math.sqrt(h*h + w*w))
	# A new square image will contain any version of the original image rotated around the center
	new_img = np.zeros((mx,mx), np.uint8)

	#hh = (mx-h)//2
	#wh = (mx-w)//2
	hh = mx // 2 - h
	wh = mx // 2

	# Position the original image with its bottom left corner on the center
	new_img[hh:hh+h, wh:wh+w] = img
	
	return new_img

# Removes the black margins added for rotation
def remove_margins(img, width, height):
	# Get the size of the image and calculate the center coordinates
	(h, w) = img.shape[:2]
	center = (w//2, h//2)
	new_img = img[center[1]-height:center[1], center[0]:center[0]+width]
	
	return new_img

# Converts the coordinates as if margins removed
def remove_margin_coords(points, img, width, height):
	# Get the size of the image and calculate the center coordinates
	(h, w) = img.shape[:2]
	center = (w//2, h//2)
	new_points = [(int(p[0])-center[0], int(p[1])-center[1]+height) for p in points]
	
	return new_points

def drawPoints(img, points, radius=5):
	color = (255, 0, 0)
	thickness = 1

	# This is to convert the image from grayscale to RGB
	img_new = np.stack((img,)*3, axis=-1)

	for coord in points:
		#Draw circles for given points 
		point = (int(coord[0]), int(coord[1]))
		img_new = cv.circle(img_new, point, radius, color, thickness)
	
	return img_new

def rotateAndDrawPoints(img, angle):
	# Set color to red 
	col = (255,0,0)
	thk = 1

	# Get the matrix for rotation around the center of the image
	height, width = img.shape[:2] 
	center = (width/2, height/2) 
	#rotate_matrix = cv.getRotationMatrix2D(center, -angle, 1.0)
	cos = math.cos(math.radians(-angle));
	sin = math.sin(math.radians(-angle));

	# We will use the global variable
	global coords_combined

	#img_new = imutils.rotate_bound(img, angle = angle)
	img_new = imutils.rotate(img, angle = angle)
#	img_new = imutils.rotate_bound(img_new, angle = -angle)

	# No idea what does this one do
	#img_new = np.stack((img_new,)*3, axis=-1)

	# Detect keypoints and draw them on the image
	kp = orb.detect(img_new, None)
	# Rotate the detected points back to the initial orientation
	#coords_transformed = getTransformedPixel([(k.pt[0], k.pt[1]) for k in kp], angle, scale = 1.0)
	
	# Consider that y axis is directed down in screen coordinates
	coords_transformed = [(k.pt[0] - center[0], center[1] - k.pt[1]) for k in kp]
	
	# Rotate all the points around the center of coordinates
	coords_transformed = [(cos * k[0] - sin * k[1], sin * k[0] + cos * k[1]) for k in coords_transformed]

	# Translate all the points back and add to the combined list
	coords_combined = coords_combined + [(int(k[0] + center[0]), int(center[1] - k[1])) for k in coords_transformed]

	return cv.drawKeypoints(img_new, kp, None, color=(255,0,0), flags=0)
	#return drawPoints(img_new, coords_combined, 1)

# Returns the number of keypoints for the 2D histogram
def count_points(points, img, width, height):
	# Get the size of the image
	(h, w) = img.shape[:2]
	rows, cols = (math.ceil(h/height), math.ceil(w/width))
	
	# Introduce a 2D list for storing numbers
	number_points = [[0 for i in range(cols)] for j in range(rows)]
	for coord in points:
		number_points[math.floor(coord[1]/height)][math.floor(coord[0]/width)] += 1
	
	return number_points

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


# returns the dictionary of nearby points (key) with the match count (value)
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

	#for i in range(max_images):
	#	coord = coords[i]
	#	img_arr[i,:,:] = img[coord[0]-20:coord[0]+20,coord[1]-20:coord[1]+20]
	return img_arr

#---------------------------------------------------------------
# BEGIN
#---------------------------------------------------------------
# Get the image file name from the first argument
fname = sys.argv[1]
img_init = cv.imread(fname, cv.IMREAD_GRAYSCALE)
#img_init = cv.imread(fname, cv.IMREAD_COLOR)

#---------------------------------------------------------------
#img = normalize(img_)
img_cropped = crop_image(img_init, 2)
img_margin = add_margins(img_cropped)

# Remember the size of the initial cropped image
(height_cropped, width_cropped) = img_cropped.shape[:2]

# Screen 1: Demonstration of the keypoint transformation
fig, axarr = plt.subplots(1,3)
fig.suptitle('1. Initial preparation')
axarr[0].imshow(img_init, cmap='gray')
axarr[1].imshow(img_cropped, cmap='gray')
axarr[2].imshow(img_margin, cmap='gray')
plt.show()

#---------------------------------------------------------------
img = img_margin

# Initiate ORB detector
orb = cv.ORB_create()

matchDict = {}
# The comnined list to store all detected keypoints 
coords_combined = []

kp = orb.detect(img, None)

# Get all the coordinates and convert them to int
coords = [(int(k.pt[0]), int(k.pt[1])) for k in kp]

# Screen 2: Demonstration of the keypoint transformation
#fig, axarr = plt.subplots(2,3)
#fig.suptitle('2. Demonstration of the keypoint transformation')
#axarr[0,0].imshow(drawPoints(img, coords))
#axarr[0,1].imshow(getImageWithTransformedKeypoints(img, 30, coords))
#axarr[0,2].imshow(getImageWithTransformedKeypoints(img, -30, coords))
#axarr[1,0].imshow(getImageWithTransformedKeypoints(img, 15, coords))
#axarr[1,1].imshow(getImageWithTransformedKeypoints(img, -15, coords))
#axarr[1,2].imshow(getImageWithTransformedKeypoints(img, 45, coords))
#plt.show()

#---------------------------------------------------------------

# Screen 3: Keypoints in various images
fig, axarr = plt.subplots(2,3)
fig.suptitle('3. Keypoints in various images')
axarr[0,0].imshow(drawPoints(img, coords))

# A list for storing all keypoints combined
coords_combined = []
axarr[0,1].imshow(rotateAndDrawPoints(img, 30))
axarr[0,2].imshow(rotateAndDrawPoints(img, -30))
axarr[1,0].imshow(rotateAndDrawPoints(img, 15))
axarr[1,1].imshow(rotateAndDrawPoints(img, -15))
axarr[1,2].imshow(rotateAndDrawPoints(img, 45))
plt.show()

#---------------------------------------------------------------

# Screen 3½: Showing all detected keypoints and the original image
fig, axarr = plt.subplots(1,2)
fig.suptitle('3½. All detected keypoints combined')
axarr[0].imshow(remove_margins(drawPoints(img, coords, 2), width_cropped, height_cropped))
#axarr[0].imshow(remove_margins(drawPoints(img, coords_combined, 2), width_cropped, height_cropped))

# Remove the margins and convert the detected keypoints to original coordinates
coords_combined = remove_margin_coords(coords_combined, img, width_cropped, height_cropped)
img = remove_margins(img, width_cropped, height_cropped)
axarr[1].imshow(drawPoints(img, coords_combined, 2))
plt.show()

#---------------------------------------------------------------

# Screen 4: Showing the 20 common keypoints and the 20x20 regions around them
coords_sorted = sorted(coords)
histogram_numbers = count_points(coords_combined, img, 20, 20)
for row in histogram_numbers:
	print(row)

# md = matchPoints(coords2,trPixels)
# print(md)
fig, axarr = plt.subplots(4,5)
fig.suptitle('4. 20 common keypoints and the 20x20 regions around them')
img_arr  = getImages(img,coords,20)
for i in range(4):
	for j in range(5):
		axarr[i,j].imshow(img_arr[i*5+j],cmap='gray')
plt.show()
