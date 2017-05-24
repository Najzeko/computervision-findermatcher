#!usr/bin/python

import cv2
import numpy as np
from matplotlib import pyplot as plt

#Load images for matching; a query image (source) and a sample image (train)
source = cv2.imread("/Users/student_mac1/Desktop/marko_miletic/cvimages/hexagon.png")
source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) 
train = cv2.imread("/Users/student_mac1/Desktop/marko_miletic/cvimages/shapes.jpeg", 0)
final = None


#SIFT initialization and descriptor calculation
sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.006, 10)

kp1, des1 = sift.detectAndCompute(source, None)
kp2, des2 = sift.detectAndCompute(train, None)


#FLANN parameter setup and initialization
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 500)
flann = cv2.FlannBasedMatcher(index_params, search_params)


#creates a list of best k matches for each keypoint by using the FLANN algorithm
matches = flann.knnMatch(des1, des2, k=2)


#to show only best matches, a mask is created
matchesMask = [[0,0] for i in xrange(len(matches))]

for i, (m,n) in enumerate(matches):
	if m.distance < 0.9 * n.distance:
		matchesMask[i] = [1,0]


#dictionary that describes drawing details when the result is displayed
draw_params = dict(matchColor = (0, 0, 255), singlePointColor = (255,0,0), matchesMask = matchesMask, flags = 0)


#creates final image
final = cv2.drawMatchesKnn(source, kp1, train, kp2, matches, final, **draw_params)

#cv2.imwrite("/Users/student_mac1/Desktop/matchoctagon2.jpeg", final)

#displays result
cv2.namedWindow("final", cv2.WINDOW_NORMAL)
cv2.imshow("final", final)

cv2.waitKey(0)

'''
mylist = [[5,4,7],[6,9,3],[6,3,8]]

for i,j,k in mylist:
	print i
	print j
	print k
'''
