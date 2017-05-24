#!usr/bin/python

import cv2
import numpy as np
from matplotlib import pyplot as plt


source = cv2.imread("/Users/student_mac1/Desktop/hexagon.png")
source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) 
train = cv2.imread("/Users/student_mac1/Desktop/shapes.jpeg", 0)
final = None

sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.006, 10)

kp1, des1 = sift.detectAndCompute(source, None)
kp2, des2 = sift.detectAndCompute(train, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 500)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)


hits = []
for j, k in matches:
	if j.distance < 0.99 * k.distance:
		hits.append(j)
		
		


if len(hits) > 5:
	src_pts = np.float32([kp1[m.queryIdx].pt for m in hits]).reshape(-1, 1, 2)
	dst_pts = np.float32([kp2[m.trainIdx].pt for m in hits]).reshape(-1, 1, 2)
	
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	matchesMask = mask.ravel().tolist()
	
	h, w = source.shape
	pts = np.float32([[0,0], [0, (h-1)], [(w-1), (h-1)], [(w-1), 0]]).reshape(-1, 1, 2)
	dst = cv2.perspectiveTransform(pts, M)
	
	train = cv2.polylines(train, [np.int32(dst)], True, 176, 3, cv2.LINE_AA)
	
else:
	print "not good enough"
	matchesMask = None


draw_params = dict(matchColor = (0, 0, 255), singlePointColor = None, matchesMask = None, flags = 2)


final = cv2.drawMatchesKnn(source, kp1, train, kp2, matches, final, **draw_params)

#cv2.imwrite("/Users/student_mac1/Desktop/matchoctagon2.jpeg", final)

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
