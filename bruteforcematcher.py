#!usr/bin/python

import cv2
import numpy
from matplotlib import pyplot as plt


source = cv2.imread("/Users/student_mac1/Desktop/hexagon.png")
source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) 
train = cv2.imread("/Users/student_mac1/Desktop/shapes.jpeg", 0)
final = None

sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.006, 10)

kp1, des1 = sift.detectAndCompute(source, None)
kp2, des2 = sift.detectAndCompute(train, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)


good = []
for j, k in matches:
	if j.distance < 0.99 * k.distance:
		good.append([j])
		
#print good



final = cv2.drawMatchesKnn(source, kp1, train, kp2, good, final, flags = 2)

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
