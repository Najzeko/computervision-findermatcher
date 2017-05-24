#!usr/bin/python

import cv2
import numpy
from matplotlib import pyplot as plt


image = cv2.imread("/Users/student_mac1/Desktop/shapes.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#print gray, '\n'
#print image, '\n'

corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
corners = numpy.int0(corners)

#print image

#cv2.imwrite("/Users/student_mac1/Desktop/gray.jpeg", gray)

for i in corners:
	x,y = i.ravel()
	cv2.circle(image, (x,y), 3, [0,0,255], -1)
	
cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
cv2.namedWindow("corners", cv2.WINDOW_NORMAL)


cv2.imshow("gray", gray)
cv2.imshow("corners", image)

cv2.waitKey(0)
