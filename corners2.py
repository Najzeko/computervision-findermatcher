#!usr/bin/python

import cv2
import numpy

image = cv2.imread("/Users/student_mac1/Desktop/shapes.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

grayf = numpy.float32(gray)

#print grayf, '\n'
#print gray, '\n'
#print image, '\n'


dst = cv2.cornerHarris(gray, 6, 3, 0.09)
dst = cv2.dilate(dst, None)
#dst = cv2.dilate(dst, None)
print dst, '\n'

image[dst>0.04*dst.max()] = [0, 255, 0]

#print image

cv2.imwrite("/Users/student_mac1/Desktop/gray.jpeg", gray)

cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
cv2.namedWindow("corners", cv2.WINDOW_NORMAL)


cv2.imshow("gray", gray)
cv2.imshow("corners", image)


cv2.waitKey(0)
