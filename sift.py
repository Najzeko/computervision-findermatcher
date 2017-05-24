#!usr/bin/python

import cv2
import numpy

#standard image loading commands to first import a sample image and to create a grayscale from that image
image = cv2.imread("/Users/student_mac1/Desktop/sample.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#creates a sift object and then finds keypoints according to the sift algorithm
sift = cv2.xfeatures2d.SIFT_create(75)
keypoints = sift.detect(gray, None)


#draws the keypoints onto the image with one of the options designated by the flags parameter
cv2.drawKeypoints(image, keypoints, image, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


#cv2.imwrite("/Users/student_mac1/Desktop/gray.jpeg", gray)


#standard display commands to first create a window for both images then display them
cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
cv2.namedWindow("corners", cv2.WINDOW_NORMAL)

cv2.imshow("gray", gray)
cv2.imshow("corners", image)

cv2.waitKey(0)
