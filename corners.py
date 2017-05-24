#!usr/bin/python

import cv2
import numpy


#gets an image from a specified path
def getimage():
	#return cv2.imread(raw_input("enter the path of the image to be processed: "))	
	return cv2.imread("/Users/student_mac1/Desktop/football.jpg")


#makes an image grayscale
def gray(source):
	return cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)


#finds and shows corners on an image
def corners(image, grayscale):
	dst = cv2.cornerHarris(grayscale, 6, 3, 0.04)
	image[dst > 0.04 * dst.max()] = [0, 255, 0]


#displays the given image
def show(source, name):
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	cv2.imshow(name, source)
	


image = getimage()	#obtain image
gray = gray(image)	#obtain grayscale

corners(image, gray)	#apply corner detector

#display images
show(gray, "gray")
show(image, "image")


cv2.waitKey(0)
