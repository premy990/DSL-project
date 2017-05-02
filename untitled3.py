#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 22:13:00 2017

@author: pk
    cv2.imshow("Edges", edged)
"""

import cv2
#reading the image 
image = cv2.imread('/home/pk/Pictures/Capture3.JPG')
edged = cv2.Canny(image, 10, 250)
cv2.waitKey(0)
     
#applying closing function 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closed", closed)
cv2.waitKey(0)
     
#finding_contours 
(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
    cv2.imshow("/home/pk/Desktop/Output.jpg", image)
    cv2.waitKey(0)