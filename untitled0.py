
import cv2
import numpy as np
#from matplotlib import pyplot as plt

img1 = cv2.imread('/home/pk/git/linear-SVM-on-top-of-CNN-example-master/building_images/validate/House/House 1_1_shrink4.jpg', 0)  # trainImage
img2 = cv2.imread('/home/pk/git/linear-SVM-on-top-of-CNN-example-master/building_images/iss_0_1_shrink2.jpg', 0)  # queryImage

orb = cv2.ORB_create(edgeThreshold=4, patchSize=16)  

kp1 = orb.detect(img1, None)
kp2 = orb.detect(img2, None)

kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img2, kp2)

img3 = cv2.drawKeypoints(img1, kp1, None, color=(255, 0, 0))
img4 = cv2.drawKeypoints(img2, kp2, None, color=(255, 0, 0))
cv2.imwrite('tmp_hp-label.png', img3)
cv2.imwrite('tmp_input.png', img4)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)

img5 = cv2.drawMatches(img1,kp1,img2,kp2,matches, None, flags=2)

cv2.imwrite('/home/pk/Desktop/dog-face2.jpg', img5)

good = matches
#for m in matches:
#    if m.distance < 0.7:
#       good.append(m)


