import cv2
import numpy as np
import sonarGeometry
 
picPath = "D:\Pai_work\pic_sonar"
picName = "\RTheta_img_" + str(10) + ".jpg"
img = cv2.imread(picPath + picName, 0)
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = sonarGeometry.remapping(img)

cv2.imwrite("D:\Pai_work\mask.png", gray)

# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray,None)

# img = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite('sift_keypoints.jpg',img)