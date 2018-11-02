import cv2
import numpy as np
import sonarGeometry
import regisProcessing as rp
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img1 = rp.AverageMultiLook(10,1)
    img2 = rp.AverageMultiLook(20,1)
    # trans_matrix = np.float32([[1,0,0],[0,1,-16]])
    # out = cv2.warpAffine(img2, trans_matrix, (768,500))

    ### Test R ###
    for inx in range(0,20):
        trans_matrix = np.float32([[1,0,0],[0,1,(-1)*inx]])
        imgShift = cv2.warpAffine(img2, trans_matrix, (768,500))

        imgRef = img1[0:500-inx, 0:768]
        imgShift = imgShift[0:500-inx, 0:768]

        coef = cv2.matchTemplate(imgRef, imgShift, cv2.TM_CCOEFF_NORMED)
        print (inx, coef[0][0])

    ### Test R +- ###
    # for inx in range(-19,20):
    #     trans_matrix = np.float32([[1,0,0],[0,1,inx]])
    #     imgShift = cv2.warpAffine(img2, trans_matrix, (768,500))

    #     if inx < 0:
    #         imgRef = img1[0:500-inx, 0:768]
    #         imgShift = imgShift[0:500-inx, 0:768]
    #     else:
    #         imgRef = img1[inx:500, 0:768]
    #         imgShift = imgShift[inx:500, 0:768]

    #     coef = cv2.matchTemplate(imgRef, imgShift, cv2.TM_CCOEFF_NORMED)
    #     print (inx, coef[0][0])

    print ("==========================")
    ### Test Theta ###
    # for inx in range(-19,20):
    #     trans_matrix = np.float32([[1,0,inx],[0,1,0]])
    #     imgShift = cv2.warpAffine(img2, trans_matrix, (768,500))

    #     imgRef = img1[0:500, 0:768-inx]
    #     imgShift = imgShift[0:500, 0:768-inx]

    #     coef = cv2.matchTemplate(imgRef, imgShift, cv2.TM_CCOEFF_NORMED)
    #     print (inx, coef[0])

    