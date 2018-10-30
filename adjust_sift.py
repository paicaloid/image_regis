import cv2
import numpy as np
import sonarGeometry
import regisProcessing as rp
from matplotlib import pyplot as plt

nfeatures = 0
nOctaveLayers = 3
contrastThreshold = 0.12
edgeThreshold = 10
sigma = 1.6

def matchPosition_BF(img1, img2, savename):
    ref_Position = []
    shift_Position = []
    try:
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create(nfeatures, nOctaveLayers, 
                                            contrastThreshold, edgeThreshold, sigma)

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        # Apply ratio test
        good = []
        
        inx = 0
    
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
                ref_Position.append(kp1[m.queryIdx].pt)
                shift_Position.append(kp2[m.trainIdx].pt)
  
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

        # print (savename)
        if len(good) > 0 :
            plt.imshow(img3),plt.savefig(savename)
        return ref_Position, shift_Position
    except:
        return ref_Position, shift_Position

if __name__ == '__main__':
    img1 = rp.AverageMultiLook(10,1)
    img2 = rp.AverageMultiLook(20,1)

    # cv2.imshow("img10", img1)
    # cv2.imshow("img20", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    imgLeft = rp.BlockImage()
    imgLeft.Creat_Block(img1)
    imgLeft.Adjsut_block(14)

    imgRight = rp.BlockImage()
    imgRight.Creat_Block(img2)
    imgRight.Adjsut_block(14)

    savePath = "D:\Pai_work\pic_sonar\\result\SIFT"

    ref_Position = []
    shift_Position = []

    for i in range(0,30):
        fileName = "\SIFT_" + str(i) + ".jpg"
        refPos, shiftPos = matchPosition_BF(imgLeft.blockImg[i], imgRight.blockImg[i], savePath + fileName)
        if len(refPos) != 0:
            for m in refPos:
                ref_Position.append(m)
        if len(shiftPos) != 0:
            for n in shiftPos:
                shift_Position.append(n)
    input_row = []
    input_col = []
    for i in ref_Position:
        xx, yy = i
        input_row.append(xx)
        input_col.append(yy)

    output_row = []
    output_col = []
    for i in shift_Position:
        xx, yy = i
        output_row.append(xx)
        output_col.append(yy)

    print (len(input_row))
    # print (len(input_col))
    # print (len(output_row))
    # print (len(output_col))

    paramX, paramY = rp.linear_Approx(input_row, input_col, output_row, output_col)
    rp.linearRemap(img1, paramX, paramY)