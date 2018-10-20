import cv2
import numpy as np
import matplotlib.pyplot as plt

import sonarPlotting
import FeatureMatch
import regisProcessing as rp

picPath = "D:\Pai_work\pic_sonar"

if __name__ == '__main__':
    img1 = rp.AverageMultiLook(10, 5)
    img2 = rp.AverageMultiLook(15, 5)

    imgRef = rp.BlockImage()
    imgRef.Creat_Block(img1)
    imgRef.Adjsut_block(14)

    imgFloat = rp.BlockImage()
    imgFloat.Creat_Block(img2)
    imgFloat.Adjsut_block(14)

    # TODO : Feature match part
    ## Save Matching for checking SIFT parameter
    # for i in range(0,30):
    #     saveName = picPath + "\exp\Img10_SIFT_Img15_#" + str(i) + ".png"
    #     FeatureMatch.BF_saveMatching(imgRef.blockImg[i], imgFloat.blockImg[i], saveName)

    # xPos, yPos, xOut, yOut = rp.matchingPair(imgRef, imgFloat)

    # paramX, paramY = rp.linear_Approx(xPos, yPos, xOut, yOut)
    # rp.linearRemap(img1, paramX, paramY)

    # paramA, paramB = rp.polynomial_Approx(xPos, yPos, xOut, yOut)
    # rp.polynomialRemap(img1, paramA, paramB)
    
    # TODO : Fourier part
    position = rp.fourierBlockImg(imgRef, imgFloat)
    print (position)