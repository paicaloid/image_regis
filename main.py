import cv2
import numpy as np
import matplotlib.pyplot as plt

import sonarPlotting
import FeatureMatch
import regisProcessing as rp

picPath = "D:\Pai_work\pic_sonar"

if __name__ == '__main__':
    img1 = rp.AverageMultiLook(10, 10)
    img2 = rp.AverageMultiLook(30, 10)

    cv2.imshow("ref", img1)
    cv2.imshow("float", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    imgRef = rp.BlockImage()
    imgRef.Creat_Block(img1)
    imgRef.Adjsut_block(14)
    # imgRef.Show_allBlock()

    imgFloat = rp.BlockImage()
    imgFloat.Creat_Block(img2)
    imgFloat.Adjsut_block(14)
    # imgFloat.Show_allBlock()

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

    # TODO : Test CA-CFAR
    # kernel = np.ones((7,7),np.uint8)
    # res = rp.cafar(img1, 41, 15, 0.27)

    # opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

    # cv2.imshow("img", res)
    # cv2.imshow("CFAR", opening)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # TODO : Test CA-CFAR with blockImage
    imgRef.cfarBlock(41, 7, 0.29, 7)
    imgRef.Calculate_Mean_Var()
    imgRef.Show_cafarBlock()

    imgFloat.cfarBlock(41, 7, 0.29, 7)
    imgFloat.Calculate_Mean_Var()
    imgFloat.Show_cafarBlock()

    pipeList = []

    for i in range(0,30):
        if imgRef.meanList[i] > imgRef.averageMean or imgFloat.meanList[i] > imgFloat.averageMean:
            # print (imgRef.meanList[i], imgFloat.meanList[i])
            pipeList.append(i)
    
    print (pipeList)

    xPos, yPos, xOut, yOut = rp.matchingPair(imgRef, imgFloat)
    paramX, paramY = rp.linear_Approx(xPos, yPos, xOut, yOut)
    rp.linearRemap(img1, paramX, paramY)

    paramX, paramY = rp.polynomial_Approx(xPos, yPos, xOut, yOut)
    rp.polynomialRemap(img1, paramX, paramY)

    xPos, yPos, xOut, yOut = rp.matchingSpecific(imgRef, imgFloat, pipeList)
    paramX, paramY = rp.linear_Approx(xPos, yPos, xOut, yOut)
    rp.linearRemap(img1, paramX, paramY)


    # TODO : Fourier part
    position, fixPosition = rp.fourierBlockImg(imgRef, imgFloat)
    
    sum_row = 0
    sum_col = 0
    countRow = 0
    countCol = 0

    for i in pipeList:
        # if fixPosition[i][0] != 0 or fixPosition[i][1] != 0:
        print (fixPosition[i])

        # if fixPosition[i][0] != 0:
        sum_row = sum_row + fixPosition[i][0]
        countRow = countRow + 1
        # if fixPosition[i][1] != 0:
        sum_col = sum_col + fixPosition[i][1]
        countCol = countCol + 1
    
    if countRow != 0:
        sum_row = int(sum_row/countRow)
    if countCol != 0:
        sum_col = int(sum_col/countCol)

    print (sum_row, sum_col)

    trans_matrix = np.float32([[1,0,sum_row],[0,1,sum_col]])
    res = cv2.warpAffine(img2, trans_matrix, (768,500))

    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()