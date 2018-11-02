import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

import sonarPlotting
import FeatureMatch
import regisProcessing as rp

picPath = "D:\Pai_work\pic_sonar"

if __name__ == '__main__':
    ## construct the argument parse and parse the arguments ##
    ap = argparse.ArgumentParser()
    ap.add_argument("-cf", "--cafar", required=True, help="select cafar")
    ap.add_argument("-m", "--mode", required=True, help="select mode")
    args = vars(ap.parse_args())

    ## Import image ##
    img1 = rp.AverageMultiLook(10, 1)
    img2 = rp.AverageMultiLook(20, 1)
    
    print (img1.shape)
    print (img2.shape)

    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    imgRef = rp.BlockImage()
    imgRef.Creat_Block(img1)
    imgRef.Adjsut_block(14)

    imgFloat = rp.BlockImage()
    imgFloat.Creat_Block(img2)
    imgFloat.Adjsut_block(14)

    # TODO : Test CA-CFAR with blockImage
    if args["cafar"] == "true":
        imgRef.cfarBlock(59, 11, 0.25, 7)
        imgRef.Calculate_Mean_Var()
        # imgRef.Show_cafarBlock()

        imgFloat.cfarBlock(59, 11, 0.25, 7)
        imgFloat.Calculate_Mean_Var()
        # imgFloat.Show_cafarBlock()

        pipeList = []

        for i in range(0,30):
            if imgRef.meanList[i] > imgRef.averageMean*2 or imgFloat.meanList[i] > imgFloat.averageMean*2:
                pipeList.append(i)
        print (pipeList)

    # TODO : Matching
    if args["mode"] == "match":
        xPos, yPos, xOut, yOut = rp.matchingPair(imgRef, imgFloat)
        paramX, paramY = rp.linear_Approx(xPos, yPos, xOut, yOut)
        rp.linearRemap(img1, paramX, paramY)

        paramX, paramY = rp.polynomial_Approx(xPos, yPos, xOut, yOut)
        rp.polynomialRemap(img1, paramX, paramY)

        xPos, yPos, xOut, yOut = rp.matchingSpecific(imgRef, imgFloat, pipeList)
        paramX, paramY = rp.linear_Approx(xPos, yPos, xOut, yOut)
        rp.linearRemap(img1, paramX, paramY)


    # TODO : Fourier part
    if args["mode"] == "fft":
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

        trans_matrix = np.float32([[1,0,sum_col],[0,1,sum_row]])
        res = cv2.warpAffine(img2, trans_matrix, (768,500))

        cv2.imshow("res", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()