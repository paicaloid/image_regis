import cv2
import numpy as np
import random
import csv

import sonarGeometry
import regisProcessing as rp
import FeatureMatch
from matplotlib import pyplot as plt

def multiplyImage(img1, img2):
    # ! Change dtype to avoid overflow (unit8 -> int32)
    if(len(img1) == 3):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if(len(img2) == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = img1.astype(np.int32)
    img2 = img2.astype(np.int32)
    out = cv2.multiply(img1, img2)
    out = out/255.0
    out = out.astype(np.uint8)
    return out

def coefShift(img1, img2, num):
    colRange = np.arange((-1)*num, num+1, 1)
    rowRange = colRange * (-1)
    rowList = []
    coefList = []
    for rowInx in rowRange:
        colList = []
        for colInx in colRange:
            trans_matrix = np.float32([[1,0,colInx],[0,1,rowInx]])
            imgShift = cv2.warpAffine(img2, trans_matrix, (768,500))

            # ! Quadrant 1
            if rowInx > 0 and colInx > 0:
                imgRef = img1[rowInx:500, colInx:768]
                imgShift = imgShift[rowInx:500, colInx:768]
            # ! Quadrant 2
            elif rowInx > 0 and colInx < 0:
                imgRef = img1[rowInx:500, 0:768+colInx]
                imgShift = imgShift[rowInx:500, 0:768+colInx]
            # ! Quadrant 3
            elif rowInx < 0 and colInx < 0:
                imgRef = img1[0:500+rowInx, 0:768+colInx]
                imgShift = imgShift[0:500+rowInx, 0:768+colInx]
            # ! Quadrant 4
            elif rowInx < 0 and colInx > 0:
                imgRef = img1[0:500+rowInx, colInx:768]
                imgShift = imgShift[0:500+rowInx, colInx:768]
            
            coef = cv2.matchTemplate(imgRef, imgShift, cv2.TM_CCOEFF_NORMED)
            colList.append(coef[0][0])
        coefList.append(max(colList))
        rowList.append((rowInx, max(enumerate(colList), key=(lambda x: x[1]))))
    posShift = rowList[np.argmax(coefList)]
    shiftRow = posShift[0]
    shiftCol = posShift[1][0] - num
    print (shiftRow, shiftCol)
    # return shiftRow, shiftCol

def matchingPair(img1, img2):
    refPos, shiftPos = FeatureMatch.matchPosition_BF(img1, img2, "0")
    
    input_row = []
    input_col = []
    for i in refPos:
        xx, yy = i
        input_row.append(xx)
        input_col.append(yy)

    output_row = []
    output_col = []
    for i in shiftPos:
        xx, yy = i
        output_row.append(xx)
        output_col.append(yy)

    return input_row, input_col, output_row, output_col

def coef_remap(img):
    print (img.shape[0])
    print (img.shape[1])
    row, col = np.mgrid[:img.shape[0],:img.shape[1]]
    print (row)
    print (col)
    new_row = row + 14 
    new_col = col + 1

    res = cv2.remap(img, new_col.astype('float32'), new_row.astype('float32'), cv2.INTER_LINEAR)

    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def linear_approx(in_row, in_col, out_row, out_col):
    vectorA = np.vstack([np.ones(len(in_row)), in_row, in_col]).T

    rowList = np.linalg.lstsq(vectorA, out_row)[0]
    colList = np.linalg.lstsq(vectorA, out_col)[0]

    return rowList, colList

def remap_linear(img, list_row, list_col):
    row, col = np.mgrid[:img.shape[0],:img.shape[1]]
    print (row)
    print (col)
    new_row = list_row[0] + (list_row[1]*row) + (list_row[2]*col)
    new_col = list_col[0] + (list_col[1]*row) + (list_col[2]*col)

    res = cv2.remap(img, new_col.astype('float32'), new_row.astype('float32'), cv2.INTER_LINEAR)

    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img1 = rp.AverageMultiLook(10,1)
    img2 = rp.AverageMultiLook(20,1)

    kernel = np.ones((7,7),np.uint8)
    cfar1 = rp.cafar(img1, 59, 11, 0.25)
    cfar1 = cv2.erode(cfar1, kernel, iterations = 1)
    cfar1 = cv2.dilate(cfar1, kernel, iterations = 3)
    mul_1 = multiplyImage(img1, cfar1)

    cfar2 = rp.cafar(img2, 59, 11, 0.25)
    cfar2 = cv2.erode(cfar2, kernel, iterations = 1)
    cfar2 = cv2.dilate(cfar2, kernel, iterations = 3)
    mul_2 = multiplyImage(img2, cfar2)


    # ! Finish run : [-14, -1]
    # coefShift(img1, img2, 15)
    ref_dis = np.power(-14,2) + np.power(-1,2)
    # print (ref_dis)
    # coef_remap(img2)

    # input_row, input_col, output_row, output_col = matchingPair(img2, img1)
    input_row, input_col, output_row, output_col = matchingPair(mul_2, mul_1)

    paramRow, paramCol = linear_approx(input_row, input_col, output_row, output_col)

    remap_linear(img2, paramRow, paramCol)

    r_list = []
    xPos = []
    yPos = []
    xOut = []
    yOut = []

    for i in range(0,len(input_row)):
        row_diff = input_row[i] - output_row[i]
        col_diff = input_col[i] - output_col[i]
        dis = np.power(row_diff,2) + np.power(col_diff,2)
        r_list.append(dis)
        if np.abs(row_diff) < 24 and np.abs(row_diff) > 4:
            xPos.append(input_row[i])
            yPos.append(input_col[i])
            xOut.append(output_row[i])
            yOut.append(output_col[i])

    paramRow, paramCol = linear_approx(xPos, yPos, xOut, yOut)

    remap_linear(img2, paramRow, paramCol)

    
