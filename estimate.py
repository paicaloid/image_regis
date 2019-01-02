import cv2
import numpy as np
import random
import csv
import math
import FeatureMatch
from matplotlib import pyplot as plt

def correlation(img1, img2, shift_row, shift_col):
    row, col = img1.shape
    trans_matrix = np.float32([[1,0,shift_col],[0,1,shift_row]])
    img2 = cv2.warpAffine(img2, trans_matrix, (col,row))
    # ! Quadrant 1
    if shift_row > 0 and shift_col > 0:
        img1 = img1[shift_row:row, shift_col:col]
        img2 = img2[shift_row:row, shift_col:col]
    # ! Quadrant 2
    elif shift_row > 0 and shift_col < 0:
        img1 = img1[shift_row:row, 0:col+shift_col]
        img2 = img2[shift_row:row, 0:col+shift_col]
    # ! Quadrant 3
    elif shift_row < 0 and shift_col < 0:
        img1 = img1[0:row+shift_row, 0:col+shift_col]
        img2 = img2[0:row+shift_row, 0:col+shift_col]
    # ! Quadrant 4
    elif shift_row < 0 and shift_col > 0:
        img1 = img1[0:row+shift_row, shift_col:col]
        img2 = img2[0:row+shift_row, shift_col:col]

    coef = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
    print (coef[0][0])

def read_image(sec):
    image_perSec = 3.3
    picPath = "D:\Pai_work\pic_sonar\\"
    picName = "RTheta_img_" + str(math.ceil(sec*image_perSec)) + ".jpg"
    img = cv2.imread(picPath + picName, 0)
    print ("success import...   " + picName)
    return img

def coefShift(img1, img2, num):
    colRange = np.arange((-1)*num, num+1, 1)
    rowRange = colRange * (-1)
    rowList = []
    coefList = []
    row, col = img1.shape
    for rowInx in rowRange:
        colList = []
        for colInx in colRange:
            trans_matrix = np.float32([[1,0,colInx],[0,1,rowInx]])
            imgShift = cv2.warpAffine(img2, trans_matrix, (col,row))

            # ! Quadrant 1
            if rowInx > 0 and colInx > 0:
                imgRef = img1[rowInx:row, colInx:col]
                imgShift = imgShift[rowInx:row, colInx:col]
            # ! Quadrant 2
            elif rowInx > 0 and colInx < 0:
                imgRef = img1[rowInx:row, 0:col+colInx]
                imgShift = imgShift[rowInx:row, 0:col+colInx]
            # ! Quadrant 3
            elif rowInx < 0 and colInx < 0:
                imgRef = img1[0:row+rowInx, 0:col+colInx]
                imgShift = imgShift[0:row+rowInx, 0:col+colInx]
            # ! Quadrant 4
            elif rowInx < 0 and colInx > 0:
                imgRef = img1[0:row+rowInx, colInx:col]
                imgShift = imgShift[0:row+rowInx, colInx:col]
            
            coef = cv2.matchTemplate(imgRef, imgShift, cv2.TM_CCOEFF_NORMED)
            colList.append(coef[0][0])
        coefList.append(max(colList))
        rowList.append((rowInx, max(enumerate(colList), key=(lambda x: x[1]))))
    posShift = rowList[np.argmax(coefList)]
    shiftRow = posShift[0]
    shiftCol = posShift[1][0] - num
    print ("coefShift :", shiftRow, shiftCol)

def read_dvl(sec):
    first_check = True
    xPos = []
    yPos = []
    zPos = []
    auv_state = []
    with open('recordDVL.csv') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for data in reader:
            if first_check:
                first_check = False
                init_time = int(data[2][0:10])
            else:
                if (init_time + sec == int(data[2][0:10])):
                    xPos.append(float(data[3]))
                    yPos.append(float(data[4]))
                    zPos.append(float(data[5]))
                elif (init_time + sec < int(data[2][0:10])):
                    auv_state.append((init_time + sec, np.mean(xPos) * sec, np.mean(yPos) * sec, np.mean(zPos) * sec))
                    break
        print ("import... auv position @sec", sec)
        return auv_state[0]

def dvlShift(sec1, sec2):
    range_perPixel = 0.04343
    degree_perCol = 0.169

    state_1 = read_dvl(3)
    state_2 = read_dvl(5)

    row_shift = math.ceil((state_2[1] - state_1[1]) / range_perPixel)
    col_shift = math.ceil((state_2[2] - state_1[2]) / degree_perCol)

    print ("dvlShift :", row_shift, col_shift)

if __name__ == '__main__':

    img1 = read_image(3)
    img2 = read_image(5)

    # coefShift(img1, img2, 10)

    dvlShift(3,5)

    correlation(img1, img2, -8, -2)
    correlation(img1, img2, -9, -1)

    # cv2.imshow("img1", img1)
    # cv2.imshow("img2", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()