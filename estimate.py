import cv2
import numpy as np
import random
import csv
import math
import FeatureMatch
import regisProcessing as rp
import coef_shifting as coef
from matplotlib import pyplot as plt

def cafar(img_gray):
    box_size = 59
    guard_size = 11
    pfa = 0.25
    if(len(img_gray) == 3):
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    n_ref_cell = (box_size * box_size) - (guard_size * guard_size)
    alpha = n_ref_cell * (math.pow(pfa, (-1.0/n_ref_cell)) - 1)

    ## Create kernel
    kernel_beta = np.ones([box_size, box_size], 'float32')
    width = round((box_size - guard_size) / 2.0)
    height = round((box_size - guard_size) / 2.0)
    kernel_beta[width: box_size - width, height: box_size - height] = 0.0
    kernel_beta = (1.0 / n_ref_cell) * kernel_beta
    beta_pow = cv2.filter2D(img_gray, ddepth = cv2.CV_32F, kernel = kernel_beta)
    thres = alpha * (beta_pow)
    out = (255 * (img_gray > thres)) + (0 * (img_gray < thres))
    out = out.astype(np.uint8)

    kernel = np.ones((7,7),np.uint8)
    out = cv2.erode(out, kernel, iterations = 1)
    out = cv2.dilate(out, kernel, iterations = 3)
    return out

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
    img = img[0:500, 0:768]
    return img

def read_multilook(sec):
    image_perSec = 3.3
    picPath = "D:\Pai_work\pic_sonar\\"
    start = math.ceil(sec * image_perSec)
    stop = math.ceil((sec + 1) * image_perSec)

    picName = "RTheta_img_" + str(start) + ".jpg"
    ref = cv2.imread(picPath + picName, 0)

    for i in range(start + 1, stop):
        picName = "RTheta_img_" + str(i) + ".jpg"
        img = cv2.imread(picPath + picName, 0)
        # print ("multilook...   " + picName)
        ref = cv2.addWeighted(ref, 0.5, img, 0.5, 0)
    picName = "RTheta_img_" + str(start) + ".jpg"
    # print ("success multilook...   " + picName)
    ref = ref[0:500, 0:768]
    return ref

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
    return shiftRow, shiftCol

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
        # print ("import... auv position @sec", sec)
        return auv_state[0]

def dvlShift(sec1, sec2):
    range_perPixel = 0.04343
    degree_perCol = 0.169

    state_1 = read_dvl(sec1)
    state_2 = read_dvl(sec2)

    row_shift = math.ceil((state_2[1] - state_1[1]) / range_perPixel)
    col_shift = math.ceil((state_2[2] - state_1[2]) / degree_perCol)

    print ("dvlShift :", row_shift, col_shift)
    return (row_shift, col_shift)

def merge_image(img1, img2, rowShift, colShift):
    row, col = img1.shape
    band_1 = cafar(img1)
    band_2 = cafar(img2)
    band_3 = np.zeros((row, col),np.uint8)

    res = cv2.merge((band_1, band_2, band_3))

    mask_and = cv2.bitwise_and(band_1, band_2)

    cv2.imshow("res", res)
    # cv2.imshow("mask_and", mask_and)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if True:
        trans_matrix = np.float32([[1,0,colShift],[0,1,rowShift]])
        band_2 = cv2.warpAffine(band_2, trans_matrix, (col,row))
        res = cv2.merge((band_1, band_2, band_3))
        cv2.imshow("res2", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':

    if False:
        img1 = read_image(3)
        img2 = read_image(4)
        mul1 = read_multilook(6)
        mul2 = read_multilook(7)

        # coefShift(img1, img2, 10)

        pos_shift = dvlShift(6,7)

        # correlation(img1, img3, -8, -2)
        # correlation(img1, img3, -9, -1)

        # correlation(mul1, mul3, -8, -2)
        correlation(img1, img2, pos_shift[0], pos_shift[1])

        merge_image(img1, img2, pos_shift[0], pos_shift[1])

        # cv2.imshow("img1", img1)
        # cv2.imshow("res", res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    if True:
        for i in range(1,21):
            mul1 = read_multilook(i)
            mul2 = read_multilook(i+1)
            
            pos_shift = dvlShift(i,i+1)
            correlation(mul1, mul2, pos_shift[0], pos_shift[1])

            # pose = coefShift(mul1, mul2, 10)
            # correlation(mul1, mul2, pose[0], pose[1])
    
    if False:
        img1 = rp.AverageMultiLook(70,1)
        img2 = rp.AverageMultiLook(71,1)
        img3 = rp.AverageMultiLook(72,1)
        img4 = rp.AverageMultiLook(73,1)
        img5 = rp.AverageMultiLook(74,1)

        res = coef.shiftImage(img1, img2, img3, img4, img5, 15)

        cv2.imshow("img", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()