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
    print (shiftRow, shiftCol)
    # return shiftRow, shiftCol

def matchingPair(img1, img2):
    refPos, shiftPos = FeatureMatch.matchPosition_BF(img1, img2, "0")
    # refPos, shiftPos = FeatureMatch.FLANN_saveMatching(img1, img2, "0")
    
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

def polynomial_approx(in_row, in_col, out_row, out_col):
    rowPow_in = np.power(in_row, 2)
    colPow_in = np.power(in_col, 2)
    row_col = np.asarray(in_row) * np.asarray(in_col)
    row_col = row_col.tolist()

    ## rewrite the line equation as x = Ap
    ## where A = [[1 x y x^2 y^2 xy]] 
    ## and p = [a0, a1, a2, a3, a4, a5]
    vectorA = np.vstack([np.ones(len(in_row)), in_row, in_col, rowPow_in, colPow_in, row_col]).T
    listA = np.linalg.lstsq(vectorA, out_row)[0]
    
    ## rewrite the line equation as y = Aq
    ## where B = [[1 x y x^2 y^2 xy]] 
    ## and q = [b0, b1, b2, b3, b4, b5]
    listB = np.linalg.lstsq(vectorA, out_col)[0]

    return listA, listB

def remap_linear(img, list_row, list_col):
    row, col = np.mgrid[:img.shape[0],:img.shape[1]]
    # print (row)
    # print (col)
    new_row = list_row[0] + (list_row[1]*row) + (list_row[2]*col)
    new_col = list_col[0] + (list_col[1]*row) + (list_col[2]*col)

    res = cv2.remap(img, new_col.astype('float32'), new_row.astype('float32'), cv2.INTER_LINEAR)

    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return res

def remap_poly(img, list_row, list_col):
    row, col = np.mgrid[:img.shape[0],:img.shape[1]]
    new_x = list_row[0] + (list_row[1]*row) + (list_row[2]*col) + (list_row[3]*np.power(row,2)) + (list_row[4]*np.power(col,2)) + (list_row[5]*row*col)
    new_y = list_col[0] + (list_col[1]*row) + (list_col[2]*col) + (list_col[3]*np.power(row,2)) + (list_col[4]*np.power(col,2)) + (list_col[5]*row*col)

    res = cv2.remap(img,new_x.astype('float32'),new_y.astype('float32'),cv2.INTER_LINEAR)

    cv2.imshow("Img", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return res

def draw_matching(img1, img2, rowOut, colOut, rowIn, colIn, inx):
    img = np.hstack((img1, img1))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    row, col = img1.shape
    for i in inx:
        rand_color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
        # center_1 = (int(rowOut[i]), int(colOut[i]))
        # center_2 = (int(rowIn[i]), int(colIn[i] + col))
        center_1 = (int(colOut[i]), int(rowOut[i]))
        center_2 = (int(colIn[i] + col), int(rowIn[i]))
        cv2.circle(img, center_1, 5, rand_color, -1)
        cv2.circle(img, center_2, 5, rand_color, -1)
        cv2.line(img, center_1, center_2, rand_color, 1)
    cv2.imshow("match image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def dvl_all(sec):
    first_check = True
    xPos = []
    yPos = []
    zPos = []
    auv_state = []
    time = 0
    position_x = 0
    position_y = 0
    position_z = 0
    with open('recordDVL.csv') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for data in reader:
            if first_check:
                first_check = False
                init_time = int(data[2][0:10])
            else:
                if (init_time + time == int(data[2][0:10])):
                    xPos.append(float(data[3]))
                    yPos.append(float(data[4]))
                    zPos.append(float(data[5]))
                elif (init_time + time + 1 == int(data[2][0:10])):
                    time += 1
                    x_speed = np.mean(xPos)
                    y_speed = np.mean(yPos)
                    z_speed = np.mean(zPos)
                    position_x = (x_speed * time) + position_x
                    position_y = (y_speed * time) + position_y
                    position_z = (z_speed * time) + position_z
                    auv_state.append((time, position_x, position_y, position_z))
                    xPos = []
                    yPos = []
                    zPos = []
                    xPos.append(float(data[3]))
                    yPos.append(float(data[4]))
                    zPos.append(float(data[5]))
                    if (time == sec):
                        break
        
        return auv_state

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
        return auv_state

if __name__ == '__main__':
    img1 = rp.AverageMultiLook(9,1)
    img1 = img1[0:500, 0+5:768-5]
    img2 = rp.AverageMultiLook(12,1)
    img2 = img2[0:500, 0+5:768-5]

    print (img1.shape)

    kernel = np.ones((7,7),np.uint8)
    cfar1 = rp.cafar(img1, 59, 11, 0.25)
    cfar1 = cv2.erode(cfar1, kernel, iterations = 1)
    cfar1 = cv2.dilate(cfar1, kernel, iterations = 3)
    mul_1 = multiplyImage(img1, cfar1)

    cfar2 = rp.cafar(img2, 59, 11, 0.25)
    cfar2 = cv2.erode(cfar2, kernel, iterations = 1)
    cfar2 = cv2.dilate(cfar2, kernel, iterations = 3)
    mul_2 = multiplyImage(img2, cfar2)

    # ! Finish run : [-4, 0]
    # coefShift(img1, img2, 10)
    # ref_dis = np.power(-14,2) + np.power(-1,2)
    # print (ref_dis)
    # coef_remap(img2)

    if True:
        range_perPixel = 0.04343
        image_perSec = 3.2
        degree_perCol = 0.169

        # print (int(3 * image_perSec))
        # print (int(4 * image_perSec))

        auv_state_1 = read_dvl(3)
        auv_state_2 = read_dvl(4)

        row_shift = np.abs(auv_state_1[0][1] - auv_state_2[0][1]) / range_perPixel
        col_shift = np.abs(auv_state_1[0][2] - auv_state_2[0][2]) / degree_perCol
        print (row_shift)
        print (col_shift)

        # print (auv_state_1)
        # print (auv_state_2)

        # pos = dvl_all(5)

        # print (pos[2][1])

        # row_shift = np.abs(pos[3][1]) / range_perPixel
        # col_shift = np.abs(pos[2][2] - pos[3][2]) / degree_perCol
        # print (row_shift)
        # print (col_shift)
        # print (pos)

    if False:
        output_row, output_col, input_row, input_col = matchingPair(img1, img2)
        # input_row, input_col, output_row, output_col = matchingPair(mul_2, mul_1)

        r_list = []
        rowIn = []
        colIn = []
        rowOut = []
        colOut = []

        for i in range(0,len(input_row)):
            row_diff = input_row[i] - output_row[i]
            col_diff = input_col[i] - output_col[i]
            dis = np.power(row_diff,2) + np.power(col_diff,2)
            r_list.append(dis)
        
        dis_mean = np.mean(r_list)
        dis_med = np.median(r_list)
        print (dis_mean, dis_med)
        dis_index = []
        for i in range(len(r_list)):
            if r_list[i] < dis_med:
                dis_index.append(i)
        print (dis_index)

        for i in dis_index:
            rowIn.append(input_row[i])
            colIn.append(input_col[i])
            rowOut.append(output_row[i])
            colOut.append(output_col[i])
        
        draw_matching(img1, img2, output_row, output_col, input_row, input_col, dis_index)

        paramRow, paramCol = linear_approx(input_row, input_col, output_row, output_col)
        print (len(paramRow))
        img_remap =  remap_linear(img2, paramRow, paramCol)

        paramRow, paramCol = linear_approx(rowIn, colIn, rowOut, colOut)
        img_remap_upgrade =  remap_linear(img2, paramRow, paramCol)

        paramRow, paramCol = polynomial_approx(input_row, input_col, output_row, output_col)
        print (len(paramRow))
        img_poly = remap_poly(img2, paramRow, paramCol)

        paramRow, paramCol = polynomial_approx(rowIn, colIn, rowOut, colOut)
        img_poly_upgrade = remap_poly(img2, paramRow, paramCol)

    if False:
        coef = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        print ("No remap : " + str(coef[0][0]))
        coef = cv2.matchTemplate(img1, img_remap, cv2.TM_CCOEFF_NORMED)
        print ("Linear approx : " + str(coef[0][0]))
        coef = cv2.matchTemplate(img1, img_remap_upgrade, cv2.TM_CCOEFF_NORMED)
        print ("Linear approx reduce : " + str(coef[0][0]))
        coef = cv2.matchTemplate(img1, img_poly, cv2.TM_CCOEFF_NORMED)
        print ("Poly approx reduce : " + str(coef[0][0]))
        coef = cv2.matchTemplate(img1, img_poly_upgrade, cv2.TM_CCOEFF_NORMED)
        print ("Poly approx reduce : " + str(coef[0][0]))

        row, col = img1.shape
        trans_matrix = np.float32([[1,0,0],[0,1,-4]])
        imgShift = cv2.warpAffine(img2, trans_matrix, (col,row))
        img1 = img1[0:row-2,0:col]
        imgShift = imgShift[0:row-2,0:col]

        coef = cv2.matchTemplate(img1, imgShift, cv2.TM_CCOEFF_NORMED)
        print ("Coef shift : " + str(coef[0][0]))