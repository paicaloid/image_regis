import cv2
import numpy as np
import random
import csv
import math
import transformations as tf
from matplotlib import pyplot as plt

def read_image(sec):
    image_perSec = 3.3
    picPath = "D:\Pai_work\pic_sonar\\"
    picName = "RTheta_img_" + str(math.ceil(sec*image_perSec)) + ".jpg"
    img = cv2.imread(picPath + picName, 0)
    print ("success import...   " + picName)
    # img = img[0:500, 0:768]
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
    # ref = ref[0:500, 0:768]
    return ref

def positioning(sec):
    xSpeed = []
    ySpeed = []
    zSpeed = []
    first_check = True
    inx = 0
    auv_state = []
    with open('recordDVL.csv') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for data in reader:
            if (first_check):
                first_check = False
                init_time = int(data[2][0:10])
                xSpeed.append(float(data[3]))
                ySpeed.append(float(data[4]))
                zSpeed.append(float(data[5]))
            else:
                if (int(data[2][0:10]) == init_time + inx):
                    xSpeed.append(float(data[3]))
                    ySpeed.append(float(data[4]))
                    zSpeed.append(float(data[5]))
                else:
                    xMean = np.mean(xSpeed)
                    yMean = np.mean(ySpeed)
                    zMean = np.mean(zSpeed)
                    auv_state.append((inx, xMean, yMean, zMean))

                    inx += 1
                    # init_time = int(data[2][0:10])
                    xSpeed = []
                    ySpeed = []
                    zSpeed = []
                    xSpeed.append(float(data[3]))
                    ySpeed.append(float(data[4]))
                    zSpeed.append(float(data[5]))

                if (inx == sec):
                    break
        xPos = 0.0
        yPos = 0.0
        zPos = 0.0
        pose = []
        for state in auv_state:
            if (state[0] > 0):
                xPos = xPos + state[1]
                yPos = yPos + state[2]
                zPos = zPos + state[3]
                pose.append((state[0], xPos, yPos, zPos))
        return auv_state, pose

def position_shift(pos1, pos2):
    range_perPixel = 0.04343
    degree_perCol = 0.169

    row_shift = math.ceil((pos2[1] - pos1[1]) / range_perPixel)
    col_shift = math.ceil((pos2[2] - pos1[2]) / degree_perCol)

    print ("poseShift :", row_shift, col_shift)

    if True:
        x_shift = (pos2[1] - pos1[1])
        y_shift = (pos2[2] - pos1[2])
        r_shift = np.sqrt(np.power(x_shift,2) + np.power(y_shift,2))
        theta_shift = np.arctan2(y_shift, x_shift)
        print (y_shift, x_shift)
        print (math.degrees(theta_shift) + 205)
        print ("RThetaShift :", math.ceil((-1)*r_shift/range_perPixel), math.ceil(math.degrees(theta_shift)/degree_perCol))

    # return (row_shift, col_shift)
    # return (math.ceil((-1)*r_shift/range_perPixel), math.ceil(theta_shift/degree_perCol))

def observe_energy(img1, img2):
    row, col = img1.shape
    img1 = img1.astype(np.int32)
    img2 = img2.astype(np.int32)
    energy = img1 - img2
    energy = np.power(energy, 2)
    obv_eng = np.sum(energy)
    print (obv_eng/(row*col))

def create_shift(img1, img2, shift_row, shift_col):
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

    return img1, img2

class elevation:
    def __init__(self, img):
        if(len(img) == 3):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        self.row, self.col = img.shape
        self.pixel_per_range = 0.04343
        self.pixel_per_degree = 0.169
        self.img = img
        self.elv = []

        for i in range(1, self.row +1):
            dis = i * self.pixel_per_range
            elv = np.arctan2(2, dis)
            elv = math.degrees(elv)
            if (elv < 20.0):
                self.elv.append((i, elv))

if __name__ == '__main__':

    img1 = read_multilook(3)
    ele_map1 = elevation(img1)
    print (type(img1))
    img2 = read_multilook(5)
    ele_map2 = elevation(img2)

    observe_energy(img1, img1)
    # img11, img22 = create_shift(img1, img2, -8, 0)
    # observe_energy(img11, img22)
    # img11, img22 = create_shift(img1, img2, -8, 1)
    # observe_energy(img11, img22)
    num = 10
    colRange = np.arange((-1)*num, num+1, 1)
    rowRange = colRange * (-1)
    # for rowInx in rowRange:
    #     img11, img22 = create_shift(img1, img2, rowInx, 0)
    #     print (rowInx)
    #     observe_energy(img11, img22)

    # print (ele_map1.elv)
    # cv2.imshow("ele_map1", ele_map1.img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()