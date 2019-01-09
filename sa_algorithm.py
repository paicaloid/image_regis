import cv2
import numpy as np
import random
import csv
import math
import transformations as tf
from matplotlib import pyplot as plt

# # Only testing maximum observe energy
# ! Remove after finish testing
def test_obv(img1, img2, num):
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
            print (rowInx, colInx)
            # ! Quadrant 1
            if rowInx > 0 and colInx > 0:
                print ("Quard 1")
                imgRef = img1[rowInx:row, colInx:col]
                imgShift = imgShift[rowInx:row, colInx:col]
            # ! Quadrant 2
            elif rowInx > 0 and colInx < 0:
                print ("Quard 2")
                imgRef = img1[rowInx:row, 0:col+colInx]
                imgShift = imgShift[rowInx:row, 0:col+colInx]
            # ! Quadrant 3
            elif rowInx < 0 and colInx < 0:
                print ("Quard 3")
                imgRef = img1[0:row+rowInx, 0:col+colInx]
                imgShift = imgShift[0:row+rowInx, 0:col+colInx]
            # ! Quadrant 4
            elif rowInx < 0 and colInx > 0:
                print ("Quard 4")
                imgRef = img1[0:row+rowInx, colInx:col]
                imgShift = imgShift[0:row+rowInx, colInx:col]
            # ! Origin
            elif colInx == 0 and rowInx == 0:
                print ("Origin")
                imgRef = img1[0:row, 0:col]
                imgShift = imgShift[0:row, 0:col]
            # ! row axis
            elif colInx == 0 and rowInx != 0:
                print ("row axis")
                if rowInx > 0:
                    imgRef = img1[rowInx:row, 0:col]
                    imgShift = imgShift[rowInx:row, 0:col]
                elif rowInx < 0:
                    imgRef = img1[0:row+rowInx, 0:col]
                    imgShift = imgShift[0:row+rowInx, 0:col]
            # ! col axis
            elif rowInx == 0 and colInx != 0:
                print ("col axis")
                if colInx > 0:
                    imgRef = img1[0:row, colInx:col]
                    imgShift = imgShift[0:row, colInx:col]
                elif colInx < 0:
                    imgRef = img1[0:row, 0:col+colInx]
                    imgShift = imgShift[0:row, 0:col+colInx]
            
            num_pixel = imgRef.shape[0] * imgRef.shape[1]
            eng = np.power((imgRef - imgShift),2)
            eng = np.sum(eng) / num_pixel
            colList.append(eng)
        coefList.append(max(colList))
        rowList.append((rowInx, max(enumerate(colList), key=(lambda x: x[1]))))
    posShift = rowList[np.argmax(coefList)]
    shiftRow = posShift[0]
    shiftCol = posShift[1][0] - num
    # print (posShift)
    print ("coefShift :", shiftRow, shiftCol)

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

    # print ("poseShift :", row_shift, col_shift)

    if False:
        x_shift = (pos2[1] - pos1[1])
        y_shift = (pos2[2] - pos1[2])
        r_shift = np.sqrt(np.power(x_shift,2) + np.power(y_shift,2))
        theta_shift = np.arctan2(y_shift, x_shift)
        print (y_shift, x_shift)
        print (math.degrees(theta_shift) + 205)
        # print ("RThetaShift :", math.ceil((-1)*r_shift/range_perPixel), math.ceil(math.degrees(theta_shift)/degree_perCol))

    return (row_shift, col_shift)
    # return (math.ceil((-1)*r_shift/range_perPixel), math.ceil(theta_shift/degree_perCol))

def shift_and_crop(img1, img2, rowInx, colInx):
    row, col = img1.shape
    # ! Quadrant 1
    if rowInx > 0 and colInx > 0:
        imgRef = img1[rowInx:row, colInx:col]
        imgShift = img2[rowInx:row, colInx:col]
    # ! Quadrant 2
    elif rowInx > 0 and colInx < 0:
        imgRef = img1[rowInx:row, 0:col+colInx]
        imgShift = img2[rowInx:row, 0:col+colInx]
    # ! Quadrant 3
    elif rowInx < 0 and colInx < 0:
        imgRef = img1[0:row+rowInx, 0:col+colInx]
        imgShift = img2[0:row+rowInx, 0:col+colInx]
    # ! Quadrant 4
    elif rowInx < 0 and colInx > 0:
        imgRef = img1[0:row+rowInx, colInx:col]
        imgShift = img2[0:row+rowInx, colInx:col]
    # ! Origin
    elif colInx == 0 and rowInx == 0:
        imgRef = img1[0:row, 0:col]
        imgShift = img2[0:row, 0:col]
    # ! row axis
    elif colInx == 0 and rowInx != 0:
        if rowInx > 0:
            imgRef = img1[rowInx:row, 0:col]
            imgShift = img2[rowInx:row, 0:col]
        elif rowInx < 0:
            imgRef = img1[0:row+rowInx, 0:col]
            imgShift = img2[0:row+rowInx, 0:col]
    # ! col axis
    elif rowInx == 0 and colInx != 0:
        if colInx > 0:
            imgRef = img1[0:row, colInx:col]
            imgShift = img2[0:row, colInx:col]
        elif colInx < 0:
            imgRef = img1[0:row, 0:col+colInx]
            imgShift = img2[0:row, 0:col+colInx]
    
    return imgRef, imgShift

class sa_optimize:
    def __init__(self, sec1, sec2):
        self.img1 = read_multilook(sec1)
        self.img2 = read_multilook(sec2)
        self.shifting(sec1, sec2)
        self.observe_energy()
        
        self.init_z_energy(100)
        
        self.loop_state = False
        # if (self.loop_state):
        #     self.update_energy(100)
        for i in range(0,100):
            self.update_energy(200)
        
    def shifting(self, sec1, sec2):
        speed, pose = positioning(sec1 + sec2)
        self.poseShift = position_shift(pose[sec1], pose[sec2])
        # print (self.poseShift)

    def observe_energy(self):
        rowInx = self.poseShift[0]
        colInx = self.poseShift[1]
        row, col = self.img1.shape
        trans_matrix = np.float32([[1,0,colInx],[0,1,rowInx]])
        imgShift = cv2.warpAffine(self.img2, trans_matrix, (col,row))
        
        imgRef, imgShift = shift_and_crop(self.img1, imgShift, rowInx, colInx)
        eng = np.power((imgRef - imgShift),2)
        self.obv_energy = np.sum(eng) / 1000.0
        # print (self.obv_energy)

    def init_z_energy(self, num):
        row, col = self.img1.shape
        z_init = 2 * np.ones((row,col), dtype=float) 
        update = np.random.uniform(low=-1.0, high=1.0, size=(row,col))
        z_update = z_init + update

        rowInx = self.poseShift[0]
        colInx = self.poseShift[1]
        row, col = self.img1.shape
        trans_matrix = np.float32([[1,0,colInx],[0,1,rowInx]])
        z_Shift = cv2.warpAffine(z_update, trans_matrix, (col,row))
        z_Ref, z_Shift = shift_and_crop(z_init, z_Shift, rowInx, colInx)

        row, col = z_Ref.shape
        ran_row = np.random.randint(1+5, row + 1-5, size=num)
        ran_col = np.random.randint(1+5, col + 1-5, size=num)
        z_eng = 0
        for i in range(0,num):
            row_pose = ran_row[i]
            col_pose = ran_col[i]
            sum_diff = 0
            for m in range(row_pose-1, row_pose+2):
                for n in range(col_pose-1, col_pose+2):
                    if m == row_pose and n == col_pose:
                       pass
                    else:
                        diff = z_Shift[row_pose][col_pose] - z_Ref[m][n]
                        diff = np.power(diff ,2)
                        sum_diff = sum_diff + diff
            z_eng = z_eng + sum_diff

            # neighbor_ref = z_Ref[row_pose-1:row_pose+2, col_pose-1:col_pose+2]
            # neighbor_Shift = z_Shift[row_pose-1:row_pose+2, col_pose-1:col_pose+2]
            # eng = np.power((neighbor_ref - neighbor_Shift),2)
            # z_eng =  z_eng + np.sum(eng)

        self.z_init = z_update
        self.sum_energy = self.obv_energy + z_eng
        # print ("Obv Energy :", self.obv_energy)
        # print ("Z Energy :", z_eng)
        # print ("Sum Energy :", self.sum_energy)
        # print ("=================")
    
    def update_energy(self, num):
        row, col = self.img1.shape 
        update = np.random.uniform(low=-1.0, high=1.0, size=(row,col))
        z_update = self.z_init + update

        rowInx = self.poseShift[0]
        colInx = self.poseShift[1]
        row, col = self.img1.shape
        trans_matrix = np.float32([[1,0,colInx],[0,1,rowInx]])
        z_Shift = cv2.warpAffine(z_update, trans_matrix, (col,row))
        z_Ref, z_Shift = shift_and_crop(self.z_init, z_Shift, rowInx, colInx)
        
        row, col = z_Ref.shape
        ran_row = np.random.randint(1+5, row + 1-5, size=num)
        ran_col = np.random.randint(1+5, col + 1-5, size=num)
        z_eng = 0
        for i in range(0,num):
            row_pose = ran_row[i]
            col_pose = ran_col[i]
            sum_diff = 0
            for m in range(row_pose-1, row_pose+2):
                for n in range(col_pose-1, col_pose+2):
                    if m == row_pose and n == col_pose:
                       pass
                    else:
                        diff = z_Shift[row_pose][col_pose] - z_Ref[m][n]
                        diff = np.power(diff ,2)
                        sum_diff = sum_diff + diff
            z_eng = z_eng + sum_diff
        sum_new = self.obv_energy + z_eng
        # print ("Obv Energy :", self.obv_energy)
        # print ("Z Energy :", z_eng)
        # print ("Sum Energy :", sum_new)
        # print ("=================")
        # print (sum_new, self.sum_energy)
        if self.loop_state:
            self.z_init = z_update
            self.sum_energy = sum_new
            self.loop_state = False
        else:
            if (sum_new < self.sum_energy):
                print ("+++++   Decrease   +++++")
                self.z_init = z_update
                self.sum_energy = sum_new
            else:
                print ("+++++   Increase   +++++")
                self.sum_energy = sum_new
                self.loop_state = False

        

if __name__ == '__main__':
    sa_optimize(3,4)
    # print ("END")
    img1 = read_multilook(3)
    img2 = read_multilook(4)
    # test_obv(img1, img2, 10)