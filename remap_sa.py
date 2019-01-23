import cv2
import numpy as np
import random
import csv
import math
# import transformations as tf
from matplotlib import pyplot as plt
import sonarPlotting

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

def accepted_prob(new_eng, old_eng, temp):
    diff_eng = np.power((new_eng - old_eng), 2)
    prob = np.exp(diff_eng) / temp
    return prob

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

class sa_optimize:
    def __init__(self, img1, img2, pose):
        self.row, self.col = img1.shape
        self.img_init = img1
        self.state = img2 
        self.old_energyZ = self.obv_energy(img1, img2)
        self.new_energyZ = 0
        self.row_state = 0
        self.col_state = 0
        
        self.temp = 1.0
        self.min_temp = 0.00001
        self.alpha = 0.9
        self.loop_end = True

        self.count_increase = 0
        self.count_decrease = 0

        while(self.loop_end):
            self.annealing()

        if self.temp > self.min_temp:
            print ("MORE")
        else:
            print ("LESS")

        print ("Increase :", self.count_increase, "times")
        print ("Decrease :", self.count_decrease, "times")

        print (self.row_state, self.col_state)
        print (pose[0], pose[1])

    def annealing(self):
        rand_row = np.random.random_integers(low = -1, high = 1)
        rand_col = np.random.random_integers(low = -1, high = 1)
        row_next = self.row_state + rand_row
        col_next = self.col_state + rand_col

        trans = np.float32([[1,0,col_next],[0,1,row_next]])
        imgShift = cv2.warpAffine(self.state, trans, (self.col,self.row))
        img_ref, img_shift = shift_and_crop(self.img_init, imgShift, rand_row, rand_col)

        self.new_energyZ = self.obv_energy(img_ref, img_shift)

        if self.new_energyZ < self.old_energyZ:
                # print ("Decrease")
                self.count_decrease += 1
                self.old_energyZ = self.new_energyZ
                self.row_state = row_next
                self.col_state = col_next
        else:
            # print ("Increase")
            self.count_increase += 1
            if self.temp > self.min_temp:
                prob = accepted_prob(self.new_energyZ, self.old_energyZ, self.temp)
                if prob > random.random():
                    # self.state_z = self.next_state_z
                    self.old_energyZ = self.new_energyZ
                    self.row_state = row_next
                    self.col_state = col_next
                else:
                    self.loop_end = False
            else:
                self.loop_end = False
            self.temp = self.temp * self.alpha

    def obv_energy(self, img1, img2):
        eng = np.power((img1 - img2), 2)
        self.observe = np.sum(eng)
        return self.observe
        
    def update(self, pose):
        rand_move = np.random.uniform(low = -0.05, high = 0.05, size = (self.row,self.col))
        self.next_state_z = self.state_z + rand_move

        self.z_energy(pose)

        if self.old_energyZ == 0:
            print ("First Initial")
            self.old_energyZ = self.new_energyZ
            self.state_z = self.next_state_z
        else:
            # print (self.new_energyZ, self.old_energyZ)
            if self.new_energyZ < self.old_energyZ:
                print ("Decrease")
                self.state_z = self.next_state_z
                self.old_energyZ = self.new_energyZ
            else:
                print ("Increase")
                if self.temp > self.min_temp:
                    prob = accepted_prob(self.new_energyZ, self.old_energyZ, self.temp)
                    if prob > random.random():
                        self.state_z = self.next_state_z
                        self.old_energyZ = self.new_energyZ
                    else:
                        self.loop_end = False
                else:
                    self.loop_end = False

                self.temp = self.temp * self.alpha

    def z_energy(self, pose):
        num = 100
        z_new = 0
        rowShift = pose[0]
        colShift = pose[1]
        trans = np.float32([[1,0,colShift],[0,1,rowShift]])
        z_Shift = cv2.warpAffine(self.next_state_z, trans, (self.col,self.row))
        z_ref, z_remap = shift_and_crop(self.state_z, z_Shift, rowShift, colShift)
        
        row, col = z_ref.shape
        ran_row = np.random.randint(1+5, row + 1-5, size=num)
        ran_col = np.random.randint(1+5, col + 1-5, size=num)
        for i in range(0,num):
            row_pose = ran_row[i]
            col_pose = ran_col[i]
            sum_diff = 0
            for m in range(row_pose-1, row_pose+2):
                for n in range(col_pose-1, col_pose+2):
                    if m == row_pose and n == col_pose:
                       pass
                    else:
                        diff = z_remap[row_pose][col_pose] - z_ref[m][n]
                        # diff = z_remap[row_pose][col_pose] - z_remap[m][n]
                        diff = np.power(diff ,2)
                        sum_diff = sum_diff + diff
            z_new = z_new + sum_diff
        self.new_energyZ = z_new

if __name__ == '__main__':
    start = 3
    stop = 5
    img1 = read_multilook(start)
    img2 = read_multilook(stop)
    speed, pose = positioning(20)
    pose_Shift = position_shift(pose[start], pose[stop])

    sa_optimize(img1, img2, pose_Shift)

    # rand = np.random.random_integers(low = -10, high = 10)
    # print (rand)