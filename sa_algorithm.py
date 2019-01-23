import cv2
import numpy as np
import random
import csv
import math
# import transformations as tf
from matplotlib import pyplot as plt
import sonarPlotting

from skimage import img_as_ubyte


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
    ref = ref[0:500, 384:768]
    # ref = ref[0:500, 0:348]
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
    # diff_eng = np.power((new_eng - old_eng), 2)
    # prob = np.exp(diff_eng / temp) 
    # diff_eng = new_eng - old_eng
    diff_eng = old_eng - new_eng
    prob = np.exp(diff_eng)
    print ("diff_eng :", diff_eng)
    print ("temp :", temp)
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
    def __init__(self, img1, img2, pose, cfar, cfar2):
        self.row, self.col = img1.shape
        self.init_state = cfar
        self.state_z = cfar2
        # self.state_z = 0.3 * np.ones((self.row,self.col), dtype = float) 
        
        self.observe = 0
        self.old_energyZ = self.energy_init(pose)
        self.new_energyZ = 0
        
        self.temp = 1.0
        self.min_temp = 0.00001
        self.alpha = 0.9
        self.loop_end = True

        self.count_increase = 0
        self.count_decrease = 0

        print ("init_energy :", self.old_energyZ)
        print ("==========================")
        # for i in range(20):
        #     self.update(pose)
        # while(self.loop_end):
        #     self.update(pose)
        while(self.temp > self.min_temp):
            self.update(pose)
            cvt_img = self.state_z / np.max(self.state_z)
            cvt_img = 255.0 * cvt_img
            cvt_img = cvt_img.astype(np.uint8)
            # cv2.imshow("self.init_state", self.init_state)
            cv2.imshow("cvt_img", cvt_img)
            cv2.waitKey(50)
            # cv2.destroyAllWindows()
            print ("----------------")


        # while(1):
        #     commd = input("Continue? :")
        #     if commd == "y":
        #         self.update(pose)
        #     else:
        #         break
        #     print ("==========================")

        if self.temp > self.min_temp:
            print ("MORE")
        else:
            print ("LESS")

        print ("Finish")
        print ("Increase :", self.count_increase, "times")
        print ("Decrease :", self.count_decrease, "times")
        print ("output_max :", np.max(self.state_z))
        print ("output_min :", np.min(self.state_z))

        # print (self.init_state.shape)
        # print (self.state_z.shape)
        # print (self.init_state.dtype)
        # print (self.state_z.dtype)
        sonarPlotting.subplot2(self.init_state, self.state_z, ["1", "2"])
        # res1 = self.init_state.astype(np.uint8)
        # res2 = self.state_z.astype(np.uint8)
        # sonarPlotting.subplot4(self.init_state, self.state_z, res1, res2, ["1", "2", "3", "4"])
        # correlation(self.init_state, self.state_z, 0, 0)
        # coef = cv2.matchTemplate(self.init_state, self.state_z, cv2.TM_CCOEFF_NORMED)
        # print (coef[0][0])
        # cvt_img = self.state_z / np.max(self.state_z)
        # cvt_img = 255 * cvt_img
        # cvt_img = cvt_img.astype(np.uint8)
        # cv2.imshow("self.init_state", self.init_state)
        # cv2.imshow("cvt_img", cvt_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def energy_init(self, pose):
        num = 100
        z_new = 0
        rowShift = pose[0]
        colShift = pose[1]
        trans = np.float32([[1,0,colShift],[0,1,rowShift]])
        z_Shift = cv2.warpAffine(self.state_z, trans, (self.col,self.row))
        # z_ref, z_remap = shift_and_crop(self.init_state, z_Shift, rowShift, colShift)
        z_ref = self.init_state
        z_remap = self.state_z

        row, col = z_ref.shape
        ran_row = np.random.randint(1+15, row + 1-15, size=num)
        ran_col = np.random.randint(1+15, col + 1-15, size=num)
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
        return z_new/num

    def obv_energy(self, img1, img2, pose):
        rowShift = pose[0]
        colShift = pose[1]
        trans = np.float32([[1,0,colShift],[0,1,rowShift]])
        imgShift = cv2.warpAffine(img2, trans, (self.col,self.row))

        img_ref, img_remap = shift_and_crop(img1, imgShift, rowShift, colShift)
        eng = np.power((img_ref - img_remap), 2)
        self.observe = np.sum(eng)
        
    def update(self, pose):
        rand_move = np.random.uniform(low = -0.01, high = 0.01, size = (self.row,self.col))
        self.next_state_z = self.state_z + rand_move

        self.z_energy(pose)

        if self.old_energyZ == 0:
            print ("First Initial")
            self.old_energyZ = self.new_energyZ
            self.state_z = self.next_state_z
        else:
            print ("old_energy :", self.old_energyZ)
            print ("new_energy :", self.new_energyZ)
            if self.new_energyZ < self.old_energyZ:
                print ("Decrease")
                self.count_decrease += 1
                self.state_z = self.next_state_z
                self.old_energyZ = self.new_energyZ
            else:
                print ("Increase")
                self.count_increase += 1
                if self.temp > self.min_temp:
                    prob = accepted_prob(self.new_energyZ, self.old_energyZ, self.temp)
                    print (prob)
                    if prob > random.random():
                        self.state_z = self.next_state_z
                        self.old_energyZ = self.new_energyZ
                    else:
                        self.loop_end = False
                else:
                    self.loop_end = False

                self.temp = self.temp * self.alpha

    def z_energy(self, pose):
        num = 50
        z_new = 0
        rowShift = pose[0]
        colShift = pose[1]
        trans = np.float32([[1,0,colShift],[0,1,rowShift]])
        z_Shift = cv2.warpAffine(self.next_state_z, trans, (self.col,self.row))
        # z_ref, z_remap = shift_and_crop(self.init_state, z_Shift, rowShift, colShift)
        z_ref = self.init_state
        z_remap = self.state_z

        row, col = z_ref.shape
        ran_row = np.random.randint(1+15, row + 1-15, size=num)
        ran_col = np.random.randint(1+15, col + 1-15, size=num)
        for i in range(0,num):
            row_pose = ran_row[i]
            col_pose = ran_col[i]
            sum_diff = 0
            for m in range(row_pose-5, row_pose+6):
                for n in range(col_pose-5, col_pose+6):
                    if m == row_pose and n == col_pose:
                       pass
                    else:
                        diff = z_remap[row_pose][col_pose] - z_ref[m][n]
                        # diff = z_remap[row_pose][col_pose] - z_remap[m][n]
                        diff = np.power(diff ,2)
                        sum_diff = sum_diff + diff
            z_new = z_new + sum_diff
        self.new_energyZ = z_new/num

# if __name__ == '__main__':
#     start = 3
#     stop = 5
#     img1 = read_multilook(start)
#     img2 = read_multilook(stop)
#     speed, pose = positioning(20)
#     pose_Shift = position_shift(pose[start], pose[stop])

#     img_cfar = cafar(img1)
#     blur = cv2.GaussianBlur(img_cfar,(21,21),5)

#     blur = (blur/255.0) * 0.5
#     added = 1 * np.ones((img_cfar.shape), dtype = float) 

#     cvt_img = blur / np.max(blur) # normalize the data to 0 - 1
#     cvt_img = cvt_img * 0.5
#     print ("init_max :", np.max(cvt_img))
#     print ("init_min :", np.min(cvt_img))
    
#     img_cfar2 = cafar(img2)
#     blur2 = cv2.GaussianBlur(img_cfar2,(21,21),5)

#     blur2 = (blur2/255.0) * 0.5

#     cvt_img2 = blur2 / np.max(blur2) # normalize the data to 0 - 1
#     cvt_img2 = cvt_img2 * 0.5

#     cv2.imshow("cvt_img2", cvt_img2)
#     cv2.imshow("cvt_img", cvt_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


#     print ("start simulated annealing...")
#     sa_optimize(img1, img2, pose_Shift, cvt_img, cvt_img2)