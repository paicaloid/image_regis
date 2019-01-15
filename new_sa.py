import cv2
import numpy as np
import random
import csv
import math
# import transformations as tf
from matplotlib import pyplot as plt
from skimage.transform import warp
from skimage import img_as_ubyte
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
    diff_eng = (old_eng - new_eng)
    prob = np.exp(diff_eng)
    # print ("diff_eng :", diff_eng)
    # print ("temp :", temp)
    return prob

def correlation(img1, img2, rowInx, colInx):
    row, col = img1.shape
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
    # ! Origin
    elif colInx == 0 and rowInx == 0:
        # print ("Origin")
        imgRef = img1[0:row, 0:col]
        imgShift = imgShift[0:row, 0:col]
    # ! row axis
    elif colInx == 0 and rowInx != 0:
        # print ("row axis")
        if rowInx > 0:
            imgRef = img1[rowInx:row, 0:col]
            imgShift = imgShift[rowInx:row, 0:col]
        elif rowInx < 0:
            imgRef = img1[0:row+rowInx, 0:col]
            imgShift = imgShift[0:row+rowInx, 0:col]
    # ! col axis
    elif rowInx == 0 and colInx != 0:
        # print ("col axis")
        if colInx > 0:
            imgRef = img1[0:row, colInx:col]
            imgShift = imgShift[0:row, colInx:col]
        elif colInx < 0:
            imgRef = img1[0:row, 0:col+colInx]
            imgShift = imgShift[0:row, 0:col+colInx]

    coef = cv2.matchTemplate(imgRef, imgShift, cv2.TM_CCOEFF_NORMED)
    print ("correlation :", coef[0][0])

class sa_optimize:
    def __init__(self, img1, img2, ref_pose):
        self.row, self.col = img1.shape
        self.img1 = img1
        self.img2 = img2

        init_pose = [0, 0]
        self.pose = init_pose
        
        self.old_energyZ = self.energy_init()
        self.new_energyZ = 0
        
        self.temp = 1.0
        self.min_temp = 0.001
        self.alpha = 0.9

        self.count_increase = 0
        self.count_decrease = 0
        self.count_accepted = 0

        # print ("init_energy :", self.old_energyZ)
        # print ("==========================")
   
        # plt.axis([-20, 20, -20, 20])
        while(self.temp > self.min_temp):
            self.update()
            # plt.scatter(self.pose[0], self.pose[1])
            # plt.pause(0.05)
            # print ("----------------")
        


        # while(1):
        #     commd = input("Continue? :")
        #     if commd == "y":
        #         self.update(pose)
        #     else:
        #         break
        #     print ("==========================")

        # if self.temp > self.min_temp:
        #     print ("MORE")
        # else:
        #     print ("LESS")

        # print ("Finish")
        print ("Increase :", self.count_increase, "times")
        print ("Accepted :", self.count_accepted, "times")
        print ("Decrease :", self.count_decrease, "times")
        print ("pose :", self.pose)
        correlation(img1, img2, self.pose[0], self.pose[1])
        print ("========")
        # plt.show()

        # print ("output_max :", np.max(self.state_z))
        # print ("output_min :", np.min(self.state_z))

        # sonarPlotting.subplot2(self.init_state, self.state_z, ["1", "2"])

    def energy_init(self):
        rowShift = self.pose[0]
        colShift = self.pose[1]
        trans = np.float32([[1,0,colShift],[0,1,rowShift]])
        z_ref = self.img1
        z_shift = cv2.warpAffine(self.img2, trans, (self.col,self.row))
        z_ref, z_shift = shift_and_crop(z_ref, z_shift, rowShift, colShift)
        
        if True:
            diff =  z_ref - z_shift
            diff = np.power(diff, 2)
            eng = np.sum(diff) / (z_ref.shape[0] * z_ref.shape[1])
        else:
            coef = cv2.matchTemplate(z_ref, z_shift, cv2.TM_CCOEFF_NORMED)
            eng = coef[0][0]
        return eng

    def update(self):
        move_row = self.pose[0] + random.choice([-2,-1,0,1,2])
        move_col = self.pose[1] + random.choice([-2,-1,0,1,2])
        self.move = [move_row, move_col]
        self.z_energy()

        
        # print ("old_energy :", self.old_energyZ)
        # print ("new_energy :", self.new_energyZ)
        if self.new_energyZ < self.old_energyZ:
            # print ("Decrease")
            self.count_decrease += 1
            self.pose = self.move
            self.old_energyZ = self.new_energyZ
        else:
            # print ("Increase")
            self.count_increase += 1
            prob = accepted_prob(self.new_energyZ, self.old_energyZ, self.temp)
            # print ("prob :", prob)
            if prob > random.random():
                self.count_accepted += 1
                self.pose = self.move
                self.old_energyZ = self.new_energyZ

            self.temp = self.temp * self.alpha

    def z_energy(self):
        rowShift = self.move[0]
        colShift = self.move[1]
        trans = np.float32([[1,0,colShift],[0,1,rowShift]])
        z_shift = cv2.warpAffine(self.img2, trans, (self.col,self.row))
        z_ref, z_shift = shift_and_crop(self.img1, z_shift, rowShift, colShift)

        if True:
            diff =  z_ref - z_shift
            diff = np.power(diff, 2)
            eng = np.sum(diff) / (z_ref.shape[0] * z_ref.shape[1])
        else:
            coef = cv2.matchTemplate(z_ref, z_shift, cv2.TM_CCOEFF_NORMED)
            eng = coef[0][0]
        
        self.new_energyZ = eng

if __name__ == '__main__':
    start = 3
    stop = 7
    img1 = read_multilook(start)
    img2 = read_multilook(stop)
    speed, pose = positioning(20)
    pose_Shift = position_shift(pose[start], pose[stop])

    print ("start simulated annealing...")
    # for i in range(0,10):
    #     print ("round :", i+1)
    #     sa_optimize(img1, img2, pose_Shift)

    if False:
        row, col = img1.shape
        point_img1 = []
        point_img2 = []
        for i in range(0,4):
            rand_row = random.randint(0,row)
            rand_col = random.randint(0,col)
            point_img1.append((rand_col+2,rand_row+19))
            point_img2.append((rand_col,rand_row))
        point_img1 = np.asarray(point_img1)
        point_img2 = np.asarray(point_img2)
        p = cv2.getPerspectiveTransform(point_img2.astype(np.float32), point_img1.astype(np.float32))
     

        f_stitched = warp(img1, p, output_shape=(600,800))

        M, N = img2.shape[:2]
        print (M,N)

        f_stitched[0:M, 0:N] = img2
        f_stitched2 = warp(img1, p, output_shape=(600,800))
        # plt.imshow(f_stitched); plt.axis('off')
        # plt.show()
        sonarPlotting.subplot2(f_stitched, f_stitched2, ["1", "2"])
        # cv2.imshow("warp", f_stitched)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # xy = np.array([[ 157, 32],[ 211, 37],[ 222,107],[ 147,124]])
        # xaya = np.array([[  6, 38],[ 56, 31],[ 82, 87],[ 22,118]])
        # print (xy)
        # print (xaya)
        # P = cv2.getPerspectiveTransform(xy.astype(np.float32), xaya.astype(np.float32))
        # print (P)

    if True:
        cf1 = cafar(img1)
        cf2 = cafar(img2)

        # row, col = img1.shape
        # img1[0:row, 0:col] = cf1


        out = cv2.addWeighted(img1, 0.5, cf1, 0.5, 0)

        # cv2.imshow("out", out)
        
        # col_list  = []
        # inx = int(128/2)
        # for i in range(1,7):
        #     crop = img1[0:500, i*inx:(i*inx)+1]
        #     # col_list.append(crop)
        #     # cf_list = cf1[0:500, 500:501]
        #     plt.plot(crop)
        #     plt.show()

        col_list = img1[0:500, 500:501]
        cf_list = cf1[0:500, 500:501]

        col_list = np.flip(col_list,0)
        cf_list = np.flip(cf_list,0)
        pose_cf = []

        # print (np.max(cf_list))
        # print (np.min(cf_list))
        for i in range(len(cf_list)):
            if cf_list[i] == 255:
                pose_cf.append(i)
        # print (pose_cf)
        # print (pose_cf[0])
        # print (pose_cf[len(pose_cf)-1])

        start_cf = pose_cf[0]
        stop_cf = pose_cf[len(pose_cf)-1]

        for j in pose_cf:
            if col_list[j] == np.max(col_list):
                peak = j

        # print (peak)

        shadow = col_list[stop_cf:500]
        sha_list = []
        for k in range(len(shadow)):
            if shadow[k] < 70:
                sha_list.append(k)
            else:
                break
        # print (sha_list)

        start_sha = sha_list[0] + stop_cf
        stop_sha = sha_list[len(sha_list)-1] + stop_cf

        
        # res = col_list * cf_list
        # res = res.astype(np.uint8)

        # sonarPlotting.subplot2(out, out, ["1", "2"])
        pix_range = 30.0 / 660.0
        print (start_cf, stop_cf, start_sha, stop_sha)
        print (start_cf*pix_range, stop_cf*pix_range, stop_sha*pix_range)

        auv = 2.2

        height = auv * (((stop_sha*pix_range) - (peak*pix_range))/(stop_sha*pix_range))

        print (height)

        # plt.plot(res)
        # plt.plot(col_list)
        # plt.plot(cf_list)
        # plt.plot(50 * np.ones(500))
        # plt.plot(shadow)
        # plt.show()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()