import cv2
import numpy as np
import math
import random

from skimage import img_as_ubyte
from skimage.feature import register_translation
from matplotlib import pyplot as plt

import denoise_normal

class simulated_annealing:
    def __init__(self, img, temp, min_temp, alpha, gamma, beta):
        self.temp = temp
        self.min_t = min_temp
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

        self.loop = True
        self.old_eng = 0
        self.new_eng = 0

        self.ref = img.astype(int)
        self.current = img.astype(int)
        self.next = img.astype(int)

        row_pos, col_pos = np.where(img > 0)
        print (row_pos.min(), row_pos.max())
        print (col_pos.min(), col_pos.max())
        self.inx_row = np.arange(row_pos.min(), row_pos.max())
        self.inx_col = np.arange(col_pos.min(), col_pos.max())
    
    def annealing(self, nb_size, move_size):
        # print ("annealing...")
        c_decrease = 0
        c_increase = 0
        for i in range(len(self.inx_row)):
            for j in range(len(self.inx_col)):
                row = self.inx_row[i]
                col = self.inx_col[j]

                old = self.neighbor_energy(self.current[row][col], row, col, nb_size)
                
                move = np.random.randint(-move_size,move_size)
                new_center = self.current[row][col] + move
                if new_center < 0:
                    new_center = 0
                elif new_center > 255:
                    new_center = 255
                
                new = self.neighbor_energy(new_center, row, col, nb_size)

                if new <= old:
                    self.next[row][col] = new_center
                    c_decrease += 1
                else:
                    self.next[row][col] = self.current[row][col]
                    c_increase += 1
        # print (c_decrease, c_increase)  

    def neighbor_energy(self, center, row, col, nb_size):
        summ = 0
        for i in range(row-nb_size, row+nb_size+1):
            for j in range(col-nb_size, col+nb_size+1):
                if i == row and j == col:
                    pass
                else:
                    # diff = math.pow(self.ref[i][j] - center,2)
                    diff = math.pow(self.current[i][j] - center,2)
                    summ = summ + diff
        return summ

    def energy(self):
        # ! Energy Observation
        obv_eng = 0
        for i in range(len(self.inx_row)):
            for j in range(len(self.inx_col)):
                row = self.inx_row[i]
                col = self.inx_col[j]
                diff = self.ref[row][col] - self.next[row][col]
                diff = math.pow(diff, 2)
                obv_eng = obv_eng + diff
        obv_eng = obv_eng / (len(self.inx_row) * len(self.inx_col))
        # ! Energy Z_map
        z_eng = 0
        inx_row, inx_col = np.where(self.next > 0)
        for i in range(100):
            rand = np.random.randint(len(inx_row))
            row = inx_row[rand]
            col = inx_col[rand]
            z_eng = z_eng + self.neighbor_energy(self.next[row][col], row, col, 1)
        z_eng = z_eng / 100
        # print ("obv_eng : ", obv_eng, ", z_eng : ", z_eng)
        # print (obv_eng, ",", z_eng)
        sum_eng = (obv_eng * self.gamma) + (z_eng * self.beta)
        # print ("sum_eng" , sum_eng)
        return sum_eng

    def update_state(self):
        if self.new_eng < self.old_eng:
            print ("decrease...")
            self.old_eng = self.new_eng
            self.current = self.next
        else:
            diff = (-1) * (self.new_eng - self.old_eng)
            diff = diff / self.temp
            prob = math.exp(diff)
            print (prob)
            if prob > random.random():
                print ("Increase... pass")
                self.old_eng = self.new_eng
                self.current = self.next
            else:
                print ("Increase... fail")
        self.temp = self.temp - self.alpha
        
class elv_map:
    def __init__(self, row, col):
        self.map = np.zeros((row, col),np.uint8)
        self.ratio = 30.0 / 660.0
        # self.draw_map = read_image(3)
        # self.draw_map = np.flip(self.draw_map, 0)

        # self.create_map(height, position)
        # self.map = np.flip(self.map, 0)

    def create_map(self, height, position):
        for i in range(len(height)):
            pose = position[i]
            size = height[i] / self.ratio
            size = int(size)
            if size % 2 == 0:
                size = size - 1
            kernel = self.reduce_kernel(size, height[i])
            row, col = kernel.shape
            half = int((size-1)/2)
            # self.map[pose[0]-size:pose[0]+size+1+row, pose[1]-size:pose[1]+size+1+col] = kernel
            self.map[pose[0]-half:pose[0]+half+1, pose[1]-half:pose[1]+half+1] = kernel

    def extended_map(self, height, position, ref_img):
        row, col = ref_img.shape
        draw_map = np.zeros((row,col), np.uint8)
        # draw_map = read_image(3)

        avg_h = np.average(height)
        intensity_list = []
        for pose in position:
            intensity_list.append(ref_img[pose[0]][pose[1]])
            # cv2.circle(draw_map, (pose[1],pose[0]), 5, 255, -1)
            # cv2.circle(draw_map, (pose[1],pose[0]), 5, (0,0,255), -1)
        avg_intensity = int(np.average(intensity_list))

        # print (avg_h)
        # print (avg_intensity)
        # print (intensity_list)

        for i in range(row):
            for j in range(col):
                if avg_intensity + 170 < ref_img[i][j]:
                    # cv2.circle(draw_map, (j,i), 3, (0,255,0), -1)
                    cv2.circle(draw_map, (j,i), 3, 255, -1)
        # draw_map = np.flip(draw_map, 0)
        return draw_map
    
    def test_create_map(self):
        z = 0.5
        size = 11
        pose = (100,100)
        kernel = self.reduce_kernel(size, z)
        row, col = kernel.shape
        cv2.imshow("after", self.map)
        self.map[100:100+row, 100:100+col] = kernel
        cv2.imshow("after", self.map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def reduce_kernel(self, size, height):
        value = int((size-1) / 2)
        h = height
        h_min = h / 2.0
        h_step = (h - h_min) / (value - 1)
        h_list = []
        for i in range(value):
            h_list.append(h - (h_step * i))
        h_list = np.flip(h_list, 0)
        # print (h_list)
        base = np.zeros((size,size))
        for i in range(value):
            temp = size - (2 * i)
            a = np.ones((temp,temp)) * h_list[i]
            b = np.zeros((temp-2,temp-2))
            ref = np.zeros((size,size))
            row, col = b.shape
            a[1:row+1, 1:col+1] = b
            row, col = a.shape
            ref[i:row+i, i:col+i] = a
            base = base + ref
        row, col = base.shape
        base[int((row-1)/2)][int((col-1)/2)] = height
        base = img_as_ubyte(base)
        # print (base)
        return base

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

def read_regis(sec):
    image_perSec = 3.3
    picPath = "D:\Pai_work\pic_sonar\\"
    start = math.ceil(sec * image_perSec)
    stop = math.ceil((sec + 1) * image_perSec)

    picName = "RTheta_img_" + str(start) + ".jpg"
    ref = cv2.imread(picPath + picName, 0)
    row, col = ref.shape
    summ = cv2.imread(picPath + picName, 0)

    for i in range(start + 1, stop):
        picName = "RTheta_img_" + str(i) + ".jpg"
        img = cv2.imread(picPath + picName, 0)
        shift, error, diffphase = register_translation(ref, img)
        # print (shift)
        trans_matrix = np.float32([[1,0,int(shift[1])],[0,1,int(shift[0])]])
        shift_img = cv2.warpAffine(img, trans_matrix, (col,row))
        summ = cv2.addWeighted(summ, 0.5, shift_img, 0.5, 0)

    print ("success multilook... " + str(start) + " to " +str(stop-1))
    cv2.imshow("sonar_output", summ)
    summ = summ[0:500, 0:768]
    return summ

def calculate_height(auv, peak, edge):
    ratio = 30.0 / 660.0
    height = auv * (((edge*ratio) - (peak*ratio))/(edge*ratio))
    # print (height[0])
    return height

def height_no_cfar(img, step):
    row, col = img.shape
    img = cv2.medianBlur(img,5)

    inx_col = 0
    h_list = []
    position = []
    auv = 2.5

    while(1):
        inx_col = inx_col + step
        if inx_col > col:
            break
        intensity = img[0:row, inx_col:inx_col+1]
        intensity = np.flip(intensity,0)
        maxima = np.max(intensity)
        minima = np.min(intensity)
        
        pose_max = np.argmax(intensity)
        pose_min = np.argmin(intensity)
        # map_img = draw_dot(map_img, pose_max, pose_min, inx_col, pose_min, 0)
        if 1.0 < (pose_min/pose_max) < 1.4:
            if (maxima/minima) > 5.0:
            # if 5.0 < (maxima/minima) < 9.0:
                # map_img = draw_dot(map_img, pose_max, pose_min, inx_col, 0, 0)
                thres = intensity[pose_min] + 50
                for i in range(0,len(intensity) - pose_min):
                    if intensity[pose_min+i] > thres:
                        height = calculate_height(auv, pose_max, pose_min+i)
                        # print (height)
                        h_list.append(height)
                        position.append((pose_max, inx_col))
                        break
    return h_list, position

def generate_map(img):
    row, col= img.shape
    height, position = height_no_cfar(img, 5)
    z_map = elv_map(row, col)
    z_map.create_map(height, position)
    z_map.map = np.flip(z_map.map, 0)
    z_map.map = cv2.medianBlur(z_map.map,5)

    return z_map.map.astype(int)

def count_height(img):
    # for i in range(0,10):
    #     inx_bot = i * 0.1
    #     inx_bot = int(inx_bot * 255)

    #     inx_up = (i+1) * 0.1
    #     inx_up = int(inx_up * 255)

    #     row, col = np.where(inx_bot < img < inx_up)
    #     print (inx_bot," to ", inx_up, " : ", len(row))
    count = 0
    row, col = np.where(img > 0)
    for i in range(len(row)):
        if 38 < img[row[i]][col[i]] < 102:
            count += 1
    print (len(row))
    print (count)

def rmse(ref, shift):
    summ = 0
    row, col = np.where(ref > 0)
    for i in range(len(row)):
        inx_r = row[i]
        inx_c = col[i]
        err = ref[inx_r][inx_c] - shift[inx_r][inx_c]
        err = math.pow(err, 2)
        summ = summ + err
    summ = summ / len(row)
    summ = np.sqrt(summ)
    print (summ)
    # return summ

def root_mean_sqrt(ref, shift):
    ref = ref.astype(float)
    ref = ref/255.0
    # print (ref.max(), ref.min())
    shift = shift.astype(float)
    shift = shift/255.0

    diff = ref - shift
    diff = np.power(diff, 2)
    summ = np.sum(diff)
    print (summ)

# img_list = []
# for i in range(3,21):
#     img = read_regis(i)
#     row, col = img.shape
#     range_ratio = 30/660
#     des = denoise_normal.denoise_range(img, 0.0, row*range_ratio)
#     mapp = generate_map(des)

#     saveName = "multilook_sec" + str(i) + ".jpg"
#     cv2.imwrite(saveName, mapp.astype(np.uint8))

    # plt.subplot(121)
    # plt.imshow(img)
    # plt.subplot(122)
    # plt.imshow(mapp)
    # plt.show()
start = 11
    
img1 = read_regis(start)
cv2.imshow("cropped", img1)
row, col = img1.shape
range_ratio = 30/660
des1 = denoise_normal.denoise_range(img1, 0.0, row*range_ratio)
cv2.imshow("denoise", des1.astype(np.uint8))
cv2.waitKey(0)
map1 =generate_map(des1)

img2 = read_regis(start+1)
des2 = denoise_normal.denoise_range(img2, 0.0, row*range_ratio)
map2 =generate_map(des2)

img3 = read_regis(start+2)
des3 = denoise_normal.denoise_range(img3, 0.0, row*range_ratio)
map3 =generate_map(des3)

img4 = read_regis(start+3)
des4 = denoise_normal.denoise_range(img4, 0.0, row*range_ratio)
map4 =generate_map(des4)

shift, error, diffphase = register_translation(des1, des2)
# print (shift)
trans_matrix = np.float32([[1,0,int(shift[1])],[0,1,int(shift[0])]])
map2 = cv2.warpAffine(map2.astype(np.uint8), trans_matrix, (col,row))
# rmse(map1, map2)
root_mean_sqrt(map1, map2)

shift, error, diffphase = register_translation(des1, des3)
# print (shift)
trans_matrix = np.float32([[1,0,int(shift[1])],[0,1,int(shift[0])]])
map3 = cv2.warpAffine(map3.astype(np.uint8), trans_matrix, (col,row))
# rmse(map1, map3)
root_mean_sqrt(map1, map3)

shift, error, diffphase = register_translation(des1, des4)
# print (shift)
trans_matrix = np.float32([[1,0,int(shift[1])],[0,1,int(shift[0])]])
map4 = cv2.warpAffine(map4.astype(np.uint8), trans_matrix, (col,row))
# rmse(map1, map4)
root_mean_sqrt(map1, map4)

add1 = 0.25 * map1.astype(float)
add2 = 0.25 * map2.astype(float)
add3 = 0.25 * map3.astype(float)
add4 = 0.25 * map4.astype(float)
merge = add1 + add2 + add3 + add4
merge = merge.astype(np.uint8)

cv2.imshow("height_map",map1.astype(np.uint8))
cv2.imshow("registration", merge.astype(np.uint8))
cv2.waitKey(0)
print (merge.max(), merge.min())
plt.imshow(merge)
plt.show()


# plt.subplot(221)
# plt.imshow(map1)
# plt.subplot(222)
# plt.imshow(map2)
# plt.subplot(223)
# plt.imshow(map3)
# plt.subplot(224)
# plt.imshow(map4)
# plt.show()

# ! Simulated annealing
opt = simulated_annealing(merge, 1000, 10, 10, 0.05, 0.95)
while(1):
    opt.annealing(1, 15)
    eng = opt.energy()
    if opt.loop:
        opt.loop = False
        opt.old_eng = eng
        opt.current = opt.next
    else:
        opt.new_eng = eng
        opt.update_state()
    
    if opt.temp < opt.min_t:
        break

plt.subplot(121)
plt.imshow(merge)
plt.subplot(122)
plt.imshow(opt.next)
plt.show()

cv2.imwrite("before_5.jpg", merge.astype(np.uint8))
cv2.imwrite("after_5.jpg", opt.next.astype(np.uint8))

# ! Check accuracy 
# row, col = np.where(merge > 0)
# summ = 0
# for i in range(len(row)):
#     summ = summ + merge[row[i]][col[i]]
# print (summ/len(row))

# row, col = np.where(opt.next > 0)
# summ = 0
# for i in range(len(row)):
#     summ = summ + opt.next[row[i]][col[i]]
# print (summ/len(row))

# ! Count height pixel
# count_height(merge)
# print ("+++++++++++++")
# count_height(opt.next)

# ! Check correlation
# coef = cv2.matchTemplate(merge.astype(np.uint8), merge.astype(np.uint8), cv2.TM_CCOEFF_NORMED)
# print (coef[0][0])
# coef = cv2.matchTemplate(merge.astype(np.uint8), opt.next.astype(np.uint8), cv2.TM_CCOEFF_NORMED)
# print (coef[0][0])