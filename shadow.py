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
import coef_shifting as coef

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
    img = cv2.imread(picPath + picName)
    # print ("success import...   " + picName)
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

def get_peak(intensity, mask):
    pose = []
    for i in range(len(mask)):
        if mask[i] == 255:
            pose.append(i)
    if len(pose) == 0:
        return 0 , 0, 0
    else:
        start_pipe = pose[0]
        end_pipe = pose[len(pose)-1]
        peak = 0
        for j in pose:
            if intensity[j] == np.max(intensity):
                peak = j
                break
        return peak, end_pipe, start_pipe

def get_shadow_edge(intensity, start_shadow, row):
    shadow = intensity[start_shadow:row]
    sha_list = []
    for k in range(len(shadow)):
        if shadow[k] < intensity[start_shadow-1]:
            sha_list.append(k)
        else:
            break
    # print (len(sha_list))
    if len(sha_list) == 0:
        return 0
    else:
        edge_shadow = sha_list[len(sha_list)-1] + start_shadow
        return edge_shadow

def calculate_height(auv, peak, edge):
    ratio = 30.0 / 660.0
    height = auv * (((edge*ratio) - (peak*ratio))/(edge*ratio))
    # print (height[0])
    return height

def draw_dot(map_img, peak, edge, inx_col, start_cf, stop_cf):
    # cv2.circle(map_img, (peak,inx_col), 3, (255,0,0), -1)
    # cv2.circle(map_img, (edge,inx_col), 3, (0,255,0), -1)
    map_img = cv2.flip(map_img, 0)
    cv2.circle(map_img, (inx_col,peak), 3, (255,0,0), -1)
    cv2.circle(map_img, (inx_col,edge), 3, (0,255,0), -1)
    cv2.circle(map_img, (inx_col,start_cf), 3, (0,0,255), -1)
    # cv2.circle(map_img, (inx_col,stop_cf), 3, (0,0,255), -1)
    map_img = cv2.flip(map_img, 0)

    return map_img

def height_with_cfar(img, step):
    row, col = img.shape
    cf = cafar(img)
    img = cv2.medianBlur(img,5)
    map_img = read_image(3) 

    inx_col = 0
    h_list = []
    position = []
    while(1):
        inx_col = inx_col + step
        if inx_col > col:
            break

        intensity = img[0:row, inx_col:inx_col+1]
        intensity = np.flip(intensity,0)
        mask_cf = cf[0:row, inx_col:inx_col+1]
        mask_cf = np.flip(mask_cf,0)

        peak, start_shadow, start_pipe = get_peak(intensity, mask_cf)
        edge = get_shadow_edge(intensity, start_shadow, row)
        auv = 2.2

        if peak != 0 and start_shadow != 0 and edge != 0:
            height = calculate_height(auv, peak, edge)
            # print (height)
            h_list.append(height)
            map_img = draw_dot(map_img, peak, edge, inx_col, start_shadow, start_pipe)
            position.append((peak,inx_col))
        
    cv2.imshow("map", map_img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return h_list, position

def reduce_error_height(h_list, position):
    ele_list = []
    pose = []
    lower = np.mean(h_list) - 0.2
    upper = np.mean(h_list) + 0.2

    for i in range(len(h_list)):
        if lower <= h_list[i] <= upper:
            ele_list.append(h_list[i])
            pose.append(position[i])
    return ele_list, pose

def height_no_cfar(img, step, inx):
    row, col = img.shape
    img = cv2.medianBlur(img,5)
    map_img = read_image(inx) 

    inx_col = 0
    h_list = []
    position = []
    auv = 2.2

    while(1):
        inx_col = inx_col + step
        if inx_col > col:
            break
        intensity = img1[0:row, inx_col:inx_col+1]
        intensity = np.flip(intensity,0)
        maxima = np.max(intensity)
        minima = np.min(intensity)
        
        pose_max = np.argmax(intensity)
        pose_min = np.argmin(intensity)

        if 1.0 < (pose_min/pose_max) < 1.4:
            if (maxima/minima) > 5.0:
                # map_img = draw_dot(map_img, pose_max, pose_min, inx_col, 0, 0)
                thres = intensity[pose_min] + 50
                for i in range(0,len(intensity) - pose_min):
                    if intensity[pose_min+i] > thres:
                        height = calculate_height(auv, pose_max, pose_min+i)
                        # print (height)
                        h_list.append(height)
                        position.append((pose_max, inx_col))
                        map_img = draw_dot(map_img, pose_max, pose_min, inx_col, pose_min+i, 0)
                        break
    return h_list, position

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

class elv_map:
    def __init__(self, height, position, row, col):
        self.map = np.zeros((row, col),np.uint8)
        self.ratio = 30.0 / 660.0
        # self.create_map(height, position)
        # self.reduce_kernel(11, 0.5)
        # self.test_create_map()
        self.create_map(height, position)
        self.map = np.flip(self.map, 0)
        # cv2.imshow("after", self.map)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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

    # def add_height(self, size):


if __name__ == '__main__':
    # i = 5
    # img1 = read_multilook(4)
    # row, col = img1.shape
    # height, pose = height_no_cfar(img1, 5, 3)
    # z_map1 = elv_map(height,pose,row, col)

    # cv2.imshow("zMap1", z_map1.map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # img1 = read_multilook(3)
    # img2 = read_multilook(7)
    # img = coef.coefShift(img1, img2, 30)
    if False:
        for i in range(3,20):
            img1 = read_multilook(i)
            row, col = img1.shape
            # height, pose = height_with_cfar(img1, 10)
            height, pose = height_no_cfar(img1, 5, i)

            z_map = elv_map(height,pose,row, col)
            cv2.imwrite("zMap_" + str(i) + ".jpg", z_map.map)

    if True:
        z_map1 = cv2.imread("D:\Pai_work\zMap3.png", 0)
        z_map2 = cv2.imread("D:\Pai_work\zMap4.png", 0)

        row, col = z_map1.shape
        num_pix = row * col
        err = np.power((z_map1 - z_map2), 2)
        err = np.sum(err)
        print (err/num_pix)

        # ref1, ref2 = shift_and_crop(z_map1, z_map2, -21, 0)
        # row, col = ref1.shape
        # num_pix = row * col
        # err = np.power((ref1 - ref2), 2)
        # err = np.sum(err)
        # print (err/num_pix)

    if False:
        row, col = img1.shape
        cf1 = cafar(img1)
        # img1 = cv2.GaussianBlur(img1,(15,15),0)
        # img1 = cv2.blur(img1,(15,15))
        img1 = cv2.medianBlur(img1,5)

        map_img = read_image(3) 
        # for i in range(1,40):
        #     inx_col = 500 + (i*5)
        inx_col = 0
        h_list = []
        position = []
        while(1):
            inx_col = inx_col + 10
            if inx_col > col:
                break

            intensity = img1[0:row, inx_col:inx_col+1]
            intensity = np.flip(intensity,0)
            mask_cf = cf1[0:row, inx_col:inx_col+1]
            mask_cf = np.flip(mask_cf,0)

            peak, start_shadow, start_pipe = get_peak(intensity, mask_cf)
            edge = get_shadow_edge(intensity, start_shadow, row)
            auv = 2.2

            if peak != 0 and start_shadow != 0 and edge != 0:
                height = calculate_height(auv, peak, edge)
                # print (height)
                h_list.append(height)
                map_img = draw_dot(map_img, peak, edge, inx_col, start_shadow, start_pipe)
                ()
                position.append((peak,inx_col))
    if False:
        ele_list = []
        pose = []
        print (np.average(h_list))
        print (np.mean(h_list))
        lower = np.mean(h_list) - 0.2
        upper = np.mean(h_list) + 0.2

        for i in range(len(h_list)):
            if lower <= h_list[i] <= upper:
                ele_list.append(h_list[i])
                pose.append(position[i])

        # h_map = elv_map(ele_list, pose, row, col)

        gauss = cv2.getGaussianKernel(ksize = 11, sigma = 4)
        # gauss = (gauss/np.max(gauss))*0.5
        gauss = gauss.reshape(1,11)
        gauss = np.dot(gauss.T,gauss)
        # print (gauss)
        print ((gauss/np.max(gauss))*0.5)

        # print (ele_list)
        # ele_list.append(1.0)
        # ele_list = np.asarray(ele_list)
        # print (ele_list.dtype)
        # test1 = img_as_ubyte(ele_list)
        # print (test1)

        # test2 = ele_list / ele_list.max() 
        # test2 = 255 * test2
        # test2 = test2.astype(np.uint8)
        # print (test2)

        # cv2.imshow("map", map_img)
        # cv2.imshow("cf1", cf1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

