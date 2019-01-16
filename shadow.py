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
    # print (height)

def draw_dot(map_img, peak, edge, inx_col, start_cf, stop_cf):
    # cv2.circle(map_img, (peak,inx_col), 3, (255,0,0), -1)
    # cv2.circle(map_img, (edge,inx_col), 3, (0,255,0), -1)
    map_img = cv2.flip(map_img, 0)
    cv2.circle(map_img, (inx_col,peak), 11, (255,0,0), -1)
    cv2.circle(map_img, (inx_col,edge), 3, (0,255,0), -1)
    cv2.circle(map_img, (inx_col,start_cf), 3, (0,0,255), -1)
    cv2.circle(map_img, (inx_col,stop_cf), 3, (0,0,255), -1)
    map_img = cv2.flip(map_img, 0)

    return map_img

img1 = read_multilook(3)
row, col = img1.shape
cf1 = cafar(img1)
img1 = cv2.GaussianBlur(img1,(15,15),0)

map_img = read_image(3) 
# for i in range(1,40):
#     inx_col = 500 + (i*5)
inx_col = 0
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
        calculate_height(auv, peak, edge)
        map_img = draw_dot(map_img, peak, edge, inx_col, start_shadow, start_pipe)
    # else:
    #     print ("No pipeline")

cv2.imshow("map", map_img)
cv2.imshow("cf1", cf1)
cv2.waitKey(0)
cv2.destroyAllWindows()

