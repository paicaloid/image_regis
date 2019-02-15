import numpy as np
import cv2
import sonarPlotting
from matplotlib import pyplot as plt
import math
from skimage.feature import register_translation

def read_image(sec):
    image_perSec = 3.3
    picPath = "D:\Pai_work\pic_sonar\\"
    picName = "RTheta_img_" + str(math.ceil(sec*image_perSec)) + ".jpg"
    img = cv2.imread(picPath + picName)
    # print ("success import...   " + picName)
    img = img[0:500, 0:768]
    return img

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
        intensity = img[0:row, inx_col:inx_col+1]
        intensity = np.flip(intensity,0)
        maxima = np.max(intensity)
        minima = np.min(intensity)
        
        pose_max = np.argmax(intensity)
        pose_min = np.argmin(intensity)
        # # map_img = draw_dot(map_img, pose_max, pose_min, inx_col, pose_min, 0)
        # if 1.0 < (pose_min/pose_max) < 1.4:
        #     if (maxima/minima) > 5.0:
        #     # if 5.0 < (maxima/minima) < 9.0:
        #         # map_img = draw_dot(map_img, pose_max, pose_min, inx_col, 0, 0)
        #         thres = intensity[pose_min] + 50
        #         for i in range(0,len(intensity) - pose_min):
        #             if intensity[pose_min+i] > thres:
        #                 height = calculate_height(auv, pose_max, pose_min+i)
        #                 # print (height)
        #                 h_list.append(height)
        #                 position.append((pose_max, inx_col))
        #                 map_img = draw_dot(map_img, pose_max, pose_min, inx_col, pose_min+i, 0)
        #                 break
        map_img = draw_dot(map_img, pose_max, 0, inx_col, 0, 0)
        position.append((pose_max, inx_col))
    cv2.imshow("map", map_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return position

def rmse(ref, shift):
    ref = ref.astype(int)
    shift = shift.astype(int)
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
    return summ

bf1 = cv2.imread("D:\Pai_work\\result\\dataset\\before_1.jpg", 0)
bf2 = cv2.imread("D:\Pai_work\\result\\dataset\\before_2.jpg", 0)
bf3 = cv2.imread("D:\Pai_work\\result\\dataset\\before_3.jpg", 0)
bf4 = cv2.imread("D:\Pai_work\\result\\dataset\\before_4.jpg", 0)
bf5 = cv2.imread("D:\Pai_work\\result\\dataset\\before_5.jpg", 0)

af1 = cv2.imread("D:\Pai_work\\result\\dataset\\after_1.jpg", 0)
af2 = cv2.imread("D:\Pai_work\\result\\dataset\\after_2.jpg", 0)
af3 = cv2.imread("D:\Pai_work\\result\\dataset\\after_3.jpg", 0)
af4 = cv2.imread("D:\Pai_work\\result\\dataset\\after_4.jpg", 0)
af5 = cv2.imread("D:\Pai_work\\result\\dataset\\after_5.jpg", 0)

# img_list = [bf2, bf3, bf4, bf5]
# row, col = bf1.shape
# for img in img_list:
#     shift, error, diffphase = register_translation(bf1, img)
#     trans_matrix = np.float32([[1,0,int(shift[1])],[0,1,int(shift[0])]])
#     img = cv2.warpAffine(img.astype(np.uint8), trans_matrix, (col,row))
#     rmse(bf1, img)

img_list = [af2, af3, af4, af5]
row, col = af1.shape
for img in img_list:
    shift, error, diffphase = register_translation(bf1, img)
    trans_matrix = np.float32([[1,0,int(shift[1])],[0,1,int(shift[0])]])
    img = cv2.warpAffine(img.astype(np.uint8), trans_matrix, (col,row))
    rmse(af1, img)

# rmse(bf1, bf2)
# rmse(bf1, bf3)
# rmse(bf1, bf3)
# rmse(bf1, bf5)

rmse(af1, af2)
rmse(af1, af3)
rmse(af1, af3)
rmse(af1, af5)

# plt.subplot(121)
# plt.imshow(bf)
# plt.subplot(122)
# plt.imshow(af)
# plt.show()

# row, col = np.where(bf > 0)
# print (len(row))

# row, col = np.where(af > 0)
# print (len(row))
