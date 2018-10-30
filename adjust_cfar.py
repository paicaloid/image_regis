import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

import sonarPlotting
import FeatureMatch
import regisProcessing as rp

# picPath = "D:\Pai_work\pic_sonar"
# picPath = "D:\Pai_work\pic_sonar\jpg_file"


if __name__ == '__main__':
    start = 10
    ## construct the argument parse and parse the arguments ##
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=True, help="select image file")
    args = vars(ap.parse_args())

    if args["file"] == "jpg":
        picPath = "D:\Pai_work\pic_sonar"
        picName = "\RTheta_img_" + str(start) + ".jpg"
        img = cv2.imread(picPath + picName, 0)
    if args["file"] == "ppm":
        picPath = "D:\Pai_work\pic_sonar\ppm_file"
        picName = "\RTheta_img_" + str(start) + ".ppm"
        img = cv2.imread(picPath + picName, 0)

    img = img[0:500, 0:768]

    sizeList = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]
    # for i in range(1,10):
    #     inx = 0.05 * i
    #     sizeList.append(inx)
    # print (sizeList)

    # for inx in sizeList:
    #     res = rp.cafar(img, 59, 11, inx)
    #     kernel = np.ones((7,7),np.uint8)
    #     res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    #     saveName = "D:\Pai_work\pic_sonar\cfar_exp\cfar_ppm_pfa#" + str(inx*100) + ".jpg" 
    #     cv2.imwrite(saveName, res)
    
    res = rp.cafar(img, 59, 11, 0.25)
    kernel = np.ones((7,7),np.uint8)
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

    imgBlock = rp.BlockImage()
    imgBlock.Creat_Block(res)
    imgBlock.Adjsut_block(14)
    imgBlock.Calculate_Mean_Var()
    print (imgBlock.averageMean)

    for i in range(0,30):
        if imgBlock.meanList[i] > imgBlock.averageMean * 2:
            print (i, imgBlock.meanList[i])
    imgBlock.Show_allBlock()