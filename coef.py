import cv2
import numpy as np
import csv

import sonarGeometry
import regisProcessing as rp
from matplotlib import pyplot as plt

def create_coefImg(img2):
    # Load two images
    img1 = np.zeros((800,800), dtype=np.uint8)

    # I want to put logo on top-left corner, So I create a ROI
    rows,cols = img2.shape
    roi = img1[0:rows, 0:cols ]

    # Now create a mask of logo and create its inverse mask also
    # img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[150:rows+150, 16:cols+16 ] = dst

    return img1

def shift_coefImg(img2, Rshift, Cshift):
    # Load two images
    img1 = np.zeros((800,800), dtype=np.uint8)

    # I want to put logo on top-left corner, So I create a ROI
    rows,cols = img2.shape
    roi = img1[0:rows, 0:cols ]

    # Now create a mask of logo and create its inverse mask also
    # img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[150+Rshift:rows+150+Rshift, 16+Cshift:cols+16+Cshift ] = dst

    return img1

def coefShift(img1, img2, num):
    colRange = np.arange((-1)*num, num+1, 1)
    rowRange = colRange * (-1)
    rowList = []
    coefList = []
    for rowInx in rowRange:
        colList = []
        for colInx in colRange:
            trans_matrix = np.float32([[1,0,colInx],[0,1,rowInx]])
            imgShift = cv2.warpAffine(img2, trans_matrix, (768,500))

            # ! Quadrant 1
            if rowInx > 0 and colInx > 0:
                imgRef = img1[rowInx:500, colInx:768]
                imgShift = imgShift[rowInx:500, colInx:768]
            # ! Quadrant 2
            elif rowInx > 0 and colInx < 0:
                imgRef = img1[rowInx:500, 0:768+colInx]
                imgShift = imgShift[rowInx:500, 0:768+colInx]
            # ! Quadrant 3
            elif rowInx < 0 and colInx < 0:
                imgRef = img1[0:500+rowInx, 0:768+colInx]
                imgShift = imgShift[0:500+rowInx, 0:768+colInx]
            # ! Quadrant 4
            elif rowInx < 0 and colInx > 0:
                imgRef = img1[0:500+rowInx, colInx:768]
                imgShift = imgShift[0:500+rowInx, colInx:768]
            
            coef = cv2.matchTemplate(imgRef, imgShift, cv2.TM_CCOEFF_NORMED)
            colList.append(coef[0][0])
        coefList.append(max(colList))
        rowList.append((rowInx, max(enumerate(colList), key=(lambda x: x[1]))))
        # print (rowInx, max(enumerate(colList), key=(lambda x: x[1])))
    # print (coefList)
    # print (rowList)
    # print (rowList[np.argmax(coefList)])
    posShift = rowList[np.argmax(coefList)]
    shiftRow = posShift[0]
    shiftCol = posShift[1][0] - num
    # print (shiftRow, shiftCol)
    return shiftRow, shiftCol

if __name__ == '__main__':
    # img1 = rp.AverageMultiLook(10,1)
    # img2 = rp.AverageMultiLook(20,1)

    # out = create_coefImg(img1)

    # cv2.imshow('out',out)
    # cv2.waitKey(0)

    # res = shift_coefImg(img2, -14, -1)

    # imx = cv2.addWeighted(out, 0.5, res, 0.5, 0)

    # cv2.imshow('res',imx)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ### Test colBlock ###
    if False:
        img1 = rp.AverageMultiLook(10,1)
        imgBlock_1 = rp.BlockImage()
        imgBlock_1.Creat_colBlock(img1)
        imgBlock_1.Adjust_colBlock(14)

        img2 = rp.AverageMultiLook(15,1)
        imgBlock_2 = rp.BlockImage()
        imgBlock_2.Creat_colBlock(img2)
        imgBlock_2.Adjust_colBlock(14)

        xxx, yyy = rp.fourierColBlock(imgBlock_1, imgBlock_2)

        print (xxx)
        print (yyy)

        # print (imgBlock_1.blockImg[0].shape)

        # imgBlock_1.blockImg[0] = imgBlock_1.blockImg[0][0:500, 14:128]

        # cv2.imshow("img1", imgBlock_1.blockImg[0])
        # cv2.imshow("img2", imgBlock_1.blockImg[1])
        # cv2.imshow("img3", imgBlock_1.blockImg[2])
        # cv2.imshow("img4", imgBlock_1.blockImg[3])
        # cv2.imshow("img5", imgBlock_1.blockImg[4])
        # cv2.imshow("img6", imgBlock_1.blockImg[5])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # imgBlock_1.Adjust_colBlock(14)
        
        # cv2.imshow("img33", imgBlock_1.blockImg[2])
        # cv2.imshow("img44", imgBlock_1.blockImg[3])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    ### Test loop find shift ###
    if False:
        inx = np.arange(11, 21, 1)
        for num in inx:
            img1 = rp.AverageMultiLook(num-1, 1)
            img2 = rp.AverageMultiLook(num, 1)
            coefShift(img1, img2, 10)
            print ("===============")

    if False:
        img1 = rp.AverageMultiLook(10,1)
        inx = np.arange(11, 21, 1)
        rowShift = [-1,-2,-4,-5,-6,-7,-9,-11,-12,-14]
        colShift = [0,0,0,-1,-2,-3,-3,-2,-1,-1]

        # rowShift = rowShift * 5
        # colShift = colShift * 5
        # print (rowShift)
        # print (colShift)

        for i in range(0,10):
            img2 = rp.AverageMultiLook(i+11, 1)
            trans_matrix = np.float32([[1,0,colShift[i]*5],[0,1,rowShift[i]*5]])
            img2 = cv2.warpAffine(img2, trans_matrix, (768,500))
            img1 = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
            saveName = "imgAdd_" + str(i) + ".jpg"
            cv2.imwrite(saveName, img1)

    ### Test add image ###
    if False:
        img3 = rp.AverageMultiLook(12,1)
        img4 = rp.AverageMultiLook(13,1)
        img5 = rp.AverageMultiLook(14,1)
        img6 = rp.AverageMultiLook(15,1)

        trans_matrix = np.float32([[1,0,0],[0,1,-1]])
        img2 = cv2.warpAffine(img2, trans_matrix, (768,500))
        trans_matrix = np.float32([[1,0,0],[0,1,-2]])
        img3 = cv2.warpAffine(img3, trans_matrix, (768,500))
        trans_matrix = np.float32([[1,0,0],[0,1,-4]])
        img4 = cv2.warpAffine(img4, trans_matrix, (768,500))
        trans_matrix = np.float32([[1,0,-1],[0,1,-5]])
        img5 = cv2.warpAffine(img5, trans_matrix, (768,500))
        trans_matrix = np.float32([[1,0,-2],[0,1,-6]])
        img6 = cv2.warpAffine(img6, trans_matrix, (768,500))

        cv2.imshow("0", img1)
        img1 = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
        cv2.imshow("1", img1)
        img1 = cv2.addWeighted(img1, 0.5, img3, 0.5, 0)
        cv2.imshow("2", img1)
        img1 = cv2.addWeighted(img1, 0.5, img4, 0.5, 0)
        cv2.imshow("3", img1)
        img1 = cv2.addWeighted(img1, 0.5, img5, 0.5, 0)
        cv2.imshow("4", img1)
        cv2.waitKey()
        cv2.destroyAllWindows()

    ### Test specific ###
    if False:
        posRow = -6
        posCol = -2
        trans_matrix = np.float32([[1,0,posCol],[0,1,posRow]])
        imgShift = cv2.warpAffine(img2, trans_matrix, (768,500))

        imgRef = img1[0:500+posRow, 0:768+posCol]
        imgShift = imgShift[0:500+posRow, 0:768+posCol]

        coef = cv2.matchTemplate(imgRef, imgShift, cv2.TM_CCOEFF_NORMED)
        print (coef[0][0])

        cv2.imshow("imgRef", imgRef)
        cv2.imshow("imgShift",imgShift)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if False:
        file = open('testfile.txt','w') 
        with open('10shift15.csv', mode='w') as employee_file:
            employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
            print ("==========================")
            
            num = 20
            colRange = np.arange((-1)*num, num+1, 1)
            rowRange = colRange * (-1)
            rowList = []
            print (colRange)
            print (rowRange)
            print ("==========================")
            for rowInx in rowRange:
                colList = []
                for colInx in colRange:
                    trans_matrix = np.float32([[1,0,colInx],[0,1,rowInx]])
                    imgShift = cv2.warpAffine(img2, trans_matrix, (768,500))

                    # ! Quadrant 1
                    if rowInx > 0 and colInx > 0:
                        imgRef = img1[rowInx:500, colInx:768]
                        imgShift = imgShift[rowInx:500, colInx:768]
                    # ! Quadrant 2
                    elif rowInx > 0 and colInx < 0:
                        imgRef = img1[rowInx:500, 0:768+colInx]
                        imgShift = imgShift[rowInx:500, 0:768+colInx]
                    # ! Quadrant 3
                    elif rowInx < 0 and colInx < 0:
                        imgRef = img1[0:500+rowInx, 0:768+colInx]
                        imgShift = imgShift[0:500+rowInx, 0:768+colInx]
                    # ! Quadrant 4
                    elif rowInx < 0 and colInx > 0:
                        imgRef = img1[0:500+rowInx, colInx:768]
                        imgShift = imgShift[0:500+rowInx, colInx:768]
                    
                    coef = cv2.matchTemplate(imgRef, imgShift, cv2.TM_CCOEFF_NORMED)
                    # round(14.22222223, 2)
                    # colList.append(round(coef[0][0], 2))
                    colList.append(coef[0][0])
                # print (rowInx, max(enumerate(colList), key=(lambda x: x[1])))
                # print (colList)
                employee_writer.writerow(colList)
                file.write(colList)
                # print ("++++++")
                # rowList.append(colList)
        employee_file.close()
    
    ### Test loopChain shift ###
    if False:
        inx = np.arange(11, 80, 1)
        firstCheck = True
        sumRow = 0
        sumCol = 0
        for num in inx:
            img1 = rp.AverageMultiLook(num-1, 1)
            img2 = rp.AverageMultiLook(num, 1)
            rowShift, colShift = coefShift(img1, img2, 5)
            print ("===============")
            
            if firstCheck:
                firstCheck = False
                sumRow = sumRow + rowShift
                sumCol = sumCol + colShift
                if False:
                    trans_matrix = np.float32([[1,0,sumCol],[0,1,sumRow]])
                    img2 = cv2.warpAffine(img2, trans_matrix, (768,500))
                    out = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
                if True:
                    img1 = create_coefImg(img1)
                    img2 = shift_coefImg(img2, sumRow, sumCol)
                    out = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

            else:
                sumRow = sumRow + rowShift
                sumCol = sumCol + colShift
                if False:
                    trans_matrix = np.float32([[1,0,sumCol],[0,1,sumRow]])
                    img2 = cv2.warpAffine(img2, trans_matrix, (768,500))
                    out = cv2.addWeighted(out, 0.5, img2, 0.5, 0)
                if True:
                    img2 = shift_coefImg(img2, sumRow, sumCol)
                    out = cv2.addWeighted(out, 0.5, img2, 0.5, 0)
            print (sumRow, sumCol)
            picPath = "D:\Pai_work\pic_sonar\\add"
            saveName = "\coefAdd_" + str(num) + ".jpg"
            print (saveName)
            cv2.imwrite(picPath + saveName, out)
    
    ### Test Big Picture ###
    if False:
        # Load two images
        # img1 = cv2.imread('messi5.jpg')
        # img2 = cv2.imread('opencv_logo.png')
        img1 = np.zeros((800,800), dtype=np.uint8)
        # img1 = img1.astype(np.uint8)
        img3 = np.zeros((800,800), dtype=np.uint8)
        # img3 = img3.astype(np.uint8)
        img2 = rp.AverageMultiLook(10,1)

        # I want to put logo on top-left corner, So I create a ROI
        rows,cols = img2.shape
        roi = img1[0:rows, 0:cols ]

        # Now create a mask of logo and create its inverse mask also
        # img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

        # Put logo in ROI and modify the main image

        
        img1_bg = img1_bg.astype(np.uint8)
        print (img1_bg.dtype)
        print (img2_fg.dtype)
        # img2_fg = img2_fg.astype(np.uint8)
        dst = cv2.add(img1_bg,img2_fg)
        img1[150:rows+150, 16:cols+16 ] = dst
        img3[250:rows+250, 16:cols+16 ] = dst

        out = cv2.addWeighted(img1, 0.5, img3, 0.5, 0)

        cv2.imshow('res',out)
        # cv2.imshow('img1_bg',img1_bg)
        # cv2.imshow('img2_fg',img2_fg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ### Test Mosaicing ###
    if False:
        img1 = cv2.imread("D:\Pai_work\pic_sonar\\add\coefAdd_11.jpg",0)
        img2 = cv2.imread("D:\Pai_work\pic_sonar\\add\coefAdd_79.jpg",0)

        dst = cv2.add(img1,img2)

        print (img1.dtype)
        print (img2.dtype)
        print (dst.dtype)

        cv2.imshow("dst", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ### Test Stereo ###
    if True:
        imgL = cv2.imread("D:\Pai_work\pic_sonar\Rotate_img_10.jpg",0)
        imgR = cv2.imread("D:\Pai_work\pic_sonar\Rotate_img_90.jpg",0)

        imgL = imgL[0:768, 0:500]
        imgR = imgR[0:768, 0:500]

        # cv2.imshow("dst", imgL)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # stereo = cv2.createStereoBM(numDisparities=16, blockSize=15)
        # stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET,ndisparities=16, SADWindowSize=15)
        # stereo = cv2.StereoBM(ndisparities=16, SADWindowSize=15)
        stereo = cv2.StereoBM_create(numDisparities = 128, blockSize = 15)
        disparity = stereo.compute(imgL,imgR)
        plt.imshow(disparity,'gray')
        plt.show()