import cv2
import numpy as np
import csv

import sonarGeometry
import regisProcessing as rp
from matplotlib import pyplot as plt

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
    print (rowList[np.argmax(coefList)])
    posShift = rowList[np.argmax(coefList)]
    shiftRow = posShift[0]
    shiftCol = posShift[1][0] - num
    print (shiftRow, shiftCol)

if __name__ == '__main__':
    # img1 = rp.AverageMultiLook(10,1)
    # img2 = rp.AverageMultiLook(11,1)

    # coefShift(img1, img2, 10)

    ### Test loop find shift ###
    if True:
        inx = np.arange(11, 21, 1)
        for num in inx:
            img1 = rp.AverageMultiLook(num-1, 1)
            img2 = rp.AverageMultiLook(num, 1)
            coefShift(img1, img2, 10)
            print ("===============")

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