import cv2
import numpy as np
import csv

import sonarGeometry
import regisProcessing as rp
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img1 = rp.AverageMultiLook(10,1)
    img2 = rp.AverageMultiLook(15,1)
    # trans_matrix = np.float32([[1,0,0],[0,1,-16]])
    # out = cv2.warpAffine(img2, trans_matrix, (768,500))

    ### Test R ###
    # for inx in range(-20,21):
    #     trans_matrix = np.float32([[1,0,0],[0,1,inx]])
    #     imgShift = cv2.warpAffine(img2, trans_matrix, (768,500))
    #     if inx < 0:
    #         imgRef = img1[0:500+inx, 0:768]
    #         imgShift = imgShift[0:500+inx, 0:768]
    #     else:
    #         imgRef = img1[inx:500, 0:768]
    #         imgShift = imgShift[inx:500, 0:768]

    #     coef = cv2.matchTemplate(imgRef, imgShift, cv2.TM_CCOEFF_NORMED)
    #     print (inx, coef[0][0])

    ### Test specific ###
    if True:
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