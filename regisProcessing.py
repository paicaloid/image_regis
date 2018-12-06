import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import FeatureMatch

picPath = "D:\Pai_work\pic_sonar"
# picPath = "D:\Pai_work\pic_sonar\jpg_file"
# picPath = "D:\Pai_work\pic_sonar\ppm_file"

## Cell-Averaging Constant False Alarm Rate (CA-CFAR)
def cafar(img_gray, box_size, guard_size, pfa):
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
    return out

def AverageMultiLook(start, stop):
    picName = "\RTheta_img_" + str(start) + ".jpg"
    ref = cv2.imread(picPath + picName, 0)
    for i in range(1, stop):
        picName = "\RTheta_img_" + str(start+i) + ".jpg"
        img = cv2.imread(picPath + picName, 0)
        ref = cv2.addWeighted(ref, 0.5, img, 0.5, 0)
    ref = ref[0:500, 0:768]
    return ref

def ColorMultiLook(start, stop):
    picName = "\RTheta_img_" + str(start) + ".jpg"
    ref = cv2.imread(picPath + picName)
    for i in range(1, stop):
        picName = "\RTheta_img_" + str(start+i) + ".jpg"
        img = cv2.imread(picPath + picName)
        ref = cv2.addWeighted(ref, 0.5, img, 0.5, 0)
    ref = ref[0:500, 0:768]
    return ref

def multiplyImage(img1, img2):
    # ! Change dtype to avoid overflow (unit8 -> int32)
    img1 = img1.astype(np.int32)
    img2 = img2.astype(np.int32)
    out = cv2.multiply(img1, img2)
    out = out/255.0
    out = out.astype(np.uint8)
    return out

def findPhaseshift(img1, img2):
    fft_1 = np.fft.fft2(img1)
    fft_1 = np.fft.fftshift(fft_1)
    fft_2 = np.fft.fft2(img2)
    fft_2 = np.fft.fftshift(fft_2)

    A = fft_1 * np.conj(fft_2)
    B = np.abs(A)
    phaseShift = A/B
    phaseShift = np.fft.fftshift(phaseShift)
    phaseShift = np.fft.ifft2(phaseShift)
    phaseShift = np.real(phaseShift)
    phaseShift = np.abs(phaseShift)
    return phaseShift

def findShiftPosition(shifting):
    positionMax = np.amax(shifting)
    # print (positionMax)
    height, width = shifting.shape
    shift = np.unravel_index(np.argmax(shifting, axis=None), shifting.shape)
    # print (np.unravel_index(np.argmax(shifting, axis=None), shifting.shape))
    return shift

def fourierBlockImg(bImg1, bImg2):
    pos = []
    fix_pos = []
    row, col = bImg1.blockImg[0].shape
    print (row, col)
    for i in range(0,30):
        phaseShift = findPhaseshift(bImg1.blockImg[i], bImg2.blockImg[i])
        shift = findShiftPosition(phaseShift)
        fix_shift = findShiftPosition(phaseShift)
        shift = list(shift)
        fix_shift = list(fix_shift)
        pos.append(shift)

        if fix_shift[0] > row/2:
            fix_shift[0] = int(row/2 - fix_shift[0])
        if fix_shift[1] > col/2:
            fix_shift[1] = int(col/2 - fix_shift[1])
        fix_pos.append(fix_shift)

    return pos, fix_pos

def fourierColBlock(bImg1, bImg2):
    pos = []
    fix_pos = []
    row, col = bImg1.blockImg[0].shape
    print (row, col)
    for i in range(0,6):
        phaseShift = findPhaseshift(bImg1.blockImg[i], bImg2.blockImg[i])
        shift = findShiftPosition(phaseShift)
        fix_shift = findShiftPosition(phaseShift)
        shift = list(shift)
        fix_shift = list(fix_shift)
        pos.append(shift)

        if fix_shift[0] > row/2:
            fix_shift[0] = int(row/2 - fix_shift[0])
        if fix_shift[1] > col/2:
            fix_shift[1] = int(col/2 - fix_shift[1])
        fix_pos.append(fix_shift)
    return pos, fix_pos

def matchingPair(bImg1, bImg2):
    ref_Position = []
    shift_Position = []
    for i in range(0,30):
        saveName = picPath + "\exp\match#" + str(i) + ".png"
        # print (saveName)
        refPos, shiftPos = FeatureMatch.matchPosition_BF(bImg1.blockImg[i], bImg2.blockImg[i], saveName)
        if len(refPos) != 0:
            for m in refPos:
                ref_Position.append(m)
        if len(shiftPos) != 0:
            for n in shiftPos:
                shift_Position.append(n)
    
    input_row = []
    input_col = []
    for i in ref_Position:
        xx, yy = i
        input_row.append(xx)
        input_col.append(yy)

    output_row = []
    output_col = []
    for i in shift_Position:
        xx, yy = i
        output_row.append(xx)
        output_col.append(yy)

    return input_row, input_col, output_row, output_col

def matchingSpecific(bImg1, bImg2, inxList):
    ref_Position = []
    shift_Position = []
    for i in inxList:
        saveName = picPath + "\exp\Spec_match#" + str(i) + ".png"
        refPos, shiftPos = FeatureMatch.matchPosition_BF(bImg1.blockImg[i], bImg2.blockImg[i], saveName)
        if len(refPos) != 0:
            for m in refPos:
                ref_Position.append(m)
        if len(shiftPos) != 0:
            for n in shiftPos:
                shift_Position.append(n)
    
    input_row = []
    input_col = []
    for i in ref_Position:
        xx, yy = i
        input_row.append(xx)
        input_col.append(yy)

    output_row = []
    output_col = []
    for i in shift_Position:
        xx, yy = i
        output_row.append(xx)
        output_col.append(yy)

    return input_row, input_col, output_row, output_col

def linear_Approx(x_in, y_in, x_out, y_out):
    ## rewrite the line equation as x = Ap
    ## where A = [[1 x y]] 
    ## and p = [a0, a1, a2]
    vectorA = np.vstack([np.ones(len(x_in)), x_in, y_in]).T
    listA = np.linalg.lstsq(vectorA, x_out)[0]

    ## rewrite the line equation as y = Aq
    ## where B = [[1 x y x^2 y^2 xy]] 
    ## and q = [b0, b1, b2, b3, b4, b5]
    listB = np.linalg.lstsq(vectorA, y_out)[0]

    # ! Check error
    error_a = 0
    for i in range(0,len(x_in)):
        result = listA[0] + (listA[1]*x_in[i]) + (listA[2]*y_in[i])
        error_a = error_a + np.abs(x_out[i] - result)
        # print ("Error " + str(i) + " : " + str(np.abs(x_out[i] - result)))
    print (error_a/30.0)

    error_b = 0
    for i in range(0,len(x_in)):
        result = listB[0] + (listB[1]*x_in[i]) + (listB[2]*y_in[i])
        error_b = error_a + np.abs(y_out[i] - result)
        # print ("Error " + str(i) + " : " + str(np.abs(y_out[i] - result)))
    print (error_b/30.0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

    print (listA)
    print (listB)
    return listA, listB

def polynomial_Approx(x_in, y_in, x_out, y_out):
    xPow_in = np.power(x_in, 2)
    yPow_in = np.power(y_in, 2)
    xy_in = np.asarray(x_in) * np.asarray(y_in)
    xy_in = xy_in.tolist()

    ## rewrite the line equation as x = Ap
    ## where A = [[1 x y x^2 y^2 xy]] 
    ## and p = [a0, a1, a2, a3, a4, a5]
    vectorA = np.vstack([np.ones(len(x_in)), x_in, y_in, xPow_in, yPow_in, xy_in]).T
    listA = np.linalg.lstsq(vectorA, x_out)[0]
    
    ## rewrite the line equation as y = Aq
    ## where B = [[1 x y x^2 y^2 xy]] 
    ## and q = [b0, b1, b2, b3, b4, b5]
    listB = np.linalg.lstsq(vectorA, y_out)[0]

    # print (type(listA[0]))
    # print (listA[0])

    # ! Check error
    error_a = 0
    for i in range(0,len(x_in)):
        result = listA[0] + (listA[1]*x_in[i]) + (listA[2]*y_in[i]) + (listA[3]*xPow_in[i]) + (listA[4]*yPow_in[i]) + (listA[5]*xy_in[i])
        error_a = error_a + np.abs(x_out[i] - result)
        # print ("Error " + str(i) + " : " + str(np.abs(x_out[i] - result)))
    # print (error_a/30.0)

    error_b = 0
    for i in range(0,len(x_in)):
        result = listB[0] + (listB[1]*x_in[i]) + (listB[2]*y_in[i]) + (listB[3]*xPow_in[i]) + (listB[4]*yPow_in[i]) + (listB[5]*xy_in[i])
        error_b = error_a + np.abs(y_out[i] - result)
        # print ("Error " + str(i) + " : " + str(np.abs(y_out[i] - result)))
    # print (error_b/30.0)

    # print (listA)
    # print (listB)
    return listA, listB

def linearRemap(img, listA, listB):
    y, x = np.mgrid[:img.shape[0],:img.shape[1]]
    new_x = listA[0] + (listA[1]*x) + (listA[2]*y)
    new_y = listB[0] + (listB[1]*x) + (listB[2]*y)

    res = cv2.remap(img,new_x.astype('float32'),new_y.astype('float32'),cv2.INTER_LINEAR)

    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def polynomialRemap(img, listA, listB):
    y, x = np.mgrid[:img.shape[0],:img.shape[1]]
    new_x = listA[0] + (listA[1]*x) + (listA[2]*y) + (listA[3]*np.power(x,2)) + (listA[4]*np.power(y,2)) + (listA[5]*x*y)
    new_y = listB[0] + (listB[1]*x) + (listB[2]*y) + (listB[3]*np.power(x,2)) + (listB[4]*np.power(y,2)) + (listB[5]*x*y)

    res = cv2.remap(img,new_x.astype('float32'),new_y.astype('float32'),cv2.INTER_LINEAR)

    cv2.imshow("Img", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class BlockImage:
    def __init__(self):
        self.blockImg = []
        self.cfarImg = []
        self.meanList = []
        self.varList = []
        self.averageMean = 0
        self.averageVar = 0

    def Creat_Block(self, img):
        for i in range(0,5):
            for j in range(0,6):
                block_img = img[i*100:(i+1)*100, j*128:(j+1)*128]
                self.blockImg.append(block_img)
                self.cfarImg.append(block_img)

    def Creat_colBlock(self, img):
        for i in range(0,6):
            block_img = img[0:500, i*128:(i+1)*128]
            self.blockImg.append(block_img)
            self.cfarImg.append(block_img)

    def Adjsut_block(self, shiftValue):
        for i in range(0,30):
            if np.mod(i,6) < 3:
                self.blockImg[i] = self.blockImg[i][0:100, 0 + shiftValue:128]
                self.cfarImg[i] = self.cfarImg[i][0:100, 0 + shiftValue:128]
            else:
                self.blockImg[i] = self.blockImg[i][0:100, 0:128 - shiftValue]
                self.cfarImg[i] = self.cfarImg[i][0:100, 0:128 - shiftValue]
    
    def Adjust_colBlock(self, shiftValue):
        for i in range(0,6):
            if i < 3:
                print ("---")
                self.blockImg[i] = self.blockImg[i][0:500, 0 + shiftValue:128]
                self.cfarImg[i] = self.cfarImg[i][0:500, 0 + shiftValue:128]
            else:
                print ("+++")
                self.blockImg[i] = self.blockImg[i][0:500, 0:128 - shiftValue]
                self.cfarImg[i] = self.cfarImg[i][0:500, 0:128 - shiftValue]

    # ! For cfarImage
    def Calculate_Mean_Var(self):
        for i in range(0,30):
            # print (np.mean(self.blockImg[i]))
            self.meanList.append(np.mean(self.cfarImg[i]))
            self.varList.append(np.var(self.cfarImg[i]))
        self.averageMean = np.mean(self.meanList)
        self.averageVar = np.mean(self.varList)

    def Show_rowBlock(self, rowIndex):
        index = rowIndex * 6
        for i in range(0,6):
            plt.subplot(1,6,i),plt.imshow(self.blockImg[index + i], cmap = 'gray')
            plt.title("Block #" + str(index + i)), plt.xticks([]), plt.yticks([])
        plt.show()
    
    def Show_allBlock(self):
        for i in range(0,30):
            plt.subplot(5,6,i+1),plt.imshow(self.blockImg[i], cmap = 'gray')
            plt.title("Block #" + str(i)), plt.xticks([]), plt.yticks([])
        plt.show()
    
    def Show_cafarBlock(self):
        for i in range(0,30):
            plt.subplot(5,6,i+1),plt.imshow(self.cfarImg[i], cmap = 'gray')
            plt.title("cfarBlock #" + str(i)), plt.xticks([]), plt.yticks([])
        plt.show()

    def CheckAverageBlock(self):
        passMean = []
        passVar = []
        for i in range(0,30):
            if self.meanList[i] > self.averageMean:
                passMean.append(i)
            if self.varList[i] > self.averageVar:
                passVar.append(i)
        print (passMean)
        print (passVar)
        return passMean, passVar

    def Save_passBlock(self, passList):
        for i in range(0, len(passList)):
            cv2.imwrite(picPath + "\Result\BlockImg_" + str(passList[i]) + ".png", self.blockImg[passList[i]])
    
    def cfarBlock(self, box_size, guard_size, pfa, kernel_size):
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        for i in range(0,30):
            self.cfarImg[i] = cafar(self.cfarImg[i], box_size, guard_size, pfa)
            self.cfarImg[i] = cv2.morphologyEx(self.cfarImg[i], cv2.MORPH_OPEN, kernel)
