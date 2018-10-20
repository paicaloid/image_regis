import cv2
import numpy as np
import sonarGeometry
import sonarPlotting
import FeatureMatch
import matplotlib.pyplot as plt

picPath = "D:\Pai_work\pic_sonar"
plot_name2 = ['1', '2']
plot_name4 = ['1', '2', '3', '4']

def AverageMultiLook(start, stop):
    picName = "\RTheta_img_" + str(start) + ".jpg"
    ref = cv2.imread(picPath + picName, 0)
    for i in range(1, stop):
        picName = "\RTheta_img_" + str(start+i) + ".jpg"
        img = cv2.imread(picPath + picName, 0)
        ref = cv2.addWeighted(ref, 0.5, img, 0.5, 0)
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

class BlockImage:
    def __init__(self):
        self.blockImg = []
        self.meanList = []
        self.varList = []
        self.averageMean = 0
        self.averageVar = 0

    def Creat_Block(self, img):
        for i in range(0,5):
            for j in range(0,6):
                block_img = img[i*100:(i+1)*100, j*128:(j+1)*128]
                self.blockImg.append(block_img)

    def Adjsut_block(self, shiftValue):
        for i in range(0,30):
            if np.mod(i,6) < 3:
                self.blockImg[i] = self.blockImg[i][0:100, 0 + shiftValue:128]
            else:
                self.blockImg[i] = self.blockImg[i][0:100, 0:128 - shiftValue]
    
    def Calculate_Mean_Var(self):
        for i in range(0,30):
            # print (np.mean(self.blockImg[i]))
            self.meanList.append(np.mean(self.blockImg[i]))
            self.varList.append(np.var(self.blockImg[i]))
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

if __name__ == '__main__':

    ### Import image and Preprocessing ###
    img_A = AverageMultiLook(10, 5)
    imgCrop_A = img_A[0:500, 0:768]
    img_B = AverageMultiLook(15, 5)
    imgCrop_B = img_B[0:500, 0:768]

    blockA = BlockImage()
    blockA.Creat_Block(imgCrop_A)
    blockA.Adjsut_block(14)

    blockB = BlockImage()
    blockB.Creat_Block(imgCrop_B)
    blockB.Adjsut_block(14)

    ### Test Average PhaseShift of blockImage
    if False:
        shiftRow = []
        shiftRow_adjust = []
        shiftCol = []
        shiftCol_adjust = []
        for i in range(0,30):
            phase = findPhaseshift(blockB.blockImg[i], blockA.blockImg[i])
            x_shift, y_shift = findShiftPosition(phase)
            # print (position)
            shiftRow.append(x_shift)
            shiftRow_adjust.append(x_shift)
            shiftCol.append(y_shift)
            shiftCol_adjust.append(y_shift)
        # print (shiftRow, shiftCol)
        for i in range(0,30):
            if shiftRow_adjust[i] > 50:
                shiftRow_adjust[i] = 50 - shiftRow_adjust[i]
            if shiftCol_adjust[i] > 64:
                shiftCol_adjust[i] = 64 - shiftCol_adjust[i]
            # print (shiftRow[i]+","+shiftCol[i] + "-->" + shiftRow_adjust[i]+","+shiftCol_adjust[i] )
        # print (shiftRow, shiftCol)
        print (shiftRow_adjust)
        print (shiftCol_adjust)
           
        for i in range(0,30):
            M = np.float32([[1,0,shiftCol_adjust[i]],[0,1,shiftRow_adjust[i]]])
            res = cv2.warpAffine(blockA.blockImg[i], M, (114,100))
            dst = cv2.addWeighted(blockB.blockImg[i], 0.5, res, 0.5, 0)
            out = multiplyImage(blockB.blockImg[i], res)
            plot_name = ["Img15_Block" + str(i), "ShiftRow : " + str(shiftRow_adjust[i]) + ", ShiftCol : " + str(shiftCol_adjust[i]), "addWeighted", "multiply"]
            # sonarPlotting.subplot4(blockB.blockImg[i], res, dst, out, plot_name)
            sonarPlotting.saveplot4(blockB.blockImg[i], res, dst, out, plot_name, "subplot" + str(i) + ".png")

    ### BlockImage Feature Match ###
    if True:
        ref_Position = []
        shift_Position = []
        for i in range(0,30):
            saveName = picPath + "\exp\Img10_SIFT_Img15_#" + str(i) + ".png"
            # print ("Round : " + str(i))
            refPos, shiftPos = FeatureMatch.matchPosition_BF(blockA.blockImg[i], blockB.blockImg[i])
            if len(refPos) != 0:
                for m in refPos:
                    ref_Position.append(m)
            if len(shiftPos) != 0:
                for n in shiftPos:
                    shift_Position.append(n)
        print (len(ref_Position))
        print (len(shift_Position))
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

        ## rewrite the line equation as y = Ap
        ## where A = [[x y 1]] and p = [[m], [c]]
        # TODO : Calculate affine for output_row
        print (type(input_row))
        vectorA = np.vstack([input_row, input_col, np.ones(len(input_row))]).T
        a2, a1, a0 = np.linalg.lstsq(vectorA, output_row)[0]

        error_a = 0
        errThres = 10.0
        errInx = []
        for i in range(0,len(input_row)):
            res = (a2 * input_row[i]) + (a1 * input_col[i]) + a0
            # print (output_row[i], res)
            error_a = error_a + np.abs(output_row[i] - res)
            print ("Error " + str(i) + " : " + str(np.abs(output_row[i] - res)))
            if np.abs(output_row[i] - res) > errThres:
                errInx.append(i)
        print (errInx)
        print (error_a/30.0)

        # ! Test re-Calculate affine by remove some input (error > 10)
        # for i in errInx:
        #     input_row.remove(input_row[i])
        #     input_col.remove(input_col[i])
        #     output_row.remove(output_row[i])
        
        # vectorA_Prime = np.vstack([input_row, input_col, np.ones(len(input_row))]).T
        # a2, a1, a0 = np.linalg.lstsq(vectorA_Prime, output_row)[0]

        # error_a = 0
        # errThres = 10.0
        # errInx = []
        # for i in range(0,len(input_row)):
        #     res = (a2 * input_row[i]) + (a1 * input_col[i]) + a0
        #     # print (output_row[i], res)
        #     error_a = error_a + np.abs(output_row[i] - res)
        #     print ("Error " + str(i) + " : " + str(np.abs(output_row[i] - res)))
        #     if np.abs(output_row[i] - res) > errThres:
        #         errInx.append(i)
        # print (errInx)
        # print (error_a/30.0)
        
        # TODO : Calculate affine for output_col
        vectorB = np.vstack([input_row, input_col, np.ones(len(input_row))]).T
        b2, b1, b0 = np.linalg.lstsq(vectorB, output_col)[0]

        error_b = 0
        errInx = []
        for i in range(0,len(input_row)):
            res = (b2 * input_row[i]) + (b1 * input_col[i]) + b0
            error_b = error_b + np.abs(output_col[i] - res)
            print ("Error " + str(i) + " : " + str(np.abs(output_col[i] - res)))
            if np.abs(output_col[i] - res) > errThres:
                 errInx.append(i)
        print (error_b/30.0)

        # ! Test remap
        y,x = np.mgrid[:img_A.shape[0],:img_A.shape[1]]
        New_x = (a2 * x) + (a1 * y) + a0
        New_y = (b2 * x) + (b1 * y) + b0

        out1 = cv2.remap(img_A,New_x.astype('float32'),New_y.astype('float32'),cv2.INTER_LINEAR)

        # cv2.imshow("Img1", out1)
        # cv2.imshow("Img2", img_B)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
