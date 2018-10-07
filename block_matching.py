import cv2
import numpy as np
import sonarGeometry
import sonarPlotting
import Feature_match
import matplotlib.pyplot as plt

picPath = "D:\Pai_work\pic_sonar"
plot_name2 = ['1', '2']
plot_name4 = ['1', '2', '3', '4']

def multiplyImage(img1, img2):
    # ! Change dtype to avoid overflow (unit8 -> int32)
    img1 = img1.astype(np.int32)
    img2 = img2.astype(np.int32)
    out = cv2.multiply(img1, img2)
    out = out/255.0
    out = out.astype(np.uint8)
    return out

def Create_BlockImage(img):
    # row, col = img.shape
    blockList = []
    for i in range(0,5):
        for j in range(0,6):
            block_img = img[i*100:(i+1)*100, j*128:(j+1)*128]
            blockList.append(block_img)
    # print (len(blockList))
    return blockList

def AverageMultiLook(start, stop):
    picName = "\RTheta_img_" + str(start) + ".jpg"
    ref = cv2.imread(picPath + picName, 0)
    for i in range(1, stop):
        picName = "\RTheta_img_" + str(start+i) + ".jpg"
        img = cv2.imread(picPath + picName, 0)
        ref = cv2.addWeighted(ref, 0.5, img, 0.5, 0)
    return ref

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

    #### Import image and Preprocessing ####
    ## image size [660 x 768]
    img_A = AverageMultiLook(10, 5)
    imgCrop_A = img_A[0:500, 0:768]
    img_B = AverageMultiLook(15, 5)
    imgCrop_B = img_B[0:500, 0:768]
    # img_C = AverageMultiLook(20, 5)
    # imgCrop_C = img_C[0:500, 0:768]
    # img_D = AverageMultiLook(25, 5)
    # imgCrop_D = img_D[0:500, 0:768]

    # sonarPlotting.subplot2(imgCrop_A, imgCrop_B, plot_name2)

    blockA = BlockImage()
    blockA.Creat_Block(imgCrop_A)
    # blockA.Show_allBlock()

    blockA.Adjsut_block(14)
    # blockA.Show_allBlock()

    blockB = BlockImage()
    blockB.Creat_Block(imgCrop_B)
    # blockB.Show_allBlock()

    blockB.Adjsut_block(14)
    # blockB.Show_allBlock()

    # blockC =BlockImage()
    # blockC.Creat_Block(imgCrop_C)

    # blockD = BlockImage()
    # blockD.Creat_Block(imgCrop_D)

    # num = 27
    # plot_name = ["Img10 block#" + str(num), "Img20 block#" + str(num), "Img30 block#" + str(num), "Img40 block#" + str(num)]
    # sonarPlotting.subplot4(blockA.blockImg[num], blockB.blockImg[num], blockC.blockImg[num],blockD.blockImg[num], plot_name)

    # print (blockA.averageMean, blockA.averageVar)
    ### Save Multilook image ###
    if False:
        for i in range(1,8):
            img = AverageMultiLook(i*10, 10)
            img = img[0:500, 0:768]
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.imwrite(picPath + "\Result\multilook" + str(i*10) + ".png", img)
        cv2.destroyAllWindows()
    ### Test Average PhaseShift of blockImage
    if True:
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
        
        
        # for i in range(0,30):
        #     M = np.float32([[1,0,shiftCol_adjust[i]],[0,1,shiftRow_adjust[i]]])
        #     res = cv2.warpAffine(blockA.blockImg[i], M, (128,100))
        #     dst = cv2.addWeighted(blockB.blockImg[i], 0.5, res, 0.5, 0)
        #     out = multiplyImage(blockB.blockImg[i], res)
        #     sonarPlotting.subplot4(blockB.blockImg[i], res, dst, out, plot_name4)

    ### Plot dots ###
    if False:
        plt.plot(shiftRow_adjust, shiftCol_adjust, 'o', label='Original data', markersize=1)
        plt.show()