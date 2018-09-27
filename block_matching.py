import cv2
import numpy as np
import sonarGeometry
import sonarPlotting
import Feature_match

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
    print (np.unravel_index(np.argmax(shifting, axis=None), shifting.shape))
    return shift

if __name__ == '__main__':

    #### Import image and Preprocessing ####
    ## image size [660 x 768]
    img_A = AverageMultiLook(10, 10)
    imgCrop_A = img_A[0:500, 0:768]

    img_B = AverageMultiLook(20, 10)
    imgCrop_B = img_B[0:500, 0:768]

    ImageList_A = Create_BlockImage(imgCrop_A)
    ImageList_B = Create_BlockImage(imgCrop_B)

    sonarPlotting.subplot2(imgCrop_A, imgCrop_B, plot_name2)

    rowList = []
    colList = []

    for i in range(0,30):
        phase = findPhaseshift(ImageList_B[i], ImageList_A[i])
        position = findShiftPosition(phase)
        rowList.append(position[0])
        colList.append(position[1])
    
    rowShift = int(np.round(np.mean(rowList)))
    colShift = int(np.round(np.mean(colList)))
    print (rowShift, colShift)

    tranform_Matrix = np.float32([[1,0,colShift],[0,1,rowShift]])
    result = cv2.warpAffine(imgCrop_A, tranform_Matrix, (768,500))

    dst = cv2.addWeighted(imgCrop_B, 0.5, result, 0.5, 0)
    out = multiplyImage(imgCrop_B, result)

    cv2.imshow("imgB", dst)
    cv2.imshow("mul", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()