import cv2
import numpy as np
import sonarGeometry
import sonarPlotting
import Feature_match

picPath = "D:\Pai_work\pic_sonar"

def create_FFT_image(img):
    img = np.fft.fft2(img)
    img = np.fft.fftshift(img)
    mag = 20*np.log(np.abs(img))
    return mag

def is_edge_reduce(img1, img2, blur1, blur2):
    fft_1 = create_FFT_image(img1)
    fft_2 = create_FFT_image(img2)
    b_1 = create_FFT_image(blur1)
    b_2 = create_FFT_image(blur2)
    plot_name = ['img1', 'img2', 'blur1', 'blur2']
    sonarPlotting.subplot4(fft_1, fft_2, b_1, b_2, plot_name)

def multiplyImage(img1, img2):
    # ! Change dtype to avoid overflow (unit8 -> int32)
    img1 = img1.astype(np.int32)
    img2 = img2.astype(np.int32)
    out = cv2.multiply(img1, img2)
    out = out/255.0
    out = out.astype(np.uint8)
    return out

def AverageMultiLook(start, stop):
    picName = "\RTheta_img_" + str(start) + ".jpg"
    ref = cv2.imread(picPath + picName, 0)
    for i in range(1, stop):
        picName = "\RTheta_img_" + str(start+i) + ".jpg"
        img = cv2.imread(picPath + picName, 0)
        ref = cv2.addWeighted(ref, 0.5, img, 0.5, 0)
    return ref
    
def multiLook(start, stop):
    picName = "\RTheta_img_" + str(start) + ".jpg"
    ref = cv2.imread(picPath + picName, 0)
    sve = cv2.imread(picPath + picName, 0)
    for i in range(1, stop):
        picName = "\RTheta_img_" + str(start+i) + ".jpg"
        img = cv2.imread(picPath + picName, 0)
        inx = int(i/2)
        weight = 0.5 + (0.1 * inx)
        ref = cv2.addWeighted(ref, weight, img, 1.0 - weight, 0)
    return ref

def findPhaseshift(img1, img2):
    fft_1 = np.fft.fft2(img1)
    fft_1 = np.fft.fftshift(fft_1)
    fft_2 = np.fft.fft2(img2)
    fft_2 = np.fft.fftshift(fft_2)

    A = fft_1 * np.conj(fft_2)
    B = np.abs(A)
    phaseShift = A/B
    out = np.fft.fftshift(phaseShift)
    out = np.fft.ifft2(out)
    out = np.real(out)
    out = np.abs(out)
    return out

def shiftPosition(shifting):
    positionMax = np.amax(shifting)
    print (positionMax)
    height, width = shifting.shape
    print (np.unravel_index(np.argmax(shifting, axis=None), shifting.shape))

if __name__ == '__main__':

    ## Preprocessing image
    img1 = AverageMultiLook(10, 10)
    img2 = AverageMultiLook(30, 10)
    # img3 = cv2.medianBlur(img1, 5)
    # img4 = cv2.medianBlur(img2, 5)

    ## Crop image
    # img1 = img1[0:500,0:768]
    # img2 = img2[0:500,0:768]
    # img3 = img3[0:500,0:768]
    # img4 = img4[0:500,0:768]

    ## Rotation
    # M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    # img1 = cv2.warpAffine(img1,M,(cols,rows))
    # img2 = cv2.warpAffine(img2,M,(cols,rows))
    # img3 = cv2.warpAffine(img3,M,(cols,rows))
    # img4 = cv2.warpAffine(img4,M,(cols,rows))

    ## Remap : RTheta -> XY
    img1 = sonarGeometry.remapping(img1)
    img2 = sonarGeometry.remapping(img2)

    ## Create mask for reduce edge
    mask = cv2.imread(picPath + "\mask\com_mask.png",0)
    kernel = np.ones((11, 11),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 4)
    gauss_blur = cv2.GaussianBlur(erosion,(143,143),24,24)

    # gaussian = cv2.getGaussianKernel(240, 40)
    # gaussian = gaussian * gaussian.T
    # mask_b = cv2.filter2D(erosion, -1, gaussian)

    ## Reduce edge effect
    img1_blur = multiplyImage(img1, gauss_blur)
    img2_blur = multiplyImage(img2, gauss_blur)

    plot_name = ['img1', 'img2', 'blur1', 'blur2']
    sonarPlotting.subplot4(img1, img2, img1_blur, img2_blur, plot_name)
    is_edge_reduce(img1, img2, img1_blur, img2_blur)

    ## Calculate PhaseShift
    org_phase = findPhaseshift(img1, img2)
    shiftPosition(org_phase)
    blur_phase = findPhaseshift(img1_blur, img2_blur)
    shiftPosition(blur_phase)
    sonarPlotting.plotPhase(org_phase)
    sonarPlotting.plotPhase(blur_phase)



    
