import cv2
import numpy as np
import sonarGeometry
import sonarPlotting

def findPhasewithGaussian(img_1, img_2):
    im1_fft = np.fft.fft2(img_1)
    im2_fft = np.fft.fft2(img_2)
    im1_fft = np.fft.fftshift(im1_fft)
    im2_fft = np.fft.fftshift(im2_fft)
    mag_1 = 20*np.log(np.abs(im1_fft))
    mag_2 = 20*np.log(np.abs(im2_fft))

    kernel = cv2.getGaussianKernel(1320, 50)
    kernel = kernel * kernel.T
    kernel = kernel[260:1060,0:1320]
    im1_fft = im1_fft * kernel
    im2_fft = im2_fft * kernel

    g_1 = 20*np.log(np.abs(im1_fft))
    g_2 = 20*np.log(np.abs(im2_fft))

    plotName = ['img1', 'img2', 'g1', 'g2']
    sonarPlotting.subplot4(mag_1, mag_2, g_1, g_2, plotName)

    A = im1_fft * np.conj(im2_fft)
    B = np.abs(A)
    phaseShift = A/B
    phaseShift = A/B
    # phaseShift = np.fft.fftshift(phaseShift)
    phaseShift = np.fft.ifft2(phaseShift)
    phaseShift = np.real(phaseShift)
    phaseShift = np.abs(phaseShift)

    return phaseShift

def findPhaseshift(img_1, img_2):
    img_1 = img_1[200:600,420:900]
    img_2 = img_2[200:600,420:900]
    im1_fft = np.fft.fft2(img_1)
    im2_fft = np.fft.fft2(img_2)
    im1_fft = np.fft.fftshift(im1_fft)
    im2_fft = np.fft.fftshift(im2_fft)

    mag_1 = 20*np.log(np.abs(im1_fft))
    mag_2 = 20*np.log(np.abs(im2_fft))
    plotName = ['img1', 'img2', 'g1', 'g2']
    sonarPlotting.subplot4(mag_1, mag_2, img_1, img_2, plotName)

    A = im1_fft * np.conj(im2_fft)
    B = np.abs(A)
    phaseShift = A/B
    phaseShift = np.fft.fftshift(phaseShift)
    phaseShift = np.fft.ifft2(phaseShift)
    phaseShift = np.real(phaseShift)
    phaseShift = np.abs(phaseShift)
    return phaseShift

if __name__ == '__main__':
    # ! Load image number 1 and 70 
    # ? Test for find a phase different

    mode = 1
    if mode :
        picPath = "D:\Thesis\pic_sonar"
        picName = "\RTheta_img_" + str(10) + ".jpg"
        im1 = cv2.imread(picPath + picName, 0)
        im1 = sonarGeometry.remapping(im1)
        picName = "\RTheta_img_" + str(40) + ".jpg"
        im2 = cv2.imread(picPath + picName, 0)
        im2 = sonarGeometry.remapping(im2)

        rows,cols = im1.shape
        print (im1.shape)

        # im1 = im1[200:600,420:900]
        # im2 = im2[200:600,420:900]
    else:
        picPath = "D:\Thesis\pic_sonar\XY"
        picName = "\XY_img_" + str(1) + ".jpg"
        im1 = cv2.imread(picPath + picName, 0)
        im1 = im1[160:370,380:760]
        picName = "\XY_img_" + str(44) + ".jpg"
        im2 = cv2.imread(picPath + picName, 0)
        im2 = im2[160:370,380:760]



    shifting = findPhaseshift(im1, im2)
    sonarPlotting.plotPhase(shifting)
    
    positionMax = np.amax(shifting)
    print (positionMax)
    height, width = shifting.shape
    print (np.unravel_index(np.argmax(shifting, axis=None), shifting.shape))

    gauss_shift = findPhasewithGaussian(im1, im2)
    sonarPlotting.plotPhase(gauss_shift)

    positionMax = np.amax(gauss_shift)
    print (positionMax)
    h, w = gauss_shift.shape
    print (np.unravel_index(np.argmax(gauss_shift, axis=None), gauss_shift.shape))


    # cv2.imshow("image1", im1)
    # cv2.imshow("image2", im2)
    # cv2.imshow("multiLook", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
