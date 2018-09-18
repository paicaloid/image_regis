import cv2
import numpy as np
import sonarGeometry
import sonarPlotting
import Feature_match

picPath = "D:\Thesis\pic_sonar"

def create_FFT_image(img):
    img = np.fft.fft2(img)
    img = np.fft.fftshift(img)
    mag = 20*np.log(np.abs(img))
    return mag

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
    positionMax = np.amax(phaseShift)
    print (positionMax)
    height, width = phaseShift.shape
    print (np.unravel_index(np.argmax(phaseShift, axis=None), phaseShift.shape))

def AverageMultiLook(start, stop):
    picName = "\RTheta_img_" + str(start) + ".jpg"
    ref = cv2.imread(picPath + picName, 0)
    sve = cv2.imread(picPath + picName, 0)
    for i in range(1, stop):
        picName = "\RTheta_img_" + str(start+i) + ".jpg"
        # print (picName)
        img = cv2.imread(picPath + picName, 0)
        ref = cv2.addWeighted(ref, 0.5, img, 0.5, 0)
    plot_name = ['ref', 'multiLook']
    # sonarPlotting.subplot2(sve, ref, plot_name)
    return ref
    
def multiLook(start, stop):
    picName = "\RTheta_img_" + str(start) + ".jpg"
    ref = cv2.imread(picPath + picName, 0)
    sve = cv2.imread(picPath + picName, 0)
    for i in range(1, stop):
        picName = "\RTheta_img_" + str(start+i) + ".jpg"
        # print (picName)
        img = cv2.imread(picPath + picName, 0)
        inx = int(i/2)
        weight = 0.5 + (0.1 * inx)
        ref = cv2.addWeighted(ref, weight, img, 1.0 - weight, 0)
    plot_name = ['ref', 'multiLook']
    # sonarPlotting.subplot2(sve, ref, plot_name)
    return ref

if __name__ == '__main__':

    ## Preprocessing image
    img1 = AverageMultiLook(10, 10)
    img2 = AverageMultiLook(30, 10)
    img3 = cv2.medianBlur(img1, 5)
    img4 = cv2.medianBlur(img2, 5)

    ## Crop image
    # img1 = img1[0:500,0:768]
    img2 = img2[0:500,0:768]
    img3 = img3[0:500,0:768]
    img4 = img4[0:500,0:768]

    ## Rotation
    # M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    # img1 = cv2.warpAffine(img1,M,(cols,rows))
    # img2 = cv2.warpAffine(img2,M,(cols,rows))
    # img3 = cv2.warpAffine(img3,M,(cols,rows))
    # img4 = cv2.warpAffine(img4,M,(cols,rows))

    ## Remap : RTheta -> XY
    img1 = sonarGeometry.remapping(img1)
    # img2 = sonarGeometry.remapping(img2)

    img1_fft = create_FFT_image(img1)

    # plot_name = ['1','2','3','4']
    # sonarPlotting.subplot4(img1, img2, img3, img4, plot_name)

    ## Mask for reduce edge
    mask = cv2.imread("D:\Thesis\pic_sonar\mask\com_mask.png",0)
    kernel = np.ones((11, 11),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 4)
    blur_4 = cv2.GaussianBlur(erosion,(143,143),24,24)

    gaussian = cv2.getGaussianKernel(240, 40)
    gaussian = gaussian * gaussian.T
    mask_b = cv2.filter2D(erosion, -1, gaussian)

    plot_name = ['mask1', 'mask2']
    sonarPlotting.subplot2(blur_4, mask_b, plot_name)

    ## Change dtype to avoid overflow (unit8 -> int32)
    img1 = img1.astype(np.int32)
    mask_b = mask_b.astype(np.int32)
    blur_4 = blur_4.astype(np.int32)
    out = cv2.multiply(img1, mask_b)
    out = out/255.0
    out = out.astype(np.uint8)
    
    dst = cv2.multiply(img1, blur_4)
    dst = dst/255.0
    dst = dst.astype(np.uint8)

    plot_name = ['mask', 'result']
    sonarPlotting.subplot2(out, dst, plot_name)

    out_fft = create_FFT_image(out)
    dst_fft = create_FFT_image(dst)

    plot_name = ['img_fft', 'mask_fft', '3', '4']
    sonarPlotting.subplot4(img1_fft, img1_fft, out_fft, dst_fft, plot_name)

    
    # Feature_match.matching(img1, img2)
    # Feature_match.matching(img3, img4)


    
