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

    plot_name2 = ['1', '2']
    plot_name4 = ['1', '2', '3', '4']

    ## Preprocessing image ##
    img1 = AverageMultiLook(10, 10)
    img2 = AverageMultiLook(30, 10)
    # img3 = cv2.medianBlur(img1, 5)
    # img4 = cv2.medianBlur(img2, 5)

    if False:
        ## Crop image ##
        img1 = img1[0:500,0:768]
        img2 = img2[0:500,0:768]
        img3 = img3[0:500,0:768]
        img4 = img4[0:500,0:768]

        ## Rotation ##
        M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        img1 = cv2.warpAffine(img1,M,(cols,rows))
        img2 = cv2.warpAffine(img2,M,(cols,rows))
        img3 = cv2.warpAffine(img3,M,(cols,rows))
        img4 = cv2.warpAffine(img4,M,(cols,rows))

        ## Remap : RTheta -> XY ##
        img1 = sonarGeometry.remapping(img1)
        img2 = sonarGeometry.remapping(img2)

    ## Create mask for reduce edge ##
    # ! Normal mask [remove outside edge]
    mask = cv2.imread(picPath + "\mask\com_mask.png",0)
    kernel = np.ones((11, 11),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 4)
    ksize = 199
    gauss_blur = cv2.GaussianBlur(erosion,(ksize,ksize),10)
    
    # ! Special mask [remove inside edge]
    # TODO 1: blur -> remap -> erosion -> blur [not work]
    if False:
        s_mask = cv2.imread(picPath + "\mask\mask_rr.png",0)
        ksize = 31
        gauss_r = cv2.GaussianBlur(s_mask,(ksize,ksize),20)

        cv2.imshow("eee", gauss_r)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        gauss_remap = sonarGeometry.remapping(gauss_r)

        gauss_eros = cv2.erode(gauss_remap,kernel,iterations = 4)

        ksize = 199
        gauss_bb = cv2.GaussianBlur(gauss_eros,(ksize,ksize),10)

        A = multiplyImage(img1, gauss_blur)
        B = multiplyImage(img1, gauss_bb)

        sonarPlotting.subplot4(A, B, gauss_eros, gauss_bb, plot_name4)

    # TODO 2: blur -> eros [not work]
    if False:
        s_mask = cv2.imread(picPath + "\mask\mask_rr.png",0)
        imgAdd_mask = multiplyImage(img1, s_mask)
        
        kernel_2 = np.ones((5, 5),np.uint8)
        dilat = cv2.dilate(img1,kernel_2,iterations = 1)

        closing = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel_2)

        sonarPlotting.subplot2(closing, dilat, plot_name2)

    # TODO 3: cv2.inpaint -> remap -> eros -> blur
    inv_mask = cv2.imread(picPath + "\mask\inv_rr.png",0)
    img1_paint = cv2.inpaint(img1,inv_mask,3,cv2.INPAINT_TELEA)
    img1_remap = sonarGeometry.remapping(img1_paint)
    img1_blur = multiplyImage(img1_remap, gauss_blur)

    img2_paint = cv2.inpaint(img2,inv_mask,3,cv2.INPAINT_TELEA)
    img2_remap = sonarGeometry.remapping(img2_paint)
    img2_blur = multiplyImage(img2_remap, gauss_blur)

    img1_fft = create_FFT_image(img1_blur)
    img2_fft = create_FFT_image(img2_blur)

    sonarPlotting.subplot4(img1_blur, img2_blur, img1_fft, img2_fft, plot_name4)

    dst = findPhaseshift(img1_blur, img2_blur)
    shiftPosition(dst)

    # cv2.imshow("eee", img_blur)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # A = multiplyImage(img1, gauss_blur)
    # B = multiplyImage(img1, gauss_r)

    # sonarPlotting.subplot4(gauss_blur, gauss_r, A, B, plot_name)
    # gaussian = cv2.getGaussianKernel(240, 40)
    # gaussian = gaussian * gaussian.T
    # mask_b = cv2.filter2D(erosion, -1, gaussian)
    if False:
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



    
