import numpy as np
import sonarGeometry
import sonarPlotting
import cv2

picPath = "D:\Pai_work\pic_sonar"
inv_mask = cv2.imread(picPath + "\mask\inv_rr.png",0)

plot_name2 = ['1', '2']
plot_name4 = ['1', '2', '3', '4']

def create_FFT_image(img):
    img = np.fft.fft2(img)
    img = np.fft.fftshift(img)
    mag = 20*np.log(np.abs(img))
    return mag

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

def shiftPosition(shifting):
    positionMax = np.amax(shifting)
    print (positionMax)
    height, width = shifting.shape
    print (np.unravel_index(np.argmax(shifting, axis=None), shifting.shape))

if __name__ == '__main__':

    if False:
    # ! remap make process slower than cv2.inpaint
        for i in range(1,200):
            picName = "\RTheta_img_" + str(i) + ".jpg"
            img = cv2.imread(picPath + picName, 0)
            # img = sonarGeometry.remapping(img)

            dst = cv2.inpaint(img,inv_mask,3,cv2.INPAINT_TELEA)

            cv2.imshow("Image", dst)
            cv2.waitKey(30)

        cv2.destroyAllWindows()
    
    if True:
        img_L = cv2.imread("D:\Pai_work\panorama-stitching\images\Bryce_left_02.png",0)
        img_R = cv2.imread("D:\Pai_work\panorama-stitching\images\Bryce_right_02.png",0)

        # cv2.imshow("L", img_L)
        # cv2.imshow("R", img_R)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        mag1 = create_FFT_image(img_L)
        mag2 = create_FFT_image(img_R)

        # sonarPlotting.subplot2(mag1, mag2, plot_name2)

        dst = findPhaseshift(img_L, img_R)
        shiftPosition(dst)
        # sonarPlotting.plotPhase(dst)

        rows,cols = img_R.shape

        M = np.float32([[1,0,0],[0,1, -14]])
        out = cv2.warpAffine(img_R,M,(cols,rows))

        cv2.imshow("out", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # ref = cv2.addWeighted(img_L, 0.5, out, 0.5, 0)

        # cv2.imshow("ref", ref)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # sonarPlotting.subplot4(img_L, img_R, out, ref, plot_name4)