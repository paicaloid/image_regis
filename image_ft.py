import cv2
import numpy as np
import surface3d

import scipy.misc

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

## subplot 4 slot [2x2]
def subplot4(im1, im2, im3, im4, title):
    plt.subplot(221),plt.imshow(im1, cmap = 'gray')
    plt.title(title[0]), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(im2, cmap = 'gray')
    plt.title(title[1]), plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(im3, cmap = 'gray')
    plt.title(title[2]), plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(im4, cmap = 'gray')
    plt.title(title[3]), plt.xticks([]), plt.yticks([])
    plt.show()

## subplot 2 slot [1x2] 
def subplot2(im1, im2, title):
    plt.subplot(121),plt.imshow(im1, cmap = 'gray')
    plt.title(title[0]), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(im2, cmap = 'gray')
    plt.title(title[1]), plt.xticks([]), plt.yticks([])
    plt.show()

## plot Cross Power Spectrum
def plotPhase(phaseArray):
    # downscaling has a "smoothing" effect
    lena = scipy.misc.imresize(phaseArray, 0.15, interp='cubic')

    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]

    # create the figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, lena ,rstride=1, cstride=1, cmap=plt.cm.inferno, linewidth=0)

    # show it
    plt.show()

if __name__ == '__main__':
    # ! Load image number 1 and 70 
    # ? Test for find a phase different
    # picIndex = 1
    # picPath = "D:\Thesis\pic_sonar"
    # picName = "\RTheta_img_" + str(10) + ".jpg"
    # img = cv2.imread(picPath + picName, 0)
    # img = img[0:550,0:768]
    # picName = "\RTheta_img_" + str(80) + ".jpg"
    # img2 = cv2.imread(picPath + picName, 0)
    # img2 = img2[0:550,0:768]

    picPath = "D:\Thesis\pic_sonar\XY"
    picName = "\XY_img_" + str(1) + ".jpg"
    img = cv2.imread(picPath + picName, 0)
    img = img[160:370,380:760]
    picName = "\XY_img_" + str(44) + ".jpg"
    img2 = cv2.imread(picPath + picName, 0)
    img2 = img2[160:370,380:760]

    img_ft = np.fft.fft2(img)
    img_ft_shift = np.fft.fftshift(img_ft)
    mag_spec = 20 * np.log(np.abs(img_ft_shift))

    img2_ft = np.fft.fft2(img2)
    img2_ft_shift = np.fft.fftshift(img2_ft)
    mag2_spec = 20 * np.log(np.abs(img2_ft_shift))

    plot_name = ["Input Image 1", "Magnitude Spectrum 1", "Input Image 2", "Magnitude Spectrum 2"]
    subplot4(img, mag_spec, img2, mag2_spec, plot_name)

    C = img_ft_shift * np.conj(img2_ft_shift)
    D = np.abs(C)
    shift = C/D
    shift = np.fft.fftshift(shift)
    inv = np.fft.ifft2(shift)
    inv = np.real(inv)
    inv = np.abs(inv)
    # print (np.amax(inv))

    A = img_ft * np.conj(img2_ft)
    B = np.abs(A)
    phaseShift = A/B
    phaseShift = np.fft.fftshift(phaseShift)
    crossShift = np.fft.ifft2(phaseShift)
    crossShift = np.real(crossShift)
    crossShift = np.abs(crossShift)

    positionMax = np.amax(crossShift)
    print (positionMax)
    height, width = crossShift.shape
    row, col = np.unravel_index(np.argmax(crossShift, axis=None), crossShift.shape)

    # surface3d.plot(height, width, crossShift)
    
    # for i in range (0,height):
    #     for j in range (0,width):
    #         if crossShift[i][j] == positionMax:
    #             print ("Cross")
    #             print (i,j)
            # if inv[i][j] == positionMax:
            #     print ("inv")
            #     print (i,j)

    M = np.float32([[1,0,-col],[0,1,row]])
    plotPhase(crossShift)
    dst = cv2.warpAffine(img2, M, (width,height))
    plot_name = ["Original", "Shift"]
    subplot2(img, dst, plot_name)

    # cross_spectrum = (mag_spec * np.conj(mag2_spec))/np.abs(mag_spec * np.conjugate(mag2_spec))
    # cross_shift = np.fft.ifftshift(cross_spectrum)
    # img_back = np.fft.ifft2(cross_shift)
    # img_back = np.abs(img_back)


    # cv2.imshow("FFT_image", cross_spectrum)
    # cv2.imshow("IFFT_image", img_back)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # k = cv2.waitKey(0)
    # if k == 27:
    #     break
    # picIndex += 1
    # if cv2.waitKey(30) & 0xFF == ord('q'):
    #     break

    # cv2.destroyAllWindows()