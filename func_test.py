import numpy as np
import surface3d
import sonarGeometry
import sonarPlotting
import cv2

a = np.array([[1,2,4],[4,3,5]])

# print (np.amax(a))
# print (a)
# print (a[1,0])
# x,y = np.unravel_index(np.argmax(a, axis=None), a.shape)

# print (x,y)

# picPath = "D:\Thesis\pic_sonar"
# inx = 1
# picName = "\RTheta_img_" + str(10) + ".jpg"
# img = cv2.imread(picPath + picName, 0)
# mask = cv2.imread("D:\Thesis\pic_sonar\mask\com_mask.png",0)

# blur_0 = cv2.GaussianBlur(mask,(9,9),0,0)
# blur_1 = cv2.GaussianBlur(mask,(9,9),1,1)
# blur_2 = cv2.GaussianBlur(mask,(9,9),2,2)
# blur_3 = cv2.GaussianBlur(mask,(9,9),3,3)
# blur_4 = cv2.GaussianBlur(mask,(9,9),4,4)

# cv2.imshow("0", blur_0)
# cv2.imshow("1", blur_1)
# cv2.imshow("2", blur_2)
# cv2.imshow("3", blur_3)
# cv2.imshow("4", blur_4)

for i in range(1,10):
    y = int(i/2)
    x = 0.5 + (0.1*y)
    z = 1.0 - x
    print (x, z)
    # print (x)

# kernel = np.ones((9,9),np.uint8)
# erosion = cv2.erode(blur_4,kernel,iterations = 1)
# img = sonarGeometry.remapping(img)
# gaussian = cv2.getGaussianKernel(240, 40)
# gaussian = gaussian * gaussian.T

# dst = cv2.filter2D(erosion, -1, gaussian)

# im1_fft = np.fft.fft2(img)
# im1_fft = np.fft.fftshift(im1_fft)
# kernel = cv2.getGaussianKernel(1320, 20)
# kernel = kernel * kernel.T
# kernel = kernel[260:1060,0:1320]
# out = im1_fft * kernel
# magnitude_spectrum = 20*np.log(np.abs(out))
# phaseShift = np.fft.fftshift(out)
# phaseShift = np.fft.ifft2(phaseShift)
# phaseShift = np.real(phaseShift)
# phaseShift = np.abs(phaseShift)


# # get first masked value (foreground)
# fg = cv2.bitwise_or(img, img, mask=mask)

# # get second masked value (background) mask must be inverted
# mask = cv2.bitwise_not(mask)
# background = np.full(img.shape, 255, dtype=np.uint8)
# bk = cv2.bitwise_or(background, background, mask=mask)

# # combine foreground+background
# final = cv2.bitwise_or(fg, bk)

# plot_name = ['fg', 'mask', 'bk', 'final']
# sonarPlotting.subplot4(fg, mask, bk, final, plot_name)

# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# res = cv2.bitwise_and(img,img,mask = dst)
# res = cv2.addWeighted(img,0.1,dst,0.9,1)
# res = cv2.bitwise_and(res,res,mask = dst)
# img.astype(np.float64)
# print (np.amax(img))

# dst.astype(np.float64)
# print (np.amax(img))

# res = np.ones((800,1320),np.uint8)
# res = img * dst
# print (np.amax(res))


# plot_name = ['img', 'gaussian']
# sonarPlotting.subplot2(img, dst, plot_name)

# cv2.imshow("res", res)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# surface3d.surface3d(10, 20)