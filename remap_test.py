import numpy as np
import cv2

def remap(img):
    rows,cols = img.shape
    y,x = np.mgrid[:800*2,:800*2]     #y, x : flip
    r = np.sqrt((x-810)**2+(660*2-y)**2)
    theta = (np.arctan2(x-810,660*2-y)*130./np.pi)
    thetapic = (theta+65.0)/0.12

    # Rotate image before remapping
    M1 = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
    dst1 = cv2.warpAffine(img,M1,(cols,rows))

    # Remapping
    out1 = cv2.remap(dst1,r.astype('float32'),thetapic.astype('float32'),cv2.INTER_LINEAR)

    M2 = cv2.getRotationMatrix2D((2*cols, rows),-35,0.9)
    dst2 = cv2.warpAffine(out1,M2,(660*2,660*2))
    return dst2

if __name__ == '__main__':
    picPath = "D:\Thesis\pic_sonar"
    picName = "\RTheta_img_" + str(10) + ".jpg"
    img = cv2.imread(picPath + picName, 0)
    row, col = img.shape

    X, Y = np.mgrid[:col*2,:row*2]
    # print (x.shape)
    R =np.sqrt((X-col/2)**2 + (Y-row)**2)
    # print (R.min(), R.max())
    angle = (np.arctan2(Y - row, X - col/2)*130./np.pi)
    Theta = (angle+180.0)/0.18
    # print (angle.min(), angle.max())
    # print (Theta.min(), Theta.max())

    # print ("++++++++++++++++")

    y,x = np.mgrid[:512,:512]
    r = np.sqrt((x-255)**2+(y-511)**2)  #511 shift down 511
    # print (r.min(),r.max())
    theta = (np.arctan2(y-511,x-255)*180./np.pi) #x-255,511-y
    thetapic = (theta+180)/0.45
    # print (theta.min(),theta.max())
    # print (thetapic.min(), thetapic.max())

    lena = cv2.remap(img, R.astype('float32'), Theta.astype('float32'),cv2.INTER_LINEAR)

    xxx = remap(img)

    # cv2.imshow("original", img)
    cv2.imshow("Remap", xxx)
    cv2.waitKey(0)
    cv2.destroyAllWindows()