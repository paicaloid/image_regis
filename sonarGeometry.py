import numpy as np
import cv2

def remapping(img):
    # ! need to collect parameter 
    # TODO check best parameter [810,660]
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