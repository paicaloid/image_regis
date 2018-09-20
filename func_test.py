import numpy as np
import sonarGeometry
import sonarPlotting
import cv2

picPath = "D:\Pai_work\pic_sonar"
inv_mask = cv2.imread(picPath + "\mask\inv_rr.png",0)

if __name__ == '__main__':

    # ! remap make process slower than cv2.inpaint
    for i in range(1,200):
        picName = "\RTheta_img_" + str(i) + ".jpg"
        img = cv2.imread(picPath + picName, 0)
        # img = sonarGeometry.remapping(img)

        dst = cv2.inpaint(img,inv_mask,3,cv2.INPAINT_TELEA)

        cv2.imshow("Image", dst)
        cv2.waitKey(30)

    cv2.destroyAllWindows()