import numpy as np
import surface3d
import sonarGeometry
import cv2

a = np.array([[1,2,4],[4,3,5]])

print (np.amax(a))
print (a)
print (a[1,0])
x,y = np.unravel_index(np.argmax(a, axis=None), a.shape)

print (x,y)

picPath = "D:\Thesis\pic_sonar"
inx = 1
while(1):
    picName = "\RTheta_img_" + str(inx) + ".jpg"
    img = cv2.imread(picPath + picName, 0)

    cv2.imshow("Ori_image", img)
    tran_img = sonarGeometry.remapping(img)
    cv2.imshow("image", tran_img)
    cv2.waitKey(30)
    if inx == 100:
        break
    else:
        inx = inx + 1

cv2.destroyAllWindows()


# surface3d.surface3d(10, 20)