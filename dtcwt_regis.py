import numpy as np
import cv2
import dtcwt
import dtcwt.registration as registration

import sonarPlotting

if __name__ == '__main__':
    ref = cv2.imread("D:\Thesis\image_regis\multiLook_1.jpg", 0)
    src = cv2.imread("D:\Thesis\image_regis\multiLook_2.jpg", 0)

    transform = dtcwt.Transform2d()
    ref_t = transform.forward(ref, nlevels=6)
    src_t = transform.forward(src, nlevels=6)

    reg = registration.estimatereg(src_t, ref_t)
    warped_src = registration.warp(src, reg, method='bilinear')

    plot_name = ['ref', 'src', 'ref_t', 'src_t']
    sonarPlotting.subplot4(ref, src, ref, warped_src, plot_name)

    cv2.imshow("warped", warped_src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()