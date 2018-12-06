#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

import regisProcessing as rp

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


if __name__ == '__main__':
    print('loading images...')
    # ? AloeImage
    # imgL = cv.pyrDown(cv.imread("D:\Pai_work\opencv\samples\data\\aloeL.jpg"))
    # imgR = cv.pyrDown(cv.imread("D:\Pai_work\opencv\samples\data\\aloeR.jpg"))

    # ? sonarImage
    # imgL = cv.imread("D:\Pai_work\pic_sonar\RTheta_img_10.jpg")
    # imgR = cv.imread("D:\Pai_work\pic_sonar\RTheta_img_50.jpg")
    # imgL = imgL[0:500, 0:768]
    # imgR = imgR[0:500, 0:768]

    # ? Multilook sonarImage
    imgL = rp.ColorMultiLook(10,10)
    imgR = rp.ColorMultiLook(50,10)

    # ? Rotate sonarImage
    # imgL = cv.imread("D:\Pai_work\pic_sonar\Rotate_img_10.jpg")
    # imgR = cv.imread("D:\Pai_work\pic_sonar\Rotate_img_50.jpg")
    # imgL = imgL[0:768, 0:500]
    # imgR = imgR[0:768, 0:500]

    # ? Test stereoImage [PC Monitor]
    # imgL = cv.imread("D:\Pai_work\pic_sonar\stereoTest\TestL3.jpg")
    # imgR = cv.imread("D:\Pai_work\pic_sonar\stereoTest\TestR3.jpg")
    # imgL = cv.resize(imgL, (0,0), fx=0.25, fy=0.25)
    # imgR = cv.resize(imgR, (0,0), fx=0.25, fy=0.25)

    ### disparity range is tuned for 'aloe' image pair ###
    if False:
        window_size = 3
        min_disp = 16
        num_disp = 112-min_disp
        stereo = cv.StereoSGBM_create(minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = 16,
            P1 = 8*3*window_size**2,
            P2 = 32*3*window_size**2,
            disp12MaxDiff = 1,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 32
        )
        print('computing disparity...')
        disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    ### find disparity range for sonar image ###
    if True:
        window_size = 3
        min_disp = 16
        # num_disp = 112-min_disp
        num_disp = 112 + 32
        stereo = cv.StereoSGBM_create(minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = 51,
            P1 = 8*3*window_size**2,
            P2 = 32*3*window_size**2,
            disp12MaxDiff = 1,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 32
        )

        print('computing disparity...')
        disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    ### generating 3d point cloud ###
    if True:
        print('generating 3d point cloud...',)
        h, w = imgL.shape[:2]
        f = 0.8*w                          # guess for focal length
        Q = np.float32([[1, 0, 0, -0.5*w],
                        [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                        [0, 0, 0,     -f], # so that y-axis looks up
                        [0, 0, 1,      0]])
        points = cv.reprojectImageTo3D(disp, Q)
        colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
        mask = disp > disp.min()
        out_points = points[mask]
        out_colors = colors[mask]
        out_fn = 'out.ply'
        write_ply('out.ply', out_points, out_colors)
        print('%s saved' % 'out.ply')

    cv.imshow('left', imgL)
    cv.imshow('right', imgR)
    cv.imshow('disparity', (disp-min_disp)/num_disp)
    cv.waitKey()
    cv.destroyAllWindows()
