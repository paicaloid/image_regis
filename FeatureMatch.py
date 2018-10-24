import numpy as np
import cv2
from matplotlib import pyplot as plt

def matchPosition_BF(img1, img2, savename):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    ref_Position = []
    shift_Position = []
    inx = 0
    try:
        for m,n in matches:
            # print (m.distance, n.distance)
            # print (m.imgIdx, n.imgIdx)
            # print (m.queryIdx, n.queryIdx)
            # print (m.trainIdx, n.trainIdx)

            if m.distance < 0.75*n.distance:
                good.append([m])
                ref_Position.append(kp1[m.queryIdx].pt)
                shift_Position.append(kp2[m.trainIdx].pt)
                # print (kp1[inx].pt)
                # print (kp2[inx].pt)
        # print (ref_Position, shift_Position)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

        plt.imshow(img3),plt.savefig(savename)
        return ref_Position, shift_Position
    except:
        return ref_Position, shift_Position

def BF_matching(img1, img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    inx = 0
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
            # print (kp1[inx].pt)
            # print (kp2[inx].pt)
            # print (m.trainIdx, n.trainIdx)
            # print (m.queryIdx, n.queryIdx)
            # print (m.imgIdx, n.imgIdx)
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

    plt.imshow(img3),plt.show()

def BF_saveMatching(img1, img2, filename):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

    plt.imshow(img3),plt.savefig(filename)

def FLANN_matching(img1, img2):

    # Initiate SIFT detector
    # sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.08, 5)
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75*n.distance:
            matchesMask[i]=[1,0]
            print (i)
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    for c, value in enumerate(matches[0], 1):
        print(c, value)

    # print (type(matches[0][0]))
    print (matches[0][0].distance)
    print (matches[0][0].imgIdx)
    print (matches[0][0].queryIdx)
    print (matches[0][0].trainIdx)
    print (matches[0][1].distance)
    print (matches[0][1].imgIdx)
    print (matches[0][1].queryIdx)
    print (matches[0][1].trainIdx)
    print ("=================")
    print (matches[26][0].distance)
    print (matches[26][0].imgIdx)
    print (matches[26][0].queryIdx)
    print (matches[26][0].trainIdx)
    print (matches[26][1].distance)
    print (matches[26][1].imgIdx)
    print (matches[26][1].queryIdx)
    print (matches[26][1].trainIdx)
    # print (matches[11][0].distance)
    # print (matches[11][1].distance)
    # print (matches[9])
    # print (matches[9][0].distance)
    # print (matches[9][1].distance)

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    plt.imshow(img3,),plt.show()
    
def FLANN_saveMatching(img1, img2, filename):
    # Initiate SIFT detector

    ### SIFT parameter ###
    ## nfeatures : The number of best features to retain
    ## nOctaveLayers : computed automatically from the image resolution
    ## contrastThreshold : used to filter out weak features. 
    ##                     The larger the threshold, the less features are produced by the detector.
    ## edgeThreshold : used to filter out edge-like features.
    ##                 the larger the edgeThreshold, the less features are filtered out (more features are retained)
    ## sigma : The sigma of the Gaussian applied to the input image.
    ##         If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
    nfeatures = 0
    nOctaveLayers = 3
    contrastThreshold = 0.04
    edgeThreshold = 5
    sigma = 1.6
    # sift = cv2.xfeatures2d.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75*n.distance:
            matchesMask[i]=[1,0]
            print (i)
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = 0)

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    plt.imshow(img3,),plt.savefig(filename)