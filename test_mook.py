import numpy as np
import cv2
import FeatureMatch
from matplotlib import pyplot as plt
from skimage.feature import register_translation

def matchingPair(bImg1, bImg2):
    ref_Position = []
    shift_Position = []

    refPos, shiftPos = FeatureMatch.matchPosition_BF(bImg1, bImg2, "0")
    # print (refPos)
    # print ("===========")
    # print (shiftPos)
    if len(refPos) != 0:
        for m in refPos:
            ref_Position.append(m)
    if len(shiftPos) != 0:
        for n in shiftPos:
            shift_Position.append(n)
    
    input_row = []
    input_col = []
    for i in ref_Position:
        xx, yy = i
        input_row.append(xx)
        input_col.append(yy)

    output_row = []
    output_col = []
    for i in shift_Position:
        xx, yy = i
        output_row.append(xx)
        output_col.append(yy)

    return input_row, input_col, output_row, output_col

def linear_Approx(x_in, y_in, x_out, y_out):
    ## rewrite the line equation as x = Ap
    ## where A = [[1 x y]] 
    ## and p = [a0, a1, a2]
    vectorA = np.vstack([np.ones(len(x_in)), x_in, y_in]).T
    listA = np.linalg.lstsq(vectorA, x_out)[0]

    ## rewrite the line equation as y = Aq
    ## where B = [[1 x y x^2 y^2 xy]] 
    ## and q = [b0, b1, b2, b3, b4, b5]
    listB = np.linalg.lstsq(vectorA, y_out)[0]

    # ! Check error
    error_a = 0
    for i in range(0,len(x_in)):
        result = listA[0] + (listA[1]*x_in[i]) + (listA[2]*y_in[i])
        error_a = error_a + np.abs(x_out[i] - result)
        # print ("Error " + str(i) + " : " + str(np.abs(x_out[i] - result)))
    print (error_a/30.0)

    error_b = 0
    for i in range(0,len(x_in)):
        result = listB[0] + (listB[1]*x_in[i]) + (listB[2]*y_in[i])
        error_b = error_a + np.abs(y_out[i] - result)
        # print ("Error " + str(i) + " : " + str(np.abs(y_out[i] - result)))
    print (error_b/30.0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

    # print (listA)
    # print (listB)
    return listA, listB

def linearRemap(img, listA, listB, ref):
    y, x = np.mgrid[:img.shape[0],:img.shape[1]]
    # x, y =  np.mgrid[:img.shape[0],:img.shape[1]]
    # new_x = (-listA[0] + (listA[1]*x) + (listA[2]*y))
    # new_y = (-listB[0] + (listB[1]*x) + (listB[2]*y))
    new_x = 20 + x 
    new_y = 20 + y

    # print (listA)
    # print (listB)

    res = cv2.remap(img,new_x.astype('float32'),new_y.astype('float32'),cv2.INTER_LINEAR)
    # res = cv2.remap(img,new_y.astype('float32'),new_x.astype('float32'),cv2.INTER_LINEAR)

    # cv2.imshow("res", res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.subplot(121)
    plt.imshow(ref)
    plt.subplot(122)
    plt.imshow(res)
    plt.show()

ref = cv2.imread("C:\\Users\HP\Pictures\sweat16.jpg", 0)

print (ref.shape)

img1 = ref[500:700, 500:700]
img2 = ref[520:720, 520:720]

# plt.imshow(ref)
# plt.show()

plt.subplot(121)
plt.imshow(img1)
plt.subplot(122)
plt.imshow(img2)
plt.show()

shift, error, diffphase = register_translation(img1, img2)
print (shift)
row, col = img2.shape
trans_matrix = np.float32([[1,0,shift[1]],[0,1,shift[0]]])
floating1 = cv2.warpAffine(img2.astype(np.uint8), trans_matrix, (col,row))

plt.subplot(131)
plt.imshow(img1)
plt.subplot(132)
plt.imshow(img2)
plt.subplot(133)
plt.imshow(floating1)
plt.show()

ref_row, ref_col, in_row, in_col = matchingPair(img1, img2)
paramX, paramY = linear_Approx(in_row, in_col, ref_row, ref_col)
# paramX, paramY = linear_Approx(xOut, yOut, xPos, yPos)
print (paramX)
print (paramY)
linearRemap(img2, paramX, paramY, img1)

# xx, yy = FeatureMatch.matchPosition_BF(img1, img2, "0")

# print (xx)

trans_matrix = np.float32([[paramX[1],paramX[2],paramX[0]],[paramY[1],paramY[2],paramY[0]]])
floating2 = cv2.warpAffine(img2.astype(np.uint8), trans_matrix, (col,row))
plt.subplot(121)
plt.imshow(floating1)
plt.subplot(122)
plt.imshow(floating2)
plt.show()