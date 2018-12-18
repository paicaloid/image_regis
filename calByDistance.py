import cv2
import numpy as np
import random
import csv

import sonarGeometry
import regisProcessing as rp
import FeatureMatch
from matplotlib import pyplot as plt

def multiplyImage(img1, img2):
    # ! Change dtype to avoid overflow (unit8 -> int32)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = img1.astype(np.int32)
    img2 = img2.astype(np.int32)
    out = cv2.multiply(img1, img2)
    out = out/255.0
    out = out.astype(np.uint8)
    return out

def show_plot(img1, img2, img3, inx):
    save_name = "D:\Pai_work\pic_sonar\calPosition\plot_triple_match_" + str(inx) + ".jpg"
    plt.subplot(221),plt.imshow(img1, cmap = 'gray')
    plt.title('img1'), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(img2, cmap = 'gray')
    plt.title('img2'), plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(img3, cmap = 'gray')
    plt.title('img3'), plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.savefig(save_name)

class block_image:
    def __init__(self, img1, img2, img3):
        self.blockImg1 = []
        self.blockImg2 = []
        self.blockImg3 = []
        self.row_match = []
        self.col_match = []

        self.create_block(img1, img2, img3, 5, 6)
        # self.display(1)
        # self.display(2)
        # self.display(3)
        self.matching(self.blockImg1, self.blockImg2, self.blockImg3, 5, 6)
        print (self.row_match)
        print (self.col_match)
        self.draw_circle(img1, img2, img3)

    def create_block(self, img1, img2, img3, row, col):
        for i in range(0,row):
            imgList1 = []
            imgList2 = []
            imgList3 = []
            for j in range(0,col):
                block_img1 = img1[i*100:(i+1)*100, j*128:(j+1)*128]
                imgList1.append(block_img1)

                block_img2 = img2[i*100:(i+1)*100, j*128:(j+1)*128]
                imgList2.append(block_img2)

                block_img3 = img3[i*100:(i+1)*100, j*128:(j+1)*128]
                imgList3.append(block_img3)
            self.blockImg1.append(imgList1)
            self.blockImg2.append(imgList2)
            self.blockImg3.append(imgList3)
    
    def display(self, imgNum):
        inx = 1
        if imgNum == 1:
            for rowBlock in self.blockImg1:
                for block in rowBlock:
                    plt.subplot(5,6,inx),plt.imshow(block, cmap = 'gray')
                    plt.title("cfarBlock #" + str(inx)), plt.xticks([]), plt.yticks([])
                    inx += 1
            plt.show()
        elif imgNum == 2:
            for rowBlock in self.blockImg2:
                for block in rowBlock:
                    plt.subplot(5,6,inx),plt.imshow(block, cmap = 'gray')
                    plt.title("cfarBlock #" + str(inx)), plt.xticks([]), plt.yticks([])
                    inx += 1
            plt.show()
        elif imgNum == 3:
            for rowBlock in self.blockImg3:
                for block in rowBlock:
                    plt.subplot(5,6,inx),plt.imshow(block, cmap = 'gray')
                    plt.title("cfarBlock #" + str(inx)), plt.xticks([]), plt.yticks([])
                    inx += 1
            plt.show()
    
    def matching(self, block_img1, block_img2, block_img3, row_len, col_len):
        savename = "0" # ! Don't save plot
        inx = 1
        for row in range(row_len):
            for col in range(col_len):
                refPos1, shiftPos1 = FeatureMatch.matchPosition_BF(block_img1[row][col], block_img2[row][col], savename)
                refPos2, shiftPos2 = FeatureMatch.matchPosition_BF(block_img2[row][col], block_img3[row][col], savename)

                row_match = []
                col_match = []
                inx_1 = 0
                for x,y in shiftPos1:
                    inx_2 = 0
                    for xx,yy in refPos2:
                        if x == xx and y == yy:
                            row_match.append((refPos1[inx_1][0] + (row*100), xx + (row*100), shiftPos2[inx_2][0] + (row*100)))
                            col_match.append((refPos1[inx_1][1] + (col*128), yy + (col*128), shiftPos2[inx_2][1] + (col*128)))
                            self.row_match.append((refPos1[inx_1][0] + (row*100), xx + (row*100), shiftPos2[inx_2][0] + (row*100)))
                            self.col_match.append((refPos1[inx_1][1] + (col*128), yy + (col*128), shiftPos2[inx_2][1] + (col*128)))
                            break
                        else:
                            inx_2 = inx_2 + 1
                    inx_1 = inx_1 + 1
                if len(row_match) != 0:
                    print ("block number : " + str(inx))
                    print (row_match)
                    print (col_match)
                    
                inx += 1

    def draw_circle(self, img1, img2, img3):
        # img_test = cv2.imread("D:\Pai_work\pic_sonar\RTheta_img_16.jpg")        
        for i in range(len(self.row_match)):
            rand_color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            center_1 = (int(self.row_match[i][0]), int(self.col_match[i][0]))
            center_2 = (int(self.row_match[i][1]), int(self.col_match[i][1]))
            center_3 = (int(self.row_match[i][2]), int(self.col_match[i][2]))
            cv2.circle(img1, center_1, 10, rand_color, -1)
            cv2.circle(img2, center_2, 10, rand_color, -1)
            cv2.circle(img3, center_3, 10, rand_color, -1)

            cv2.imshow("cafar1", img1)
            cv2.imshow("cafar2", img2)
            cv2.imshow("cafar3", img3)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


class positioning:
    def __init__(self, inx_1, inx_2, inx_3, inx_4, inx_5):
        self.image_perSec = 3.2
        self.imu_perSec = 99.2
        self.dvl_perSec = 6.6

        # ! 30/660
        # self.range_perPixel = 0.04545 
        # ! 28/660
        self.range_perPixel = 0.04343
        self.degree_perCol = 0.169

        self.row_List = []
        self.col_List = []
        self.matching_pose = []

        self.auv_state = []
        self.result_pose = []

        # ! Delete after test triangulation
        self.obj_pose = [(353,436), (334,495), (298,587), (218,661), (143,704)]

        self.image = rp.ColorMultiLook(int(self.image_perSec * inx_1), 5)

        kernel = np.ones((7,7),np.uint8)
        self.cfar_img = rp.cafar(self.image, 59, 11, 0.25)
        self.cfar_img = cv2.erode(self.cfar_img,kernel,iterations = 1)
        self.cfar_img = cv2.dilate(self.cfar_img,kernel,iterations = 3)

        self.mul_img = multiplyImage(self.image, self.cfar_img)
        
        # self.Full_matching()
        # self.draw_circle()

        self.read_dvl(inx_1)
        self.read_dvl(inx_2)
        self.read_dvl(inx_3)
        self.read_dvl(inx_4)
        self.read_dvl(inx_5)
        self.auv_position()
        
        # print(self.auv_pose)

        self.distance_auv()

        # print (self.auv_disList)

        # self.solve_2()
        # self.solve()
        
        for i in range(len(self.obj_pose)):
            self.solve(i)
            # self.extra_solve(i)
        #     self.solve_2(i)
            print ("++++++++++++++++")
        # self.genarate_map(660,768)

    def Full_matching(self):
        savename = "D:\Pai_work\pic_sonar\calPosition\Full_Matching1.jpg"
        refPos1, shiftPos1 = FeatureMatch.matchPosition_BF(self.image_1, self.image_2, savename)
        savename = "D:\Pai_work\pic_sonar\calPosition\Full_Matching2.jpg"
        refPos2, shiftPos2 = FeatureMatch.matchPosition_BF(self.image_2, self.image_3, savename)

        # savename = "D:\Pai_work\pic_sonar\calPosition\cafar_Matching1.jpg"
        # refPos1, shiftPos1 = FeatureMatch.matchPosition_BF(self.mul_img1, self.mul_img2, savename)
        # savename = "D:\Pai_work\pic_sonar\calPosition\cafar_Matching2.jpg"
        # refPos2, shiftPos2 = FeatureMatch.matchPosition_BF(self.mul_img2, self.mul_img3, savename)

        inx_1 = 0
        for x,y in shiftPos1:
            inx_2 = 0
            for xx,yy in refPos2:
                if x == xx and y == yy:
                    self.row_List.append((refPos1[inx_1][0], xx, shiftPos2[inx_2][0]))
                    self.col_List.append((refPos1[inx_1][1], yy, shiftPos2[inx_2][1]))
                    break
                else:
                    inx_2 = inx_2 + 1
            inx_1 = inx_1 + 1
        print (self.row_List, self.col_List)
        print (len(self.col_List))

    def read_dvl(self, sec):
        first_check = True
        xPos = []
        yPos = []
        zPos = []
        with open('recordDVL.csv') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for data in reader:
                if first_check:
                    first_check = False
                    self.init_time = int(data[2][0:10])
                    # print (self.init_time)
                else:
                    if (self.init_time + sec == int(data[2][0:10])):
                        xPos.append(float(data[3]))
                        yPos.append(float(data[4]))
                        zPos.append(float(data[5]))
                        # print (data[2][0:10], data[3], data[4], data[5])
                    elif (self.init_time + sec < int(data[2][0:10])):
                        self.auv_state.append((self.init_time + sec, np.mean(xPos) * sec, np.mean(yPos) * sec, np.mean(zPos) * sec))
                        break
    
    def auv_position(self):
        row_init = 660
        col_init = 384
        self.auv_pose = []
        for pose in self.auv_state:
            xPos = pose[1] / self.range_perPixel
            xPos = row_init + xPos
            yPos = np.arctan(np.abs(pose[1]/pose[2])) / self.degree_perCol
            if pose[2] < 0:
                yPos = col_init - yPos
            else:
                yPos = col_init + yPos
            # print (xPos, yPos)
            self.auv_pose.append((xPos, yPos))

    def solve_2(self):
        d1 = self.auv_disList[0][1][0]
        d2 = self.auv_disList[0][1][1]
        d3 = self.auv_disList[0][1][2]

        b1 = d1 - d3 - np.power(self.auv_pose[0][0],2) - np.power(self.auv_pose[0][1],2) + np.power(self.auv_pose[2][0],2) + np.power(self.auv_pose[2][1],2)
        b2 = d2 - d3 - np.power(self.auv_pose[1][0],2) - np.power(self.auv_pose[1][1],2) + np.power(self.auv_pose[2][0],2) + np.power(self.auv_pose[2][1],2)
        matrix_B = np.array([b1, b2])

        rowA_1 = [2*(self.auv_pose[2][0]-self.auv_pose[0][0]), 2*(self.auv_pose[2][1]-self.auv_pose[0][1])]
        rowA_2 = [2*(self.auv_pose[2][0]-self.auv_pose[1][0]), 2*(self.auv_pose[2][1]-self.auv_pose[1][1])]

        matrix_A = np.array([rowA_1, rowA_2])

        result = np.linalg.solve(matrix_A, matrix_B)

        print (result[0], result[1])
        print (self.obj_pose[0])
        # # self.result_pose.append((np.abs(int(result[0])), np.abs(int(result[1])), np.abs(int(result[2]))))
        # self.result_pose.append((int(result[0]), int(result[1])))

    def extra_solve(self, num):
        z1 = 2.75
        z2 = 2.5
        z3 = 2.6
        z4 = 2.3
        z5 = 2.8

        d1 = self.auv_disList[num][1][0] + np.power(z1, 2)
        d2 = self.auv_disList[num][1][1] + np.power(z2, 2)
        d3 = self.auv_disList[num][1][2] + np.power(z3, 2)
        d4 = self.auv_disList[num][1][3] + np.power(z4, 2)
        d5 = self.auv_disList[num][1][4] + np.power(z5, 2)

        b1 = d1 - d4 - np.power(self.auv_pose[0][0],2) - np.power(self.auv_pose[0][1],2) - np.power(z1, 2) + np.power(self.auv_pose[3][0],2) + np.power(self.auv_pose[3][1],2) + np.power(z4, 2)
        b2 = d2 - d4 - np.power(self.auv_pose[1][0],2) - np.power(self.auv_pose[1][1],2) - np.power(z2, 2) + np.power(self.auv_pose[3][0],2) + np.power(self.auv_pose[3][1],2) + np.power(z4, 2)
        b3 = d3 - d4 - np.power(self.auv_pose[2][0],2) - np.power(self.auv_pose[2][1],2) - np.power(z3, 2) + np.power(self.auv_pose[3][0],2) + np.power(self.auv_pose[3][1],2) + np.power(z4, 2)
        b5 = d5 - d4 - np.power(self.auv_pose[4][0],2) - np.power(self.auv_pose[4][1],2) - np.power(z5, 2) + np.power(self.auv_pose[3][0],2) + np.power(self.auv_pose[3][1],2) + np.power(z4, 2)

        matrix_B = np.array([[b1], [b2], [b3], [b5]])

        rowA_1 = [2*(self.auv_pose[3][0]-self.auv_pose[0][0]), 2*(self.auv_pose[3][1]-self.auv_pose[0][1]), 2*(z4 - z1)]
        rowA_2 = [2*(self.auv_pose[3][0]-self.auv_pose[1][0]), 2*(self.auv_pose[3][1]-self.auv_pose[1][1]), 2*(z4 - z2)]
        rowA_3 = [2*(self.auv_pose[3][0]-self.auv_pose[2][0]), 2*(self.auv_pose[3][1]-self.auv_pose[2][1]), 2*(z4 - z3)]
        rowA_5 = [2*(self.auv_pose[3][0]-self.auv_pose[4][0]), 2*(self.auv_pose[3][1]-self.auv_pose[4][1]), 2*(z4 - z5)]

        matrix_A = np.array([rowA_1, rowA_2, rowA_3, rowA_5])

        # result = np.linalg.solve(matrix_A, matrix_B)
        # result = np.linalg.lstsq(matrix_A, matrix_B)[0]
        # print (result)

        out = np.matmul(np.transpose(matrix_A), matrix_A)
        out = np.matmul(np.linalg.inv(out), np.transpose(matrix_A))
        out = np.matmul(out, matrix_B)

        print (out)

    def solve(self, num):
        ### init Z (don't measure now)
        z1 = 200.75 / self.range_perPixel
        z2 = 200.5 / self.range_perPixel
        z3 = 200.8 / self.range_perPixel
        z4 = 200.3 / self.range_perPixel

        d1 = self.auv_disList[num][1][0] + np.power(z1, 2)
        d2 = self.auv_disList[num][1][1] + np.power(z2, 2)
        d3 = self.auv_disList[num][1][2] + np.power(z3, 2)
        d4 = self.auv_disList[num][1][3] + np.power(z4, 2)

        b1 = d1 - d4 - np.power(self.auv_pose[0][0],2) - np.power(self.auv_pose[0][1],2) - np.power(z1, 2) + np.power(self.auv_pose[3][0],2) + np.power(self.auv_pose[3][1],2) + np.power(z4, 2)
        b2 = d2 - d4 - np.power(self.auv_pose[1][0],2) - np.power(self.auv_pose[1][1],2) - np.power(z2, 2) + np.power(self.auv_pose[3][0],2) + np.power(self.auv_pose[3][1],2) + np.power(z4, 2)
        b3 = d3 - d4 - np.power(self.auv_pose[2][0],2) - np.power(self.auv_pose[2][1],2) - np.power(z3, 2) + np.power(self.auv_pose[3][0],2) + np.power(self.auv_pose[3][1],2) + np.power(z4, 2)

        matrix_B = np.array([[b1], [b2], [b3]])

        rowA_1 = [2*(self.auv_pose[3][0]-self.auv_pose[0][0]), 2*(self.auv_pose[3][1]-self.auv_pose[0][1]), 2*(z4 - z1)]
        rowA_2 = [2*(self.auv_pose[3][0]-self.auv_pose[1][0]), 2*(self.auv_pose[3][1]-self.auv_pose[1][1]), 2*(z4 - z2)]
        rowA_3 = [2*(self.auv_pose[3][0]-self.auv_pose[2][0]), 2*(self.auv_pose[3][1]-self.auv_pose[2][1]), 2*(z4 - z3)]

        matrix_A = np.array([rowA_1, rowA_2, rowA_3])

        result = np.linalg.solve(matrix_A, matrix_B)

        # print (matrix_A.shape)
        # print (matrix_B.shape)
        # print (result[0], result[1], result[2])
        
        # self.result_pose.append((np.abs(int(result[0])), np.abs(int(result[1])), np.abs(int(result[2]))))
        self.result_pose.append((int(result[0]), int(result[1]), int(result[2])))

        # out = (np.linalg.inv(np.transpose(matrix_A)*matrix_A) * np.transpose(matrix_A) * matrix_B)[0]
        out = np.matmul(np.transpose(matrix_A), matrix_A)
        out = np.matmul(np.linalg.inv(out), np.transpose(matrix_A))
        out = np.matmul(out, matrix_B)
        # out = np.matmul((np.linalg.inv(np.transpose(matrix_A)*matrix_A) ,B)
        print (out[0], out[1], out[2])
        # print (out)

    def distance(self):
        auv_rowPos = 660
        auv_colPos = 384 

        self.distanceList = []
        for i in range(len(self.row_List)):
            disList = []
            for j in range(0,3):
                diff_row = self.row_List[i][j] - auv_rowPos
                diff_col = self.col_List[i][j] - auv_colPos
                diff_row = np.power(diff_row, 2)
                diff_col = np.power(diff_col, 2)
                dis = diff_row + diff_col
                dis = np.sqrt(dis)
                disList.append(dis)
            self.distanceList.append(("Point : " + str(i),disList))
        
    def distance_auv(self):
        self.auv_disList = []
        inx = 1
        for obj in self.obj_pose:
            disList = []
            for pose in self.auv_pose:
                diff_row = obj[0] - pose[0]
                diff_col = obj[1] - pose[1]
                diff_row = np.power(diff_row, 2)
                diff_col = np.power(diff_col, 2)
                dis = diff_row + diff_col
                disList.append(dis)
            self.auv_disList.append(("object : " + str(inx),disList))
            inx +=1
    def genarate_map(self, row_map, col_map):
        self.map_auv = np.zeros((row_map, col_map),np.uint8)
        self.map_obj = np.zeros((row_map, col_map),np.uint8)
        self.map_dis = np.zeros((row_map, col_map),np.uint8)

        for auv in self.auv_pose:
            cv2.circle(self.map_auv, (int(auv[1]),int(auv[0])), 10, 255, -1)

        for i in range(len(self.row_List)):
            center_1 = (int(self.row_List[i][0]), int(self.col_List[i][0]))
            center_2 = (int(self.row_List[i][1]), int(self.col_List[i][1]))
            center_3 = (int(self.row_List[i][2]), int(self.col_List[i][2]))
            cv2.circle(self.map_obj, center_1, 5, 85, -1)
            cv2.circle(self.map_obj, center_2, 5, 170, -1)
            cv2.circle(self.map_obj, center_3, 5, 255, -1)

        # ! cv2.line(img,(0,0),(511,511),(255,0,0),5)
        self.draw_distance()
        self.update_map()
        
        self.map_display = cv2.merge((self.map_auv,self.map_obj,self.map_dis))
        cv2.imshow("map", self.map_display)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def draw_distance(self):
        end_1 = (int(self.auv_pose[0][1]), int(self.auv_pose[0][0]))
        end_2 = (int(self.auv_pose[1][1]), int(self.auv_pose[1][0]))
        end_3 = (int(self.auv_pose[2][1]), int(self.auv_pose[2][0]))
        for i in range(len(self.row_List)):
            start_1 = (int(self.row_List[i][0]), int(self.col_List[i][0]))
            start_2 = (int(self.row_List[i][1]), int(self.col_List[i][1]))
            start_3 = (int(self.row_List[i][2]), int(self.col_List[i][2]))

            cv2.line(self.map_dis, start_1, end_1, 255, 1)
            cv2.line(self.map_dis, start_2, end_2, 255, 1)
            cv2.line(self.map_dis, start_3, end_3, 255, 1)
            
    
    def draw_circle(self):
        # img_test = cv2.imread("D:\Pai_work\pic_sonar\RTheta_img_16.jpg")
        inx = 1        
        for i in range(len(self.row_List)):
            rand_color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            center_1 = (int(self.row_List[i][0]), int(self.col_List[i][0]))
            center_2 = (int(self.row_List[i][1]), int(self.col_List[i][1]))
            center_3 = (int(self.row_List[i][2]), int(self.col_List[i][2]))
            cv2.circle(self.image_1, center_1, 10, rand_color, -1)
            cv2.circle(self.image_2, center_2, 10, rand_color, -1)
            cv2.circle(self.image_3, center_3, 10, rand_color, -1)

            show_plot(self.image_1, self.image_2, self.image_3, inx)
            inx += 1

        # cv2.imshow("cafar1", self.image_1)
        # cv2.imshow("cafar2", self.image_2)
        # cv2.imshow("cafar3", self.image_3)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    def update_map(self):
        print (self.result_pose)
        inx = 1
        for pose in self.result_pose:
            # ! ++++++++++++++++++++ ! #
            if pose[0] < 0:
                row_pos = pose[0] + 660
            else:
                row_pos = pose[0]
            if pose[1] < 0:
                col_pos = pose[1] + 768
            else:
                col_pos = pose[1]
            # ! ++++++++++++++++++++ ! #
            center = (col_pos, row_pos)
            print (center)
            cv2.circle(self.map_dis, center, 10, 200, -1)
            inx = inx + 1

            # cv2.imshow("res", self.map_dis)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

if __name__ == '__main__':

    time_index1 = 5
    time_index2 = 10
    time_index3 = 15
    time_index4 = 20
    time_index5 = 25
    position = positioning(time_index1, time_index2, time_index3, time_index4, time_index5)

    # img1 = cv2.cvtColor(position.image_1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(position.cfar_img1, cv2.COLOR_BGR2GRAY)

    # res = multiplyImage(img1, img2)
    # ref = rp.AverageMultiLook(10,1)
    # img = rp.AverageMultiLook(10,10)

    cv2.imshow("img", position.cfar_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    