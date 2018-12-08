import cv2
import numpy as np
import random
import csv

import sonarGeometry
import regisProcessing as rp
import FeatureMatch
from matplotlib import pyplot as plt

class positioning:
    def __init__(self, inx_1, inx_2, inx_3):
        self.image_perSec = 3.2
        self.imu_perSec = 99.2
        self.dvl_perSec = 6.6

        # ! 30/660
        # self.range_perPixel = 0.04545 
        # ! 28/660
        self.range_perPixel = 0.04343
        self.degree_perCol = 0.169

        self.triple_Row = []
        self.triple_Col = []
        self.matching_pose = []

        self.auv_state = []
        self.result_pose = []

        self.image_1 = rp.ColorMultiLook(int(self.image_perSec * inx_1), 5)
        self.image_2 = rp.ColorMultiLook(int(self.image_perSec * inx_2), 5)
        self.image_3 = rp.ColorMultiLook(int(self.image_perSec * inx_3), 5)

        self.Full_matching()

        self.read_dvl(inx_1)
        self.read_dvl(inx_2)
        self.read_dvl(inx_3)
        self.auv_position()
        
        self.distance_auv()
        # self.distance()
        
        for i in range(0,4):
            self.solve(i)

        # self.update_map()
        self.genarate_map(660,768)

    def Full_matching(self):
        savename = "D:\Pai_work\pic_sonar\calPosition\Full_Matching1.jpg"
        refPos1, shiftPos1 = FeatureMatch.matchPosition_BF(self.image_1, self.image_2, savename)
        savename = "D:\Pai_work\pic_sonar\calPosition\Full_Matching2.jpg"
        refPos2, shiftPos2 = FeatureMatch.matchPosition_BF(self.image_2, self.image_3, savename)

        inx_1 = 0
        for x,y in shiftPos1:
            inx_2 = 0
            for xx,yy in refPos2:
                if x == xx and y == yy:
                    self.triple_Row.append((refPos1[inx_1][0], xx, shiftPos2[inx_2][0]))
                    self.triple_Col.append((refPos1[inx_1][1], yy, shiftPos2[inx_2][1]))
                    break
                else:
                    inx_2 = inx_2 + 1
            inx_1 = inx_1 + 1
        # print (self.triple_Row, self.triple_Col)

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

    def solve(self, numPoint):
        # d1 = np.power(self.distanceList[numPoint][1][0], 2)
        # d2 = np.power(self.distanceList[numPoint][1][1], 2)
        # d3 = np.power(self.distanceList[numPoint][1][2], 2)

        d1 = np.power(self.auv_disList[numPoint][1][0], 2)
        d2 = np.power(self.auv_disList[numPoint][1][1], 2)
        d3 = np.power(self.auv_disList[numPoint][1][2], 2)

        matrix_B = np.array([d1 - d2, d1 - d3, d2 - d3])

        ### init Z (don't measure now)
        z1 = 2.0
        z2 = 2.5
        z3 = 2.3

        rowA_1 = [(-2)*(self.auv_pose[0][0] - self.auv_pose[1][0]), (-2)*(self.auv_pose[0][1] - self.auv_pose[1][1]), (-2)*(z1 - z2)]
        rowA_2 = [(-2)*(self.auv_pose[0][0] - self.auv_pose[2][0]), (-2)*(self.auv_pose[0][1] - self.auv_pose[2][1]), (-2)*(z1 - z3)]
        rowA_3 = [(-2)*(self.auv_pose[1][0] - self.auv_pose[2][0]), (-2)*(self.auv_pose[1][1] - self.auv_pose[2][1]), (-2)*(z1 - z3)]

        matrix_A = np.array([rowA_1, rowA_2, rowA_3])

        result = np.linalg.solve(matrix_A, matrix_B)

        # print (matrix_A)
        # print (matrix_B)
        print (result[0], result[1], result[2])
        
        self.result_pose.append((np.abs(int(result[0])), np.abs(int(result[1])), np.abs(int(result[2]))))

    def distance(self):
        auv_rowPos = 660
        auv_colPos = 384 

        self.distanceList = []
        for i in range(len(self.triple_Row)):
            disList = []
            for j in range(0,3):
                diff_row = self.triple_Row[i][j] - auv_rowPos
                diff_col = self.triple_Col[i][j] - auv_colPos
                diff_row = np.power(diff_row, 2)
                diff_col = np.power(diff_col, 2)
                dis = diff_row + diff_col
                dis = np.sqrt(dis)
                disList.append(dis)
            self.distanceList.append(("Point : " + str(i),disList))
        
    def distance_auv(self):
        self.auv_disList = []
        for i in range(len(self.triple_Row)):
            disList = []
            for j in range(0,3):
                diff_row = self.triple_Row[i][j] - self.auv_pose[j][0]
                diff_col = self.triple_Col[i][j] - self.auv_pose[j][1]
                diff_row = np.power(diff_row, 2)
                diff_col = np.power(diff_col, 2)
                dis = diff_row + diff_col
                dis = np.sqrt(dis)
                disList.append(dis)
            self.auv_disList.append(("Point : " + str(i),disList))

    def genarate_map(self, row_map, col_map):
        self.map_auv = np.zeros((row_map, col_map),np.uint8)
        self.map_obj = np.zeros((row_map, col_map),np.uint8)
        self.map_dis = np.zeros((row_map, col_map),np.uint8)

        for auv in self.auv_pose:
            cv2.circle(self.map_auv, (int(auv[1]),int(auv[0])), 10, 255, -1)

        for i in range(len(self.triple_Row)):
            center_1 = (int(self.triple_Row[i][0]), int(self.triple_Col[i][0]))
            center_2 = (int(self.triple_Row[i][1]), int(self.triple_Col[i][1]))
            center_3 = (int(self.triple_Row[i][2]), int(self.triple_Col[i][2]))
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
        for i in range(len(self.triple_Row)):
            start_1 = (int(self.triple_Row[i][0]), int(self.triple_Col[i][0]))
            start_2 = (int(self.triple_Row[i][1]), int(self.triple_Col[i][1]))
            start_3 = (int(self.triple_Row[i][2]), int(self.triple_Col[i][2]))

            cv2.line(self.map_dis, start_1, end_1, 255, 1)
            cv2.line(self.map_dis, start_2, end_2, 255, 1)
            cv2.line(self.map_dis, start_3, end_3, 255, 1)
            
    
    def draw_circle(self):
        # img_test = cv2.imread("D:\Pai_work\pic_sonar\RTheta_img_16.jpg")        
        for i in range(len(self.triple_Row)):
            rand_color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            center_1 = (int(self.triple_Row[i][0]), int(self.triple_Col[i][0]))
            center_2 = (int(self.triple_Row[i][1]), int(self.triple_Col[i][1]))
            center_3 = (int(self.triple_Row[i][2]), int(self.triple_Col[i][2]))
            cv2.circle(self.image_1, center_1, 10, rand_color, -1)
            cv2.circle(self.image_2, center_2, 10, rand_color, -1)
            cv2.circle(self.image_3, center_3, 10, rand_color, -1)

    def update_map(self):
        print (self.result_pose)
        inx = 1
        for pose in self.result_pose:
            center = (pose[1], pose[0])
            cv2.circle(self.map_dis, center, 10, 60 * inx, -1)
            inx = inx + 1

if __name__ == '__main__':

    time_index1 = 5
    time_index2 = 10
    time_index3 = 15

    position = positioning(time_index1, time_index2, time_index3)
