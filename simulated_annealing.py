import cv2
import numpy as np
import random
import csv
import math

from matplotlib import pyplot as plt
from skimage import img_as_ubyte

class special_sa:
    def __init__(self, img, ref, temp, min_temp, alpha):
        self.temp = temp
        self.min_temp = min_temp
        self.alpha = alpha
        
        self.ref = ref.astype(int)
        self.current = img.astype(int)
        self.next = img.astype(int)

        self.inx_row, self.inx_col = np.where(ref > 0)

        diff = []
        for i in range(len(self.inx_row)):
            delta = self.ref[self.inx_row[i]][self.inx_col[i]] - self.current[self.inx_row[i]][self.inx_col[i]]
            delta = math.pow(delta, 2)
            diff.append(delta)
        self.old_energy = np.sum(diff)

    def annealing(self, nb_size, move_size):
        print ("annealing...")
        for i in range(len(self.inx_row)):
            row = self.inx_row[i]
            col = self.inx_col[i]

            old = self.neighbor_energy(self.current[row][col], row, col, nb_size)
            
            move = np.random.randint(-move_size,move_size)
            new_center = self.current[row][col] + move
            if new_center < 0:
                new_center = 0
            elif new_center > 255:
                new_center = 255
            
            new = self.neighbor_energy(new_center, row, col, nb_size)

            if new <= old:
                self.next[row][col] = new_center
            else:
                self.next[row][col] = self.current[row][col]

    def neighbor_energy(self, center, row, col, nb_size):
        summ = 0
        for i in range(row-nb_size, row+nb_size+1):
            for j in range(col-nb_size, col+nb_size+1):
                if i == row and j == col:
                    pass
                else:
                    diff = math.pow(self.ref[i][j] - center,2)
                    summ = summ + diff
        return summ

    def update_state(self):
        # new_energy = np.sum(self.ref - self.next)

        diff = []
        for i in range(len(self.inx_row)):
            delta = self.ref[self.inx_row[i]][self.inx_col[i]] - self.next[self.inx_row[i]][self.inx_col[i]]
            delta = math.pow(delta, 2)
            diff.append(delta)
        new_energy = np.sum(diff)


        # print (self.old_energy, new_energy)
        if new_energy < self.old_energy:
            print ("update...")
            self.old_energy = new_energy
            self.current = self.next
        else:
            print ("Calculate accepted prob...")
            prob = new_energy - self.old_energy
            # prob = math.pow(prob, 2)
            prob = np.exp(prob/self.temp)
            print (prob)
            self.temp = self.min_temp
        
        self.temp = self.temp * self.alpha

class annealing:
    def __init__(self, map1, map2):
        self.row, self.col = map1.shape
        self.ref_map = map1
        self.current_state = map2
        self.next_state = map2

        self.old_energy = self.energy()
        self.new_energy = 0

        self.temp = 1.0
        self.min_temp = 0.00001
        self.alpha = 0.9

        self.count_increase = 0
        self.count_decrease = 0

        self.energy()

        print (self.old_energy)

    def neighbor(self, row, col):
        center = int(self.ref_map[row][col])
        zmap = self.current_state
        sum_diff = 0
        for i in range (row-1, row+2):
            for j in range (col-1, col+2):
                if i != row and j != col:
                    diff = center - int(zmap[i][j])
                    diff = math.pow(diff, 2)
                    sum_diff = sum_diff + diff
        return sum_diff

    def energy(self):
        sum_eng = 0
        for i in range (1, self.row-2):
            for j in range (1, self.col-2):
                sum_eng = sum_eng + self.neighbor(i,j)
        return sum_eng
                
    def move_to_neighbor(self, row, col):
        list_diff = []
        list_pose = []
        ref = self.ref_map
        state = self.current_state
        for i in range (row-1, row+2):
            for j in range (col-1, col+2):
                if i != row and j != col:
                    diff = np.abs(int(ref[i][j])-int(state[i][j]))
                    list_diff.append(diff)
                    list_pose.append((i,j))
        if np.max(list_diff) == 0:
            return state[i][j]
        else:
            inx = np.argmax(list_diff)
            row_pose = list_pose[inx][0]
            col_pose = list_pose[inx][1]
            return state[row_pose][col_pose]

    def update_state(self):
        for i in range (1, self.row-2):
            for j in range (1, self.col-2):
                self.next_state[i][j] = self.move_to_neighbor(i, j)


class binary_sa:
    def __init__(self, cafar):
        self.row, self.col = cafar.shape
        self.num_pix = self.row * self.col
        self.ref_state = cafar
        self.current_state = np.zeros((self.row,self.col), np.uint8)
        self.next_state = np.zeros((self.row,self.col), np.uint8)

        new = self.current_state.astype(int)
        ref = self.ref_state.astype(int)
        diff = np.power((new-ref), 2)

        self.old_energy = np.sum(diff)/self.num_pix
        self.new_energy = 0

        self.temp = 1.0
        self.min_temp = 0.00001
        self.alpha = 0.9

    def update(self, row, col):
        ref = self.ref_state
        center = self.current_state[row][col]
        diff = []
        for i in range (row-1, row+2):
            for j in range (col-1, col+2):
                if i != row and j != col:
                    diff.append(np.abs(int(ref[i][j]) - int(center)))
        old_eng = np.sum(diff)

        if center == 1:
            center = 0
        elif center == 0:
            center = 1
        else:
            print ("error")

        diff = []
        for i in range (row-1, row+2):
            for j in range (col-1, col+2):
                if i != row and j != col:
                    diff.append(np.abs(int(ref[i][j]) - int(center)))
        new_eng = np.sum(diff)

        if new_eng < old_eng:
            self.next_state[row][col] = center
        else:
            self.next_state[row][col] = self.current_state[row][col]

    def icm(self):
        # for i in range (1, self.row-2):
        #     for j in range (1, self.col-2):
        #         self.update(i,j)

        for i in range (1000):
            ran_row = np.random.randint(1,self.row-2)
            ran_col = np.random.randint(1,self.col-2)
            self.update(ran_row,ran_col)

        new = self.next_state.astype(int)
        ref = self.ref_state.astype(int)

        diff = np.power((new-ref), 2)
        self.new_energy = np.sum(diff)/self.num_pix

    def annealing(self):
        for i in range (50000):
            ran_row = np.random.randint(1,self.row-2)
            ran_col = np.random.randint(1,self.col-2)
            center = int(self.current_state[ran_row][ran_col])
            if center == 1:
                center = 0
            elif center == 0:
                center = 1
            self.next_state[ran_row][ran_col] = center
        new = self.next_state.astype(int)
        ref = self.ref_state.astype(int)
        diff = np.power((new-ref), 2)
        self.new_energy = np.sum(diff)/self.num_pix
        
    def accepted_prob(self):
        # diff_eng = np.power((new_eng - old_eng), 2)
        # prob = np.exp(diff_eng / temp) 
        # diff_eng = new_eng - old_eng
        # diff_eng = self.old_energy - self.new_energy
        diff_eng = self.new_energy - self.old_energy
        prob = np.exp(diff_eng/self.temp)
        print ("diff_eng :", diff_eng)
        # print ("temp :", self.temp)
        return prob

class icm:
    def __init__(self, z_ref, cf):
        print ("init...")
        self.row, self.col = z_ref.shape
        self.num_pix = self.row * self.col

        self.temp = 1.0
        self.min_temp = 0.00001
        self.alpha = 0.9

        self.ref = z_ref
        self.current_state = 10*np.ones((self.row,self.col), np.uint8)
        # self.current_state = cf
        self.next_state = np.ones((self.row,self.col), np.uint8)

        self.old_energy = self.energy(self.current_state)
        self.new_energy = 0
        self.loop = True
        
    def energy(self, state):
        ref = self.ref.astype(int)
        state = state.astype(int)
        diff = ref - state
        diff = np.power(diff,2)
        # diff = math.pow(diff, 2)
        diff = np.sum(diff)
        return diff

    def neighbor_energy(self, center, row, col):
        center = int(center)
        ref = self.ref.astype(int)
        diff = []
        for i in range (row-1, row+2):
            for j in range (col-1, col+2):
                diff.append(np.abs(ref[i][j] - center))
        return np.sum(diff)

    def change_value(self, center):
        center = int(center)
        rand = np.random.randint(-15,15)
        new_center = center + rand
        #! check case new_center < 0
        if new_center < 0:
            new_center = 0
        elif new_center > 255:
            new_center = 255
        return new_center
    
    def process(self):
        print ("process...")
        for i in range (1, self.row-2):
            for j in range (1, self.col-2):
                # print (i,j)
                current_eng = self.neighbor_energy(self.current_state[i][j], i, j)
                next_center = self.change_value(self.current_state[i][j])
                next_eng = self.neighbor_energy(next_center, i, j)

                if next_eng < current_eng:
                    self.next_state[i][j] = next_center
                else:
                    self.next_state[i][j] = self.current_state[i][j]
    
    def random_process(self):
        print ("random process...")
        for i in range (500):
            ran_row = np.random.randint(1,self.row-2)
            ran_col = np.random.randint(1,self.col-2)
            # print (i,j)
            current_eng = self.neighbor_energy(self.current_state[ran_row][ran_col], ran_row, ran_col)
            next_center = self.change_value(self.current_state[ran_row][ran_col])
            next_eng = self.neighbor_energy(next_center, ran_row, ran_col)

            if next_eng < current_eng:
                self.next_state[ran_row][ran_col] = next_center
            else:
                self.next_state[ran_row][ran_col] = self.current_state[ran_row][ran_col]

    def change_state(self):
        print ("change state...")
        self.new_energy = self.energy(self.next_state)
        print (self.new_energy, self.old_energy)
        if self.new_energy < self.old_energy:
            self.current_state = self.next_state
            self.old_energy = self.new_energy
            self.loop = True
        else:
            self.loop = False


