
# coding: utf-8

# In[54]:


import numpy as np
import random

obj_pose = [random.randint(0,100),random.randint(0,100),random.randint(0,100)]
pose_1 = [random.randint(0,100),random.randint(0,100),random.randint(0,100)]
pose_2 = [random.randint(0,100),random.randint(0,100),random.randint(0,100)]
pose_3 = [random.randint(0,100),random.randint(0,100),random.randint(0,100)]
pose_4 = [random.randint(0,100),random.randint(0,100),random.randint(0,100)]

print (obj_pose)
print (pose_1)
print (pose_2)
print (pose_3)
print (pose_4)


# In[55]:


diff_row = obj_pose[0] - pose_1[0]
diff_col = obj_pose[1] - pose_1[1]
diff_z = obj_pose[2] - pose_1[2]

d1 = np.power(diff_row,2) + np.power(diff_col,2) + np.power(diff_z,2)

print (d1)


# In[56]:


diff_row = obj_pose[0] - pose_2[0]
diff_col = obj_pose[1] - pose_2[1]
diff_z = obj_pose[2] - pose_2[2]

d2 = np.power(diff_row,2) + np.power(diff_col,2) + np.power(diff_z,2)

print (d2)


# In[57]:


diff_row = obj_pose[0] - pose_3[0]
diff_col = obj_pose[1] - pose_3[1]
diff_z = obj_pose[2] - pose_3[2]

d3 = np.power(diff_row,2) + np.power(diff_col,2) + np.power(diff_z,2)

print (d3)


# In[58]:


diff_row = obj_pose[0] - pose_4[0]
diff_col = obj_pose[1] - pose_4[1]
diff_z = obj_pose[2] - pose_4[2]

d4 = np.power(diff_row,2) + np.power(diff_col,2) + np.power(diff_z,2)

print (d4)


# In[59]:


dis1 = d1 - d4
dis1 = dis1 - np.power(pose_1[0],2) - np.power(pose_1[1],2) - np.power(pose_1[2],2)
dis1 = dis1 + np.power(pose_4[0],2) + np.power(pose_4[1],2) + np.power(pose_4[2],2)


dis2 = d2 - d4
dis2 = dis2 - np.power(pose_2[0],2) - np.power(pose_2[1],2) - np.power(pose_2[2],2)
dis2 = dis2 + np.power(pose_4[0],2) + np.power(pose_4[1],2) + np.power(pose_4[2],2)

dis3 = d3 - d4 
dis3 = dis3 - np.power(pose_3[0],2) - np.power(pose_3[1],2) - np.power(pose_3[2],2)
dis3 = dis3 + np.power(pose_4[0],2) + np.power(pose_4[1],2) + np.power(pose_4[2],2)

matrix_B = np.array([dis1, dis2, dis3])

print (matrix_B)


# In[60]:


rowA_1 = [2*(pose_4[0] - pose_1[0]), 2*(pose_4[1] - pose_1[1]), 2*(pose_4[2] - pose_1[2])]
rowA_2 = [2*(pose_4[0] - pose_2[0]), 2*(pose_4[1] - pose_2[1]), 2*(pose_4[2] - pose_2[2])]
rowA_3 = [2*(pose_4[0] - pose_3[0]), 2*(pose_4[1] - pose_3[1]), 2*(pose_4[2] - pose_3[2])]

matrix_A = np.array([rowA_1, rowA_2, rowA_3])

print (matrix_A)


# In[61]:


result = np.linalg.solve(matrix_A, matrix_B)

print (result[0], result[1], result[2])
print (obj_pose)


# In[ ]:


# Two Equation

