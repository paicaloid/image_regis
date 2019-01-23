from scipy import misc
import numpy as np
import random
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

im = misc.imread('lena512.bmp')
im = np.asarray(im)
im = where(im<100,-1,1)
noise = np.zeros((im.shape))
noise = np.random.rand(512,512)
noisy = where(noise<0.1,-1,1)
noisy_im = im*noisy

imgplot = plt.imshow(im, cmap = cm.Greys_r)
plt.show()

noise = np.zeros((im.shape))
noise = np.random.rand(512,512)
noisy = where(noise<0.1,-1,1)
noisy_im = im*noisy
noisy_im_old = noisy_im

h = 1
beta = 1.0
eta = 2.1

# ICM
for br in range(0,50): # aribtrary, could also be some error function instead
	for i in range(0,512):
		for y in range(0,512):
			# the three factors as per Bishop, PRML, Chapter 8
			factor_three = 0 # the cliques that represent pairs between latent variables (neighbours)
			factor_two = 0 # the energy function for the cliques between latent and observed values
			factor_one = 0 # bias
			x_i = 1
			y_i = noisy_im_old[i][y]
			# noisy_im[][] are the x_j neighbours as per the textbook
			if i-1 >= 0:
				factor_three += noisy_im[i-1][y]*x_i
			if i+1 < 512:
				factor_three += noisy_im[i+1][y]*x_i
			if y-1 >= 0:
				factor_three += noisy_im[i][y-1]*x_i
			if y+1 < 512:
				factor_three += noisy_im[i][y+1]*x_i
			factor_two = x_i*y_i
			factor_one = x_i
			one = h*factor_one - beta*factor_three - eta*factor_two
			factor_one = 0
			factor_three = 0
			factor_two = 0
			x_i = -1
			if i-1 >= 0:
				factor_three += noisy_im[i-1][y]*x_i
			if i+1 < 512:
				factor_three += noisy_im[i+1][y]*x_i
			if y-1 >= 0:
				factor_three += noisy_im[i][y-1]*x_i
			if y+1 < 512:
				factor_three += noisy_im[i][y+1]*x_i
			factor_two = x_i*y_i
			factor_one = x_i
			minus_one =  h*factor_one - beta*factor_three - eta*factor_two
			if one < minus_one:
				noisy_im[i][y] = 1
			else:
				noisy_im[i][y] = -1
	noisy_im_old = noisy_im

imgplot = plt.imshow(noisy_im_old, cmap = cm.Greys_r)
plt.show()