import pyximport
pyximport.install()
import cython
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import train_ml_prange

from mnist import MNIST
mndata = MNIST('/n/home04/cs205u1716/Proj/data')
images, labels = mndata.load_training()

#Build feature map
N_array = np.array([1000,5000,10000,20000]) #How many images I want to load
threads_array = np.array([1,2,5,10,20])
d = 784 #Pixels of MNIST data

for ti in np.arange(len(threads_array)):
	nthreads = threads_array[ti]
	print(' ')
	print('Threads: ', nthreads)
	for ni in np.arange(len(N_array)):

		N = N_array[ni]

		start = time.time()

		#Retrieve data and labels - do preprocessing
		y_labs = labels[0:N]

		#Loop over set of regularization parameters
		vaccs = []
		lambdas = np.array([10**q for q in np.linspace(-5,5,10)])

		#Load images
		feature_map = np.zeros((N,d))
		for i in range(N): #Just do a subset of training for now
		    feature_map[i,:] = images[i]

		#Take train test split
		sinds = range(N)
		random.shuffle(list(sinds))
		tint = int(.8*N)
		tind = sinds[0:tint]
		vind = sinds[tint:-1]

		#Get rid of bias
		fmean = feature_map.mean(axis=0)
		x_c = feature_map - fmean[np.newaxis,:]

		Xtr = x_c[0:tint]
		Xvl = x_c[tint:-1]
		y_val = np.array(y_labs[tint:-1]).astype(np.int)
		y_tr = np.array(y_labs[0:tint]).astype(np.int)

		#### CALL cython implementation ####
		vaccs,preds = train_ml_prange.train(Xtr, Xvl, y_tr, y_val, lambdas, nthreads)
		#### END cython implementation ####
		end = time.time()

		best_val = np.where(vaccs == np.max(vaccs))[0][0]
		print('Size = ', N)
		print('validation accuracy = ', vaccs[best_val])
		print('best lambda =', lambdas[best_val])
		print('elapsed time for', N, 'samples = ', end-start, 'seconds')

print('Done')
