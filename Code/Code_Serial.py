import numpy as np
import matplotlib.pyplot as plt
import time
import random

from mnist import MNIST
mndata = MNIST('/n/home04/cs205u1716/Proj/data')
images, labels = mndata.load_training()

#Build feature map
N_array = np.array([1000,5000,10000,20000]) #How many images I want to load
d = 784 #Pixels of MNIST data

def bayes_rule(x):
	if x > 0:
		return 1
	else:
		return -1
    
#label_func = lambda x,choose_label: [1 if la == choose_label else -1 for la in x]
def label_func(x, choose_label):
	if x == choose_label:
		return 1
	else:
		return -1

for ni in np.arange(len(N_array)):

	N = N_array[ni]

	start = time.time()

	#Retrieve data and labels - do preprocessing
	y_labs = labels[0:N]

	#Loop over set of regularization parameters
	vaccs = []
	lambdas = [10**q for q in np.linspace(-5,5,10)]

	#Load images
	feature_map = np.zeros((N,d))
	for i in range(N): #Just do a subset of training for now
	    feature_map[i,:] = images[i]

	#Start spark instance on points
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
	#y_val = [yy for idx,yy in enumerate(y_labs) if idx in vind]
	#y_tr = [yy for idx,yy in enumerate(y_labs) if idx in tind]
	y_val = y_labs[tint:-1]
	y_tr = y_labs[0:tint]

	for ll in lambdas:

		ws = []
		iouts = []
		classes = []

		#Loop over all labels
		for choose_label in range(10): 

			y_tr_map = [label_func(q, choose_label) for q in y_tr]
			Nt = Xtr.shape[0]
			Mt = Xtr.shape[1]


			#Translate np.dots to for loops
			#Can parallelize with pranges in Cython
			numer_sum = np.zeros(Mt)
			for i in range(Nt):
				x_iT = Xtr[i,:]
				inumer = x_iT * y_tr_map[i]
				numer_sum += inumer

			denom_sum = Nt*ll
			for i in range(Nt):
				x_iT = Xtr[i,:]
				idenom = 0
				for j in range(Mt):
					idenom += x_iT[j] * x_iT[j]
				denom_sum += idenom

			iw = numer_sum / float(denom_sum)
			iout = np.dot(Xvl, iw)
			iclass = [np.sign(q) for q in iout]


			#Append to output
			ws.append(iw)
			iouts.append(iout)
			classes.append(iclass)

		#Figure out how to spark-ify this loop
		out_pred = list(zip(*iouts))

		preds = []
		for idx in range(len(out_pred)):
			ipreds = np.asarray(out_pred[idx])
			iclass = np.where(ipreds == np.max(ipreds))[0][0] 
			preds.append(iclass)

		#Determine accuracy on validation
		vacc = np.sum([y == p for y,p in zip(y_val, preds)]) / float(len(preds))

		#Append to lambda
		vaccs.append(vacc)

	end = time.time()

	best_val = np.where(vaccs == np.max(vaccs))[0][0]
	print('Size = ', N)
	print('validation accuracy = ', vaccs[best_val])
	print('best lambda =', lambdas[best_val])
	print('elapsed time for', N, 'samples = ', end-start, 'seconds')
