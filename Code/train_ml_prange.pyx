#cython: boundscheck=False, wraparound=False, nonecheck=False

import cython
from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cross_validation import train_test_split
import random


from mnist import MNIST

DTYPE64 = np.float64
DTYPE32 = np.int
ctypedef np.int_t DTYPE32_t
ctypedef np.float64_t DTYPE64_t


def bayes_rule(int x):
	if x > 0:
		return 1
	else:
		return -1

def label_func(x, choose_label):
	if x == choose_label:
		return 1
	else:
		return -1

def train(np.ndarray[np.float64_t, ndim=2] train_data, 
	      np.ndarray[np.float64_t, ndim=2] test_data, 
		  np.ndarray[np.int64_t, ndim=1] train_labels, 
		  np.ndarray[np.int64_t, ndim=1] test_labels, 
		  np.ndarray[np.float64_t, ndim=1] lambdas,
		  int nthreads):
	"""
	Train on MNIST data using cython

	Inputs:
	:type train_data: numpy.ndarray (of numpy.float64)
	:param train_data: MNIST training images (N x 784 matrix)
	:type test_data: numpy.ndarray (of numpy.float64)
	:param test_data: MNIST testing images (N x 784 matrix)
	:type train_labels: list (of ints)
	:param train_labels: digit (0-9) for each training image
	:type test_labels: list (of ints)
	:param test_labels: digit (0-9) for each testing image
	:type lambdas: numpy.ndarray (of numpy.float64)
	:param lambdas: regularization parameters

	"""

	cdef int reg, choose_label, q, i, j, idx, p, N = 10
	cdef int Nt = train_data.shape[0],Mt = train_data.shape[1]
	cdef int Ntest = test_data.shape[0],Mtest = test_data.shape[1]
	cdef float denom_sum, idenom, vacc, ll
	cdef np.int64_t imost
	cdef np.ndarray[np.int64_t, ndim=1] y_tr_map = np.empty(Nt, dtype=DTYPE32)
	cdef np.ndarray[np.float64_t, ndim=1] numer_sum = np.empty(Mt,dtype=DTYPE64)
	cdef np.ndarray[np.float64_t, ndim=1] x_iT = np.empty_like(numer_sum)
	cdef np.ndarray[np.float64_t, ndim=1] inumer = np.empty_like(numer_sum)
	cdef np.ndarray[np.float64_t, ndim=1] iw = np.empty_like(numer_sum)
	cdef np.ndarray[np.float64_t, ndim=1] iout = np.empty(test_data.shape[0],dtype=DTYPE64)
	cdef np.ndarray[np.float64_t, ndim=1] iclass = np.empty_like(iout)
	cdef np.ndarray[np.float64_t, ndim=2] ws = np.empty([N,Mt], dtype=DTYPE64)
	cdef np.ndarray[np.float64_t, ndim=2] iouts = np.empty([N,test_data.shape[0]],
														 dtype=DTYPE64)
	cdef np.ndarray[np.float64_t, ndim=2] iclasses = np.empty([N,test_data.shape[0]],
														 dtype=DTYPE64)
	cdef np.ndarray[np.float64_t, ndim=1] ipred = np.empty_like(iout)
	cdef np.ndarray[np.float64_t, ndim=1] preds = np.empty(test_data.shape[0],dtype=DTYPE64)
	cdef np.ndarray[np.float64_t, ndim=1] vaccs = np.zeros(len(lambdas),dtype=DTYPE64)

	for reg in range(len(lambdas)):
		ll = lambdas[reg]

		### Loop over all labels
		for choose_label in range(N): 
			for i in range(Nt):
				y_tr_map[i] = label_func(train_labels[i],choose_label)

			#Can parallelize with pranges in Cython
			numer_sum = np.zeros(Mt,dtype=DTYPE64)
			for i in prange(Nt,nogil=True,schedule='static',num_threads=nthreads):
			# for i in range(Nt):	
				for j in range(Mt):
					# x_iT[j] = train_data[i,j] 
					# inumer[j] = x_iT[j] * y_tr_map[i]
					numer_sum[j] += train_data[i,j] * y_tr_map[i]
				
			denom_sum = Nt*ll
			# for i in range(Nt):
			for i in prange(Nt,nogil=True,schedule='static',num_threads=nthreads):
				idenom = 0
				for j in range(Mt):
					idenom += train_data[i,j] * train_data[i,j]
				denom_sum += idenom

			iw = numer_sum / float(denom_sum)


			# for i in range(test_data.shape[0]):
			for i in prange(Ntest,nogil=True,schedule='static',num_threads=nthreads):
				for j in range(Mtest):
					iout[i] += test_data[i,j] * iw[j] 

			for i in range(len(iout)):
				iclass[i] = np.sign(iout[i])


			#Append to output
			ws[choose_label,:] = iw
			iouts[choose_label,:] = iout
			iclasses[choose_label,:] = iclass
	 
		preds = np.zeros(Ntest,dtype=DTYPE64)
		for idx in range(Ntest):
			ipreds = iouts[:,idx]
			imost = np.where(ipreds == np.max(ipreds))[0][0] 
			preds[idx] = imost

		#Determine accuracy on validation
		vacc = 0.
		for i in range(len(test_labels)):
			if test_labels[i] == preds[i]:
				vacc += 1
		vacc /= float(len(preds))
		vaccs[reg] = vacc
	return vaccs, preds
