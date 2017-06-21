import numpy as np
import scipy.io as sio

def load_data_mat(file_name):
	"""
	For testing, this will be useful to emulate MATLAB's original behavior.
	For now, this function does exactly what `DemoCDBN_Binary_2D.m` does to
	load the dataset.
	"""
	f_data = sio.loadmat(file_name)

	# The original input format was (10000, 784). Thus, first transpose,
	# then reshape.
	train_data = f_data['trainData'].T.reshape([28, 28, 1, 10000])
	test_data  = f_data['testData'].T.reshape([28, 28, 1, 2000])

	# But now the images are all transposed (each image, separately). This
	# must be considered when giving the data to the network.

	trainL = f_data['trainLabels']
	testL  = f_data['testLabels']

	# Now, reduces the number of samples (i.e., for debugging)
	train_data = train_data[:,:,:,0:2000]
	trainL = trainL[0:2000, :]

	# No noise introduction

	return (train_data, trainL, test_data, testL)

