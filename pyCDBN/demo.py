import utils.load_data as ul
from cdbn.cdbn2D import *
from cdbn.crbm2D import *
from cdbn.layer import Layer
from cdbn.model import Model

from pylab import *


(train_data, trainL, test_data, testL) = ul.load_data_mat('./data/mnistSmall.mat')

def demo_loaded_data():
	imshow(train_data[:, :, 0, 7], cmap = 'Greys')
	show()

def demo_preprocessing():
	# This should require no preprocessing
	layer = Layer(train_data)
	preprocess_train_data2D(layer, True)

	# This should require the deletion of part of the inputdata
	layer = Layer(train_data, s_filter = [8, 8])
	preprocess_train_data2D(layer, True)

	print(layer.inputdata.shape)

	# I can only think of testing the whitening when I start looking at the
	# 'Gaussian' CDBNs

def demo_batch():
	layer = Layer(train_data)
	layer.s_inputdata = np.array([layer.inputdata.shape[0],
					layer.inputdata.shape[1]],
					dtype = np.float32)
	model = Model(layer)
	(batch_data, numbatches) = create_batch_data(layer, model)
	print("len(batch_data): {}".format(len(batch_data)))
	return batch_data, numbatches

layer = Layer(train_data)
crbm2D(layer)

