import numpy as np

# Used to suppress warnings on the usage of floats
from math import floor

class Model:
	""" A "data" class to carry the CRBM learnt parameters. """
	def __init__(self, layer):
		self.n_cd        = layer.n_cd
		self.momentum    = layer.momentum
		self.start_gau   = layer.start_gau
		self.stop_gau    = layer.stop_gau

		self.beginAnneal = float('inf')

		self.W = 0.01 * np.random.rand(layer.s_filter[0],
					layer.s_filter[1],
					layer.n_map_v,
					layer.n_map_h)

		self.v_bias  = np.zeros(layer.n_map_v)
		self.h_bias  = np.zeros(layer.n_map_h)

		self.dW      = np.zeros(self.W.shape)
		self.dV_bias = np.zeros(layer.n_map_v)
		self.dH_bias = np.zeros(layer.n_map_h)

		self.v_size  = [layer.s_inputdata[0], layer.s_inputdata[1]]
		self.v_input = np.zeros([layer.s_inputdata[0],
					layer.s_inputdata[1],
					layer.n_map_v,
					layer.batchsize])

		# TODO: to allow for `valid` convolutions, need to change this
		self.h_size  = 1 + ((layer.s_inputdata - layer.s_filter) /
							(layer.stride))

		self.h_input = np.zeros([floor(self.h_size[0]),
					floor(self.h_size[1]),
					layer.n_map_h,
					layer.batchsize])
		self.error   = np.zeros(layer.n_epoch)

		# Define the model output
		# NEED TO FIX THE STRIDE HERE
		H_out = ((layer.inputdata.shape[0] - layer.s_filter[0]) /
					layer.stride[0] + 1) / layer.s_pool[0];
		W_out = ((layer.inputdata.shape[1] - layer.s_filter[1]) /
					layer.stride[1] + 1) / layer.s_pool[1];
		self.output = np.zeros([floor(H_out),
					floor(W_out),
					layer.n_map_h,
					layer.inputdata.shape[3]]);

		# Not sure if these are ever used in the original code
		# ADD SOME OTHER PARAMETERS FOR TEST
		self.winc    = 0;
		self.hinc    = 0;
		self.vinc    = 0;

