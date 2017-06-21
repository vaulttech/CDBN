import numpy as np
import numpy.matlib as matlib

from scipy import signal
from math import exp, ceil, floor
from cdbn.model import Model

import cuCRBM

def crbm_reconstruct2D(model, layer):
	return model

def crbm_blocksum2D(model, layer):
	if (layer.type_input == 'Binary'):
		h_input = exp(model.h_input)
	else:
		h_input = exp((1.0 / (model.start_gau ** 2)) * model.h_input)

	col = h_input.shape[1]
	row = h_input.shape[0]
	N   = h_input.shape[3]
	x_stride = layer.s_pool[1]
	y_stride = layer.s_pool[0]
	block = np.zeros(h_input.shape)

	# XXX: Check carefully that this is doing the same as the original code
	for k in range(N):
		for i in range(floor(row / y_stride)):
			# XXX: These indices are probably wrong
			offset_r = list(range(i * y_stride + 1,
						(i + 1) * y_stride))

			for j in range(floor(col / x_stride)):
				offset_c = list(range(j * x_stride + 1,
							(j + 1) * x_stride))

				block_val = np.sum(h_input[offset_r,offset_c,:,k]).squeeze()
				block[offset_r,offset_c,:,k] = matlib.repmat(
							np.transpose(block_val, [1,2,0]),
							numel(offset_r),numel(offset_c));

	return block

def crbm_inference2D(model, layer, data):
	model.h_input = np.zeros(model.h_input.shape)
	N = model.h_input[3]

	for k in range(N):
		for i in range(layer.n_map_h):
			for j in range(layer.n_map_v):
				model.h_input[:,:,i,k] += signal.convolve2d(
						data[:,:,j,k],
						np.fliplr(np.flipud(model.W)),
						'valid')

			model.h_input[:,:,i,k] += model.h_bias(i)

	block = crbm_blocksum2D(model, layer)

	if (layer.type_input == 'Binary'):
		model.h_sample = exp(model.h_input) / (1 + block)
	else:
		model.h_sample = exp((1.0 / (model.start_gay ** 2)) *
						model.h_input) / (1 + block)

	return model


def calc_gradient2D_py(model, layer, data):
	model = crbm_inference2D(model, layer, data)

	model.h_sample_init = model.h_sample
	for i in range(model.n_cd):
		model = crbm_reconstruct2D(model, layer)
		model = crbm_inference2D(model, layer, model.v_sample)

	dW = np.zeros(model.W.shape)

	print("model.h_sample.shape: {}".format(model.h_sample.shape))

	for i in range(layer.n_map_h):
		for j in range(layer.n_map_v):
			# XXX: FIX THIS!
			#dW[:,:,j,i] = 
			pass

	return (model, dW)


def crbm_forward2D(model, layer, data):
	n = data.shape[3]
	x_stride = layer.s_pool[1]
	y_stride = layer.s_pool[0]
	row = model.h_sample[0]
	col = model.h_sample[1]

	output = np.zeros([floor(row/y_stride),
			floor(col/x_stride),
			layer.n_map_h,
			n])

	for k in range(n):
		batch_data = data[:,:,:,k]
		model.h_input = np.zeros([row, col, model.h_sample[2], 1])
		for i in range(layer.n_map_h):
			for j in range(layer.n_map_v):
				model.h_input[:,:,i,0] = model.h_input[:,:,i,0] + \
						signal.convolve2d(
							batch_data[:,:,j,0],
							np.fliplr(np.flipud(model.W)),
							'valid')
			model.h_input[:,:,i,0] += model.h_bias(i)

		block = crbm_blocksum2D(model, layer)
		h_sample = 1 - (1. / (1 + block))
		output[:,:,:,k] = hsample[0:y_stride:row, 0:x_stride:col, :]

	return output


def apply_gradient2D(model, layer, data, dW):
	N       = data.shape[3];
	dV_bias = np.sum(data - model.v_sample, (0,1,3))#.squeeze()
	dH_bias = np.sum((model.h_sample_init - model.h_sample), (0,1,3))#.squeeze()

	# TODO: refactor this function (it was originally like this)
	if (layer.type_input == 'Binary'):
		dW            = dW / N
		dH_bias       = dH_bias / N
		dV_bias       = dV_bias / N

		#print("model.dW.shape", model.dW.shape)
		#print("dW.shape",  dW.shape)
		model.dW      = model.momentum * model.dW + (1-model.momentum) * dW
		model.W       = model.W + layer.learning_rate * (
						model.dW - layer.lambda2 * model.W)

		penalty       = 0;
		model.dV_bias = model.momentum * model.dV_bias + \
				(1-model.momentum) * dV_bias

		print('Before: v_bias', model.v_bias)
		print('Before: layer.learning_rate', layer.learning_rate)
		print('Before: model.dV_bias', model.dV_bias)
		model.v_bias  = model.v_bias + layer.learning_rate * (
						model.dV_bias - penalty * model.v_bias)
		print('After: v_bias', model.v_bias)

		model.dH_bias = model.momentum*model.dH_bias + (1-model.momentum)*dH_bias;
		model.h_bias  = model.h_bias + layer.learning_rate * (
						model.dH_bias - penalty * model.h_bias);

		model.h_bias  = model.h_bias + \
				layer.learning_rate * layer.lambda1 * \
				(layer.sparsity - np.mean((model.h_sample_init), (0,1,3))).squeeze();

	if (layer.type_input == 'Gaussian'):
		N            = model.h_sample_init.shape[0] * \
				model.h_sample_init.shape[1] * layer.batchsize;
		dW           = (dW / N) - 0.01 * model.W;

		dh           = (np.sum(model.h_sample_init, (0,1,3)) -
				np.sum(model.h_sample, (0,1,3))) / N

		dv           = 0;

		model.winc   = model.winc * model.momentum + layer.learning_rate * dW;
		model.W      = model.winc + model.W;

		model.hinc   = model.hinc * model.momentum + layer.learning_rate * dh;
		model.h_bias = model.hinc + model.h_bias;

		model.vinc   = model.vinc * model.momentum + layer.learning_rate * dv;
		model.v_bias = model.vinc + model.v_bias;

	return model


def calc_gradient2D(model, layer, batch_data):
	batch_data = batch_data.squeeze()
	c = cuCRBM.CRBM_Data(model, layer, batch_data)

	c.gibbs_sample()

	dim_w = model.W.T.shape
	dW = c.calculate_dW(np.array(dim_w))

	model.h_sample = c.get_h_sample()
	model.h_sample_init = c.get_h_sample_init()
	model.v_sample = c.get_v_sample()

	return model, dW


def create_batch_data(layer, model):
	# This will replicate the procedure used in the original code.
	# TODO: Probably this could be much better done with sklearn
	batchsize  = layer.batchsize

	N          = layer.inputdata.shape[3]
	#numcases  = layer.inputdata.shape[3] # same as previous line O__o
	numbatches = int(ceil(N / batchsize))
	groups     = matlib.repmat(range(0,numbatches), 1, batchsize).squeeze()
	groups     = groups[0:N]
	perm       = np.random.permutation(range(N))
	groups     = groups[perm]

	batchdata  = []
	for i in range(numbatches):
		curr_batch = layer.inputdata[:, :, :, groups == i]
		batchdata.append(curr_batch)

	return batchdata, numbatches

#function model = crbm2D(layer, old_W, old_v_bias, old_h_bias)
def crbm2D(layer):
	# Following the original code
	cpu               = layer.cpu;
	layer.s_inputdata = np.array([layer.inputdata.shape[0],
				layer.inputdata.shape[1]])

	model             = Model(layer)
	dW                = np.zeros(model.dW.shape);

	(batchdata, numbatches) = create_batch_data(layer, model)

	# ---- Train the CRBM ----
	for epoch in range(layer.n_epoch):
		err = 0
		sparsity = np.zeros(numbatches)

		# tic
		for i in range(numbatches):
			batch_data = batchdata[i]

			# TODO: allow for different methods of gradient
			#	calculation (maybe using GPU)
			print("Will call calc_gradient2D()")
			(model, dW) = calc_gradient2D(model, layer, batch_data)

			model = apply_gradient2D(model, layer, batch_data, dW)

			sparsity[i] = np.mean(model.h_sample_init)

			err1        = (batch_data - model.v_sample)**2;
			print(err1)
			err         = err + err1.sum();

		if (model.start_gau > model.stop_gau):
			model.start_gau = model.start_gau*0.99;

		model.error[epoch] = err
		print('epoch {}/{}, reconstruction err {}, sparsity {}'.format(
				epoch, layer.n_epoch, err, sparsity.mean()))
		# toc

	# ---- Output the pooling layer ----
	print('Generating output for next layer...')
	# TODO: allow for different methods of output computation
	model.output = crbm_forward2D(model, layer, layer.inputdata)
	print('Finished work on this layer.')

	return model

