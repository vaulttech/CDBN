from math import floor, ceil
import numpy as np

### From `preprocess_train_data2D.m`

def crbm_whiten(im):
	# The input image has to be in gray scale.
	# Ignoring the `if` statement here from the original code.

	im -= im.mean()
	std = im.std()
	if (std != 0):
		im /= std

	# TODO: understand the black magic happening at this part of the original
	#	code.

	return im_out

def preprocess_train_data2D(layer, verbose = False):
	# For now, only supporting `valid` convolutions

	mod_1 = (1 + ((layer.inputdata.shape[0] - layer.s_filter[0]) /
					layer.stride[0])) % layer.s_pool[0]

	mod_2 = (1 + ((layer.inputdata.shape[1] - layer.s_filter[1]) /
					layer.stride[1])) % layer.s_pool[1]


	if (verbose):
		print("Image shape: {}".format(layer.inputdata.shape))
		print("s_filter: {}".format(layer.s_filter))
		print("stride: {}".format(layer.stride))
		print("s_pool: {}".format(layer.s_pool))
		print("mod_1: {}".format(mod_1))
		print("mod_2: {}".format(mod_2))

	# TODO: Put this code repetition in a separate function
	if (mod_1 != 0):
		# Remove rows of input images (half of `mod_1` from the
		# beginning, and half of `mod_1` from the end)
		end = layer.inputdata.shape[0]

		# NOTE: Python and Matlab differ in how they specify ranges.
		#	That `+1` near `floor()` is not needed in Matlab.
		mask = list(range(0, floor(mod_1/2) + 1))
		mask += (list(range(end - ceil(mod_1/2) + 1, end)))

		layer.inputdata = np.delete(layer.inputdata, mask, 0)

	if (mod_2 != 0):
		# Remove rows of input images (half of `mod_1` from the
		# beginning, and half of `mod_1` from the end)
		end = layer.inputdata.shape[1]

		# NOTE: Python and Matlab differ in how they specify ranges.
		#	That `+1` near `floor()` is not needed in Matlab.
		mask = list(range(0, floor(mod_2/2) + 1))
		mask += (list(range(end - ceil(mod_2/2) + 1, end)))

		layer.inputdata = np.delete(layer.inputdata, mask, 1)

	if layer.whiten and layer.type_input == 'Gaussian':
		m = layer.inputdata.shape[3]
		n = layer.inputdata.shape[2]
		for i in range(m):
			for j in range(n):
				layer.inputdata[:,:,j,i] = crbm_whiten(
							layer.inputdata[:,:,j,i])

	if (verbose):
		print("Images new shape: {}".format(layer.inputdata.shape))

	return layer

### From `cdbn2D.m`
def cdbn2D(layers):
	model = []

	# layers{1} = preprocess_train_data2D(layers{1});
	layers[0] = preprocess_train_data2D(layers[0])
	print('layer 0:')
	model.append(crbm2D(layers[0]))

	for i in range(1, len(layers)):
		print('layer {}:'.format(i))
		layers[i].inputdata = model[i-1].output
		model.append(crbm2D(layers[i]))

	return (model, layers)

