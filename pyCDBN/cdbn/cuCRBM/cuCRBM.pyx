import numpy as np
cimport numpy as np

np.import_array()

from model import Model
from layer import Layer

from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libc.stdint cimport intptr_t

assert sizeof(int) == sizeof(np.int32_t)
assert sizeof(float) == sizeof(np.float32_t)

cdef extern from "crbm2D_cuda.h":
	cdef cppclass C_CRBM_Data "CRBM_Data":
		float *data_input
		float *data_kernel
		float *v_bias
		float *h_bias
		float *v_sample
		float *h_sample
		float *h_sample_init
		float *h_state
		float  gauss
		int     H, W
		int     N
		int     Wfilter, Hfilter
		int     Wres, Hres
		int     Hstride, Wstride
		int     Hpool, Wpool
		int     n_map_v, n_map_h
		char    type_input
		int     run_on_gpu

		C_CRBM_Data() except +

		void setDevice(int d)
		void gibbs_sample()
		double *calculate_dW(np.npy_intp *dim_w)
		#void set_data_input(float *di)
		#void get_data_input(float **di)
		#void set_data_kernel(float *di)
		#void get_data_kernel(float **di)

	cdef void crbm_inference2D(C_CRBM_Data *p)
	cdef void crbm_reconstruct2D(C_CRBM_Data *p)

def inference2D(CRBM_Data p):
	return crbm_inference2D(p.c)

def reconstruct2D(CRBM_Data p):
	return crbm_reconstruct2D(p.c)


cdef class CRBM_Data:
	cdef C_CRBM_Data* c

	# Needed for keeping track of the length of these vectors
	cdef int data_input_dim
	cdef int h_sample_dim
	cdef int v_sample_dim

	def __init__(self, model, layer, batch_data):
		#printf("c: %p", self.c)
		pass

	# I should probably make some type checks here
	def __cinit__(self, model, layer, batch_data):
		self.c = new C_CRBM_Data()

		cdef int H, W, Hfilter, Wfilter, Hres, Wres
		cdef int Hstride, Wstride, Hpool, Wpool
		cdef int curr_vec_size, i, j

		H           = int(model.v_input.shape[0]);
		W           = int(model.v_input.shape[1]);
		N           = 1 if (len(model.v_input.shape) != 4) \
				else model.v_input.shape[3]
		Hfilter     = int(layer.s_filter[0]);
		Wfilter     = int(layer.s_filter[1]);
		Hres        = int(model.h_input.shape[0]);
		Wres        = int(model.h_input.shape[1]);
		Hstride     = int(layer.stride[0]);
		Wstride     = int(layer.stride[1]);
		Hpool       = int(layer.s_pool[0]);
		Wpool       = int(layer.s_pool[1]);
		n_map_v     = layer.n_map_v
		n_map_h     = layer.n_map_h

		# First, initialize the known-sized variables
		self.c.H          = H
		self.c.W          = W
		self.c.N          = N
		self.c.Hfilter    = Hfilter
		self.c.Wfilter    = Wfilter
		self.c.Hres       = Hres
		self.c.Wres       = Wres
		self.c.Hstride    = Hstride
		self.c.Wstride    = Wstride
		self.c.Hpool      = Hpool
		self.c.Wpool      = Wpool
		self.c.n_map_v    = n_map_v
		self.c.n_map_h    = n_map_h
		# See the following link for why there is a `ord` cast here:
		# http://stackoverflow.com/questions/28002214/cython-typeerror-an-integer-is-required
		self.c.type_input = ord(layer.type_input[0])
		self.c.gauss      = model.start_gau

		# Then, initialize the pointers
		curr_vec_size = H * W * n_map_v * N
		batch_data = batch_data.ravel(order='F')
		self.c.data_input  = <float*>malloc(sizeof(float) * curr_vec_size)
		for i in range(curr_vec_size):
			self.c.data_input[i] = batch_data[i]

		curr_vec_size = Hfilter * Wfilter * n_map_v * n_map_h
		model_W = model.W.ravel(order='F')
		self.c.data_kernel = <float*>malloc(sizeof(float) * curr_vec_size)
		for i in range(curr_vec_size):
			self.c.data_kernel[i] = model_W[i]

		h_sample_dim = Hres * Wres * n_map_h * N;
		self.c.h_sample_init = <float*>malloc(sizeof(float) * h_sample_dim)
		self.c.h_sample      = <float*>malloc(sizeof(float) * h_sample_dim)
		self.c.h_state       = <float*>malloc(sizeof(float) * h_sample_dim)
		for i in range(h_sample_dim):
			self.c.h_sample_init[i] = 0
			self.c.h_sample[i]      = 0
			self.c.h_state[i]       = 0

		v_sample_dim = H * W * n_map_v * N
		self.c.v_sample = <float*>malloc(sizeof(float) * v_sample_dim)
		for i in range(v_sample_dim):
			self.c.v_sample[i] = 0

		self.c.v_bias = <float*>malloc(sizeof(float) * n_map_v)
		for i in range(n_map_v):
			self.c.v_bias[i] = float(model.v_bias[i])

		self.c.h_bias = <float*>malloc(sizeof(float) * n_map_h)
		for i in range(n_map_h):
			self.c.h_bias[i] = float(model.h_bias[i])

	# Probably I should use __dealloc__()
	def __dealloc__(self):
		if(self.c.data_input    != NULL ):
			free(self.c.data_input)
		if(self.c.data_kernel   != NULL ):
			free(self.c.data_kernel)
		if(self.c.h_bias        != NULL ):
			free(self.c.h_bias)
		if(self.c.v_bias        != NULL ):
			free(self.c.v_bias)
		if(self.c.h_sample      != NULL ):
			free(self.c.h_sample)
		if(self.c.h_sample_init != NULL ):
			free(self.c.h_sample_init)
		if(self.c.h_state       != NULL ):
			free(self.c.h_state)
		if(self.c.v_sample      != NULL ):
			free(self.c.v_sample)

		self.c.data_input    = NULL
		self.c.data_kernel   = NULL
		self.c.h_bias        = NULL
		self.c.v_bias        = NULL
		self.c.h_sample      = NULL
		self.c.h_sample_init = NULL
		self.c.h_state       = NULL
		self.c.v_sample      = NULL

	def setDevice(self, d):
		self.c.setDevice(d)

	def gibbs_sample(self):
		self.c.gibbs_sample()

	def calculate_dW(self, np.ndarray[np.npy_intp, ndim=1, mode="c"] dim_w):
		cdef double *dW = self.c.calculate_dW(&dim_w[0])

		# FIXME: I MAY NEED TO INVERT THE DIMENSIONS HERE, AND THEN
		#        RETURN `ret.T`.

		# This expects `dW` to be in `C-style`
		# (see https://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html)
		ret = np.PyArray_SimpleNewFromData(
			4,
			&dim_w[0],
			np.NPY_FLOAT64,
			dW)

		# So, we "invert" it into `F` (Fortran) style
		return ret.T.astype(np.float32)

	def get_h_sample(self):
		ret = np.PyArray_SimpleNewFromData(
			4,
			#[self.c.Hres, self.c.Wres, self.c.n_map_h, self.c.N],
			[self.c.N, self.c.n_map_h, self.c.Wres, self.c.Hres],
			np.NPY_FLOAT32,
			self.c.h_sample)
		return ret.T

	def get_h_sample_init(self):
		ret = np.PyArray_SimpleNewFromData(
			4,
			#[self.c.Hres, self.c.Wres, self.c.n_map_h, self.c.N],
			[self.c.N, self.c.n_map_h, self.c.Wres, self.c.Hres],
			np.NPY_FLOAT32,
			self.c.h_sample_init)
		return ret.T

	def get_v_sample(self):
		ret = np.PyArray_SimpleNewFromData(
			4,
			#[self.c.H, self.c.W, self.c.n_map_v, self.c.N],
			[self.c.N, self.c.n_map_v, self.c.W, self.c.H],
			np.NPY_FLOAT32,
			self.c.v_sample)
		return ret.T

