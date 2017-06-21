class Layer:
	""" A "data" class to carry setup information for the CDBN. """
	def __init__(self,
			inputdata     = None,
			n_map_v       = 1,
			n_map_h       = 9,
			s_filter      = [7, 7],
			s_pool        = [2, 2],
			n_epoch       = 20,
			learning_rate = 0.05,
			sparsity      = 0.02,
			lambda1       = 5,
			lambda2       = 0.05,
			start_gau     = 0.2,
			stop_gau      = 0.1,
			batchsize     = 1,
			n_cd          = 1,
			momentum      = 0.9,
			whiten        = True,
			type_input    = 'Binary',
			cpu           = 'mex',
			stride        = [1, 1],
		):
		self.inputdata     = inputdata

		# This has to be set after data preprocessing (i.e., much later
		# than when Layer is created)
		self.s_inputdata   = None


		# (The following are the comments from the original code)

		# NUMBER OF VISIBLE FEATURE MAPS 
		self.n_map_v       = n_map_v
		# NUMBER OF HIDDEN FEATURE MAPS
		self.n_map_h       = n_map_h
		# SIZE OF FILTER
		self.s_filter      = s_filter
		# SIZE OF POOLING
		self.s_pool        = s_pool
		# NUMBER OF ITERATION
		self.n_epoch       = n_epoch
		# RATE OF LEARNING
		self.learning_rate = learning_rate
		# HIDDEN UNIT SPARSITY
		self.sparsity      = sparsity
		# GAIN OF THE LEARNING RATE FOR WEIGHT CONSTRAINTS
		self.lambda1       = lambda1
		# WEIGHT PENALTY
		self.lambda2       = lambda2
		# GAUSSIAN START
		self.start_gau     = start_gau
		# GAUSSIAN END
		self.stop_gau      = stop_gau
		# SIZE OF BATCH IN TRAINING STEP
		self.batchsize     = batchsize
		# NUMBER OF GIBBS SAMPLES
		self.n_cd          = n_cd
		# GRADIENT MOMENTUM FOR WEIGHT UPDATES
		self.momentum      = momentum
		# WHETHER TO BE WHITEN
		self.whiten        = whiten
		# INPUT STYPE
		self.type_input    = type_input
		# computation type 'matlab' or 'mex' or 'cuda'
		self.cpu           = cpu
		# STRIDE OF FILTER MOVE
		# YOU SHOULD OBEY THE RULE:
		# mod = ((size(inputdata)-s_filter),stride) == 0!
		# AND IN FACT, THE STRIDE PARAMETER ONLY
		# HAS EFFECTS IN FEEDDORWARD STEP
		self.stride        = stride
 
