import numpy as np

class NeuralNet(object):
	"""
	A Neural Network Walkthrough
	"""
	def __init__(self, input_layer=4, hidden_layer=3, output_layer=1):
		"""Design and Initialise the Neural Network"""
		np.random.seed(42)
		self.wh=np.random.uniform(size=(input_layer,hidden_layer))
		self.bh=np.random.uniform(size=(1,hidden_layer))
		self.wout=np.random.uniform(size=(hidden_layer,output_layer))
		self.bout=np.random.uniform(size=(1,output_layer))


	def train(self, X, y, epochs = 10, lr = 0.1, verbose = True):
		"""Train the weigths of the neural network"""

		# Begin Training
		for i in range(epochs):
			#Forward Propogation
			hidden_layer_input=np.dot(X,self.wh) + self.bh
			hiddenlayer_activations = self.sigmoid(hidden_layer_input)
			output_layer_input=np.dot(hiddenlayer_activations,self.wout) + self.bout
			output = self.sigmoid(output_layer_input)

			#Backpropagation
			E = np.subtract(y,output)
			slope_output_layer = self.derivatives_sigmoid(output)
			slope_hidden_layer = self.derivatives_sigmoid(hiddenlayer_activations)
			delta_output = np.multiply(E, slope_output_layer)
			error_hidden_layer = delta_output.dot(self.wout.T)
			delta_hiddenlayer = error_hidden_layer * slope_hidden_layer
			self.wout += hiddenlayer_activations.T.dot(delta_output) *lr
			self.bout += np.sum(delta_output, axis=0,keepdims=True) *lr
			self.wh += X.T.dot(delta_hiddenlayer) *lr
			self.bh += np.sum(delta_hiddenlayer, axis=0,keepdims=True) *lr

	def predict(self, X):
		""""""
		hidden_layer_input=np.dot(X,self.wh) + self.bh
		hiddenlayer_activations = self.sigmoid(hidden_layer_input)
		output_layer_input=np.dot(hiddenlayer_activations,self.wout) + self.bout
		output = self.sigmoid(output_layer_input)
		return output
	
	#Sigmoid Function
	def sigmoid (self,x):
		return 1/(1 + np.exp(-x))

	#Derivative of Sigmoid Function
	def derivatives_sigmoid(self,x):
		return x * (1 - x)