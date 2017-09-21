from neuralnet import NeuralNet
import numpy as np

def test1():
	X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
	y=np.array([[1],[1],[0]])
	nn = NeuralNet()
	nn.train(X, y, epochs=5000)
	pred = nn.predict(X)
	print(pred)

def test2():
	X1 = np.array([[0,0],
					[0,1],
               		[1,0],
               		[1,1]])
	y1 = np.array([0,1,1,0])
	nn = NeuralNet(input_layer=2, hidden_layer=4, output_layer=1)
	nn.train(X1, y1)
	pred = nn.predict(X1)
	print(pred)

test1()