import numpy as np 
from .NetLib import initWeight, sigmoid, ReLU, sigmoidGrad, ReLUGrad
import random

class Network:

	def __init__(self):
		self.weight = []
		self.layers = []
		self.actOfLayers = []

	def creatLayer(self, numNeurons, activation):
		self.layers.append(numNeurons)
		if len(self.actOfLayers) == 0 :
			self.actOfLayers.append("None")
		else :
			self.actOfLayers.append(activation)

		return self

	def printInfo(self):
		print("Network has ", len(self.layers), " layers : \n")

		for i in range(len(self.layers)):
			print("Layer ", i, " : ", self.layers[i], " neurons. Activation : ", self.actOfLayers[i])
		print(self.weight)

	def createWeight(self):
		for i in range(len(self.layers) - 1):
			self.weight.append(initWeight(self.layers[i], self.layers[i+1]));

	# return costfunction and grad of weight
	def costNNs(self, X, y, decay):
		ouputSize = self.layers[-1]
		grad = [np.zeros(w.shape) for w in self.weight]
		m = X.shape[1]

		numLayers = len(self.layers)

		realY = np.zeros((ouputSize, m))
		for i in range(m) :
			realY[y[0][i]][i] = 1
		cost = 0

		#forward
		activation = np.concatenate((np.ones((1, m)), X), axis=0)
		activations = [activation] # store all activation of layers
		zs = [] # store all z in layers 
		i = 1

		for w in self.weight:

			#print("len z : {}".format(len(zs)))
			#print("{}\n".format(w.shape))
			z =  np.dot(w, activation)
			zs.append(z)

			if self.actOfLayers[i] == "ReLU" :
				activation = ReLU(z)
				if  i < numLayers - 1 : 
					activation = np.concatenate((np.ones((1, m)), activation), axis=0)

			elif self.actOfLayers[i] == "sigmoid":
				activation = sigmoid(z)
				if i < numLayers - 1 :
					activation = np.concatenate((np.ones((1, m)), activation), axis=0)

			else :
				print("Cannot find activation " + self.actOfLayers[i] + "\n")
			i += 1

			activations.append(activation)

		# compute costfunction 
		#print(zs[-2])
		#print(activations[-2])
		loss = (1 / m) * sum(sum(-realY*np.log2(activation) - realY*np.log2(1 - activation) ))
		decayTerm = 0

		#compute decay term
		for w in self.weight:
			decayTerm += sum(sum( w[:, 1:]**2 ))

		decayTerm = (decay / (2)) * decayTerm
		#print("DecayTerm : {}. \n".format(decayTerm))
		cost += loss + decayTerm

		# Backpropagation:
		if self.actOfLayers[-1] == "sigmoid" :
			prime = sigmoidGrad(zs[-1])
		delta = (activation - realY) * prime
		grad[-1] = np.dot(delta, activations[-2].T)

		for l in range(2, len(self.layers)):
			z = zs[-l]
			
			if self.actOfLayers[-l] == "sigmoid":
				prime = activations[-l] * (1 - activations[-l]) #sigmoidGrad(z)
			elif self.actOfLayers[-l] == "ReLU":
				prime = 1 * (activations[-l] > 0) #ReLUGrad(z)

			delta = np.dot(self.weight[-l + 1].T, delta) * prime
			delta = np.delete(delta, 0, 0)
			#print("{} : {}\n".format("delta", delta.shape))
			grad[-l] = np.dot(delta, activations[-l - 1].T)
		
		for i in range(len(self.weight)):
			grad[i] = grad[i] + decay * np.concatenate((np.zeros((self.weight[i].shape[0], 1)), self.weight[i][:, 1:]), axis=1)
		return (cost, grad)

	def updateMiniBatch(self, miniBatch, alpha, decay):
		X = np.array(miniBatch[0])
		X = (1 / 255.0) * X.transpose()
		y = np.array(miniBatch[1])

		(cost, grad) = self.costNNs(X, y, decay)

		#for w, grad in zip(self.weight, grad):
		#	w -= alpha * grad
		self.weight = [w - alpha*grad for w, grad in zip(self.weight, grad)]
		print("Cost = {}\n".format(cost))

	def trainMiniBatch(self, dataTraining, epochs, batchSize, alpha, decay):
		x = np.array(dataTraining[0])
		y = np.array([dataTraining[1]])
		m = x.shape[0]
		print(m)

		for j in range(epochs):
			print("Epoch{} : ".format(j))
			#random.shuffle([dataTraining])
			miniBatches = [[x[k:k+batchSize, :], y[:, k:k+batchSize]] for k in range(0, m, batchSize)]

			for mini_batch in miniBatches:
				self.updateMiniBatch(mini_batch, alpha, decay)

	















		
