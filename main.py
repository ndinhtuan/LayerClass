from Network.Network import Network
from python_mnist.mnist import MNIST 
import numpy as np
from Network.NetLib import initWeight, sigmoid, ReLU, sigmoidGrad, ReLUGrad

net = Network()
net.creatLayer(784, "None").creatLayer(400, "ReLU").creatLayer(400, "ReLU").creatLayer(10, "sigmoid")
net.createWeight()

data = MNIST('python_mnist\data')
t = data.load_training()

net.trainMiniBatch(t, 3, 50, 0.01, 5e-4)
