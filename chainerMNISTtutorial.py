# @Author: Joeri Nicolaes
# status: OK
# based on https://docs.chainer.org/en/stable/tutorial/basic.html
# see https://github.com/chainer/chainer/blob/master/examples/mnist/train_mnist.py for full example using gpu wrapped in main loop

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

# 1. DATA - The MNIST dataset consists of 70,000 greyscale images of size 28x28 (i.e. 784 pixels) and corresponding digit labels. The dataset is divided into 60,000 training images and 10,000 test images by default.
train, test = datasets.get_mnist()  # get vectorized version (N,784)

# shuffle dataset for every epoch using Python iterator obÎject
train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

# 2. MODEL - simple 3x layer FC with 100 nodes each
class MLP(Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)   # n_in > n_units
            self.l2 = L.Linear(None, n_units)   # n_units > n_units
            self.l3 = L.Linear(None, n_out)  # n_units > n_out

    def __call__(self, x):
        """
        Forward pass
        :param x: input tensor (N,M) with N inputs dims and M features
        :return: output to the chain model
        """
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

# In order to compute loss values or evaluate the accuracy of the predictions, we define a classifier chain on top of the above MLP chain
class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss

#This Classifier class computes accuracy and loss, and returns the loss value.
# The pair of arguments x and t corresponds to each example in the datasets (a tuple of an image and a label).

model = L.Classifier(MLP(100, 10))  # the input size, 784, is inferred
optimizer = optimizers.SGD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (20, 'epoch'), out='result')

# start train run
# trainer.run()

# see how training proceeds
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()