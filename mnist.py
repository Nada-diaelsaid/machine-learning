import theano.tensor as T
import pandas as p 
import numpy as np 
from theano import function
from theano import shared 
import matplotlib.pyplot as plt
from six.moves import cPickle

trainData = cPickle.load(open("train.pickle", "rb"))
Test_set = cPickle.load(open("test.pickle", "rb"))


X_Test = Test_set[:, 1:]
X_Test = (X_Test - X_Test.mean()) / X_Test.std()
X_Test = np.c_[np.ones((X_Test.shape[0], 1)), X_Test]
Y_Test = Test_set[:, 0]
print Y_Test
LEARNING_RATE = 0.01
EPOCHES = 5
HIDDEN_NEURONS = 20
NUM_CLASSES = 10
NUM_FEATURES = trainData.shape[1]-1
cross_val = 0.3

x_train = trainData[cross_val * trainData.shape[0]:, 1:]
x_train = (x_train - x_train.mean()) / x_train.std()
x_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]
y_train = trainData[cross_val * trainData.shape[0]:, 0]
y_train_onehot = np.zeros((y_train.shape[0], NUM_CLASSES))
##############2asem el dataa and normaization 
#i/ps(features)
x_train = trainData[cross_val * trainData.shape[0]:,1:]
x_train = (x_train - x_train.mean())/x_train.std()
#o/p(labels)
y_train = trainData[cross_val * trainData.shape[0]:, 0]


#add the bias 
x0= np.ones((np.shape(x_train)[0],1))
x_train =  np.append(x0,x_train, axis=1)

train_onehot = np.zeros((x_train.shape[0], NUM_CLASSES))

train_onehot[np.arange(x_train.shape[0]), y_train] = 1
y_train = train_onehot


# print trainData
# print"---"
# print testData

# Declare symbols
Wji = np.random.rand(NUM_FEATURES + 1, HIDDEN_NEURONS) #weights of first layer form i/p to hidden
Wkj = np.random.rand(HIDDEN_NEURONS, NUM_CLASSES) #weights of 2nd layer from hidden to o/p

x = T.dmatrix('Inputs')
# y = T.dmatrix('Outputs')
t = T.dmatrix('Target Values')
# aj = T.dmatrix('Net of hidden layer')
# ak = T.dmatrix('Net of output layer')
wji = shared(Wji)
wkj = shared(Wkj)

#######################forward propagation
aj = T.dot(x,wji)
hj = T.nnet.sigmoid(aj)
ak = T.dot(hj,wkj)
#activate
y = T.nnet.softmax(ak)

#######################backpropagation 

error = T.mean(T.nnet.categorical_crossentropy(y,t))
# error = T.sum(T.sub(t,y)**2)/(2)

gradkj = T.grad(error,wkj)
# deltawkj = -1*LEARNING_RATE*gradkj

gradji = T.grad(error,wji)
# deltawji = -1*LEARNING_RATE*gradji
updates = [(wji, wji - LEARNING_RATE*gradji),(wkj, wkj - LEARNING_RATE*gradkj)]

mlp = function(inputs=[x,t], outputs=[error, wji,wkj], updates = updates)

X1 = T.matrix('X1')
O = T.nnet.softmax(T.dot(T.nnet.sigmoid(T.dot(X1, wji)), wkj))

forwardProp = function([X1], [O])
costs = list()

for i in xrange(EPOCHES):
	print "Epoch number: ", i
	costs.append(mlp(x_train, y_train)[0])

# print X_Test
Test_Result = forwardProp(X_Test)
print Test_Result
print np.argmax(Test_Result,axis = 0)
print Y_Test
# print np.argmax(Test_Result)
# print "ok"
# print Y_Test 
# Score = float(len(np.where(Test_Result == Y_Test)[0])) / float((Y_Test.shape[0])) * 100

# print "The model classified with an accuracy of: %.2f" % (float(Score)) + "%"

# plt.plot(range(EPOCHES), costs)
# plt.show()