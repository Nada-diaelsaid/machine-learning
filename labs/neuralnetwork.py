import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle
import csv, math
from numpy import ceil

# Declare constants
HIDDEN_NEURONS = 50
NUM_CLASSES = 7
NUM_FEATURES = 0
LEARNING_RATE = 0.01
XVAL_SIZE = 0.1
NUM_EPOCHS = 10

raw_data = open("zoo.data", 'rb')
temp = list(csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE))
data = np.array(temp)

np.random.shuffle(data)

percentage = 75
lines = 0
labels = data[:,0]
x_actual = data[:,0:]

with open("wine.data") as f:
      lines =  len(f.readlines())

X_Train = x_actual[0:int(np.ceil(lines * (percentage/100)))]
X_Test = x_actual[int(ceil((lines*(percentage/100)) )+ 1):]
print(dtype(X_Train))
# # [float(i) for i in X_Train]
# # [float(i) for i in X_Test]
# X_Train = X_Train.astype(float)
# X_Test = X_Test.astype(float)
# print(type(X_Train))

train_label = labels[:int(ceil(lines*(percentage/100)))]
test_label = labels [int(ceil((lines*(percentage/100)+1))):]

LEARNING_RATE = 0.5
NUM_EPOCHS = 200

rng = np.random.RandomState(1234)

NUM_FEATURES = X_Train.shape[1]

# Declare symbols
x = T.matrix('Inputs')
y = T.fmatrix('Outputs')
t = T.fmatrix('Target Values')
Netj = T.dmatrix('Net of hidden layer')
Netk = T.dmatrix('Net of output layer')
Aj = T.dmatrix('Activation of hidden layer')
Wj = np.random.rand(NUM_FEATURES, HIDDEN_NEURONS) * 0.01
Wk = np.random.rand(HIDDEN_NEURONS, NUM_CLASSES) * 0.01
Weights = theano.shared(value=np.concatenate((Wj.ravel(), Wk.ravel()), axis=0),
                        name="Weights ravelled")

# Define equations
Netj = T.dot(x, Weights[0:NUM_FEATURES * HIDDEN_NEURONS]
             .reshape((NUM_FEATURES, HIDDEN_NEURONS)))


Aj = T.nnet.sigmoid(Netj)

Aj_test = Aj


y_test = T.nnet.softmax(Aj_test)

cost = T.mean(T.nnet.categorical_crossentropy(y_test, t))


Grads = T.grad(cost, Weights)

# Define Functions

computeCost = theano.function([y_test, t], cost)

forwardProp = theano.function([x], y_test)
forwardProp_test = theano.function([x], y_test)


updates = [(Weights, Weights - LEARNING_RATE * (Grads))]
trainModel = theano.function([x, t], cost, updates=updates)


costs = {'training': list(), 'xval': list()}
for i in range(NUM_EPOCHS):
    print "Epoch number: " + str(i + 1)
    costs['training'].append(trainModel(X_Train, train_label))
    if(i % 10 == 0 and i > 0):
        Test_Result = np.argmax(forwardProp_test(X_Test), axis=1)
        Score = float(len(np.where(Test_Result == train_label)[0])) / float(
            (train_label.shape[0])) * 100
        print "The model classified with an accuracy of: %.2f" % (
            float(Score)) + "%"

plt.plot(range(NUM_EPOCHS), costs['training'],
         range(NUM_EPOCHS), costs['xval'])
plt.show()
Test_Result = np.argmax(forwardProp(X_Test), axis=1)
Score = float(len(np.where(Test_Result == train_label)[0])) / float(
    (train_label.shape[0])) * 100
print "The model performed with an accuracy of: %.2f" % (float(Score)) + "%"
