from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# import pandas as pd
import csv
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np 
import math
# data = pd.read_csv("wine.data",delim_whitespace=True,header=None)

raw_data = open("zoo.data", 'rb')
temp = list(csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE))
data = np.array(temp)

np.random.shuffle(data)
percentage = 75
lines = 0
labels = data[:,0]
x_actual = data[:,0:]

#Normzalizing Data
train_x = preprocessing.scale(train_x)
#get total number of data in the file
with open("wine.data") as f:
      lines =  len(f.readlines())

train_x = x_actual[:math.ceil(lines * (percentage/100))]
test_x = x_actual[:math.ceil((lines*(percentage/100))+1):]

train_label = labels[:math.ceil(lines*(percentage/100))]
test_label = labels [math.ceil((lines*(percentage/100)+1)):]
#train
gnb = LinearDiscriminantAnalysis()
gnb.fit(train_x,train_label)
#prediction
output = clf.predict(test_x)
accuracy = accuracy_score(test_label, output)
print (accuracy)
