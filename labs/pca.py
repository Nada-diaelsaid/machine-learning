from sklearn.naive_bayes import GaussianNB
# import pandas as pd
from sklearn.decomposition import PCA
import csv
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np 
import math
from matplotlib import pyplot

# data = pd.read_csv("zoo.data",delim_whitespace=True,header=None)

raw_data = open("zoo.data", 'rb')
temp = list(csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE))
data = np.array(temp)

np.random.shuffle(data)
percentage = 75
lines = 0
labels = data[:,0]
x_actual = data[:,:]
#apply PCA
pca = PCA(n_components=2)
pca.fit_transform(x_actual)

x = 0
pyplot.figure(0)
for i in labels:
	if i == 1:	
		pyplot.scatter(pca[x,0], pca[x,1], color = "red")
	elif i == 2:
		pyplot.scatter(pca[x,0], pca[x,1], color = "green")
	elif i == 3:
		pyplot.scatter(pca[x,0], pca[x,1], color = "blue")
	elif i == 4:
		pyplot.scatter(pca[x,0], pca[x,1], color = "yellow")
	elif i == 5:
		pyplot.scatter(pca[x,0], pca[x,1], color = "violet")
	x+=1

# print(pca.explained_variance_ratio_)

######################################## then train using bayes..##################################

#Normzalizing Data
train_x = preprocessing.scale(train_x)
#get total number of data in the file
with open("zoo.data") as f:
      lines =  len(f.readlines())

train_x = x_actual[:math.ceil(lines * (percentage/100))]
test_x = x_actual[:math.ceil((lines*(percentage/100))+1):]

train_label = labels[:math.ceil(lines*(percentage/100))]
test_label = labels [math.ceil((lines*(percentage/100)+1)):]
#train
gnb = GaussianNB()
gnb.fit(train_x,train_label)
#prediction
output = clf.predict(test_x)
accuracy = accuracy_score(test_label, output)
print (accuracy)
