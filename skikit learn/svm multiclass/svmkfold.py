print(__doc__)


import numpy as np
from sklearn import svm
from sklearn import cross_validation
import csv

#with open("C:/Users/Aditi/Desktop/cs412/torch/skilearn/trial1_aditi.csv", "r") as f:
 #   reader = csv.reader(f,delimiter=",")
  #  for row in reader:
   #     print row
    #    for element in 

f = open("C:/Users/Aditi/Desktop/cs412/torch/skilearn/combine2classgesture.csv", "r")
lines = f.read().split("\n") # "\r\n" if needed
X=[]
y=[]
i=0
for line in lines:
    if line != "": # add other needed checks to skip titles
        cols = line.split(",")
        X.append(list(cols[0:69]))
        y.append(cols[-1])
   
X= np.array(X).astype(np.float)     #convert intoo floats
y=np.array(y).astype(np.int)        #convert labels into int

n_sample = len(X)           # know number of samples 

np.random.seed(0)           # set seed to randomly select datapoints
order = np.random.permutation(n_sample)     # generate a random sequence of n datapoints    
X = X[order]                            #assign the input feature and labels of the permutated data
y = y[order]                            

X_folds = np.array_split(X, 10)
y_folds = np.array_split(y, 10)

for fig_num,kernel in enumerate(('linear', 'rbf', 'poly')):
    clf1 = svm.SVC(kernel=kernel, gamma=10)
    scores = list()
    for k in range(10):
        X_train = list(X_folds)
        X_test  = X_train.pop(k)
        X_train = np.concatenate(X_train)
        y_train = list(y_folds)
        y_test  = y_train.pop(k)
        y_train = np.concatenate(y_train)
        scores.append(clf1.fit(X_train, y_train).score(X_test, y_test))
        print kernel," ","kfold"," ",k," ","number of support vectors: ",clf1.n_support_;    
    print kernel," ",scores
# 