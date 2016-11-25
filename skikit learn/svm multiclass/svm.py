print(__doc__)


import numpy as np
from sklearn import svm

f = open("C:/Users/Aditi/Desktop/cs412/torch/skilearn/combine.csv", "r")
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


# break down into 10 folds combine 9 at a time
X_train = X[:.9 * n_sample]             #select 90% of the sample  as training
y_train = y[:.9 * n_sample]             # same for labels first 90%
X_test = X[.9 * n_sample:]          # last 10 % as test
y_test = y[.9 * n_sample:]
  

# fit the model
for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf1 = svm.SVC(kernel=kernel, gamma=10)
    clf1.fit(X_train, y_train)          #without cross val
    print clf1.decision_function(X_train)           #distance from margin
    
    y_pred = clf1.predict(X_test)
    match =0;
    for i in range(0,len(y_pred)):
        if (y_pred[i]==y_test[i]):
            match= match +1
                     
    print (kernel, "support vectors for each class: ",clf1.n_support_)        
    accuracy = match/(len(y_pred)*1.0)
    print (kernel," accuracy= ",accuracy);
    
   

    

 