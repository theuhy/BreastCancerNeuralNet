'''
Using Neural Network on Breast Cancer data
Author: KAUSHIK BALAKRISHNAN, PhD
Email: kaushikb258@gmail.com
For more details: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
'''

import numpy as np
import pandas as pd
import random
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


def read_data():
 data = pd.read_table("BC.data", sep=",",header=None)
 data = np.array(data)
 # randomly shuffle data by rows
 np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)

 X = data[:,2:]
 Y = []
 for i in range(data.shape[0]): 
  if data[i,1] == "M": 
    cancer = 1
  else:
    cancer = 0   
  Y.append(cancer)  
 rows = X.shape[0]
 cols = X.shape[1]
 print "rows = ", rows
 print "cols = ", cols 
 X = normalize(X)

 x_train = []
 x_test = []
 y_train = []
 y_test = []

 for i in range(rows):
   if(random.random() < 0.9):
    x_train.append(X[i,:])
    y_train.append(Y[i])
   else: 
    x_test.append(X[i,:])
    y_test.append(Y[i])

 x_train = np.array(x_train)
 y_train = np.array(y_train)
 x_test = np.array(x_test)
 y_test = np.array(y_test)
 return x_train, y_train, x_test, y_test   


def normalize(X):
 for j in range(X.shape[1]):
  xmin = np.min(X[:,j])
  xmax = np.max(X[:,j])  
  for i in range(X.shape[0]):
   X[i,j] = (X[i,j]-xmin)/(xmax-xmin)  
 return X


def get_mean_and_std(x_test):
 nr = x_test.shape[0]
 nc = x_test.shape[1]
 xmean = np.zeros(nc,dtype=np.float32)
 for i in range(nr):
  for j in range(nc):
    xmean[j] += x_test[i,j]
 for j in range(nc):
  xmean[j] = xmean[j]/np.float(nr)
 xstd = np.zeros(nc,dtype=np.float32)
 for i in range(nr):
  for j in range(nc):
    xstd[j] += (x_test[i,j] - xmean[j])**2.0
 for j in range(nc):
  xstd[j] = np.sqrt(xstd[j]/np.float(nr))
 return xmean, xstd 



def get_zscore(x,xmean,xstd):
 z = np.zeros(x.shape[0],dtype=np.float32)
 for i in range(x.shape[0]):  
  z[i] = (x[i] - xmean[i])/xstd[i]
 return z

 

if __name__ == '__main__':
 x_train, y_train, x_test, y_test = read_data()
 


solv = 2
# 1 = SGD
# 2 = ADAM
# 3 = LBFGS



if (solv == 1):
 print "SGD"
 nn = MLPClassifier(solver='sgd', hidden_layer_sizes=(16,12), alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=500, shuffle=True, random_state=None, activation="logistic", tol=1.0e-5, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False)
elif (solv == 2):
 print "ADAM"
 nn = MLPClassifier(solver='adam', hidden_layer_sizes=(16,12), alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=500, shuffle=True, random_state=None, activation="logistic", tol=1.0e-5, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
elif (solv == 3):
 print "LBFGS"
 nn = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(16,12), alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=500, shuffle=True, random_state=None, activation="logistic", tol=1.0e-5, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False)



nn.fit(x_train, y_train)
y_predict = nn.predict(x_test)


print y_test.shape[0]
print "actual class, predicted class "
for i in range(y_test.shape[0]):
 print y_test[i], y_predict[i]

error = 0
for i in range(y_test.shape[0]):
 if (y_test[i] != y_predict[i]):
   error += 1
print "error = ", error, "out of ", y_predict.shape[0]


xmean, xstd = get_mean_and_std(x_test)


tp = 0
fp = 0
tn = 0
fn = 0

for i in range(y_test.shape[0]):
 if y_test[i] == 1:
  if y_predict[i] == 1:
   tp += 1
  else:
   fn += 1
   print "zscore = ", get_zscore(x_test[i,:],xmean,xstd)
 else:
  if y_predict[i] == 0:
   tn += 1
  else:
   fp += 1

print "true positives = ", tp
print "true negatives = ", tn
print "false positives = ", fp
print "false negatives = ", fn







