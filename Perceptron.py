# -*- coding: utf-8 -*-

#Secondly we are coding the perceptron algorithm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
np.random.seed(42) #same set of numbers will appear each time

def perceptronStep(x,y,w,b,alpha=0.01):
    for i in range(len(x)):
        x1=x.iloc[i][0]
        x2=x.iloc[i][1]
        #if line positive will return one,if not 0
        y_hat=1.0 if x1 * w[0] + x2 * w[1] + b > 0 else 0.0
        
        if y[i]-y_hat==-1:
            w[0]-=x1*alpha #w are the coeff before x in equation
            w[1]-=x2*alpha
            b-=alpha
        elif y[i]-y_hat==1:
            w[0]+=x1*alpha #w are the coeff before x in equation
            w[1]+=x2*alpha
            b+=alpha
    return w,b

def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

import pandas as pd
data=pd.read_csv("data.csv",header=None)
data.head()
X_train=data.iloc[:,: -1]
y_label=data.iloc[:,2]
#w = [-0.1, 0.20653640140000007, -0.23418117710000003] 
#b=-0.1 
#W = np.array(np.random.rand(2,1))
#b = np.random.rand(1)[0]
    
#trainPerceptronAlgorithm(X_train, x_label, learn_rate = 0.01, num_epochs = 25)           

p=perceptronStep(X_train,y_label,w,b,alpha=0.01)   
    
        
plotting= trainPerceptronAlgorithm(X_train, y_label, learn_rate = 0.01, num_epochs = 25)

#plotting our dataset initially
plt.style.use('ggplot')
colors = ['red', 'blue']
plt.scatter(X_train.iloc[:,0], X_train.iloc[:,1], c=y_label, cmap=ListedColormap(colors), edgecolors='k')
plt.axis('equal')
plt.show()            
def plot_graph(slope,intercept,pattern):
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, pattern)

boundary_lines = trainPerceptronAlgorithm(X_train, y_label)

counter = 1
for line in boundary_lines:
    slope, intercept = line[0], line[1]
    if len(boundary_lines) == counter:
        plot_graph(slope, intercept, 'g--') # last line
    else:
        plot_graph(slope, intercept, 'y--') # Intermediatte line
    counter = counter + 1
    
plt.scatter(X_train.iloc[:,0], X_train.iloc[:,1], c=y_label, cmap=ListedColormap(colors), edgecolors='k')
plt.axis('equal')
plt.xlim((X_train.iloc[:,0].min() - 0.1, X_train.iloc[:,0].max() + 0.1))
plt.ylim((X_train.iloc[:,1].min() - 0.1, X_train.iloc[:,1].max() + 0.1))
plt.show()
        
    
    

                        
