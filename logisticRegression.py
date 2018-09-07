# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:16:25 2018

@author: asus
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons

class LR:
    def __init__(self):
        self.dim=2
        self.w=np.array([[1.0,1.0]])
        self.b=0
        self.eta=0.2
        
    def sigmoid(self,x):
        return 1.0/(1+np.exp(-x))
    
    def logistic_regression(self,x,y,eta):
        self.eta=eta
        row,column=np.shape(x)
        for itr in range(100):
            z=np.dot(self.w,x.T)+self.b
            A=self.sigmoid(z)
            dz=A-y
            dw=1/row*np.dot(x.T,dz.T)
            db=1/row*np.sum(dz)
            self.w -=dw.T*self.eta
            self.b -=db*self.eta
            
if __name__=='__main__':
    import matplotlib.pyplot as plt
   # data=pd.read_csv("D:/competition/exercixe/iris.csv")
   # x=data.loc[:,'Sepal.Length':'Petal.Width']  
   # y=data.loc[:,'Species']
    x,y=make_moons(250,noise=0.25)
    col = {0:'r',1:'b'}
    lr=LR()
    lr.logistic_regression(x,y,eta=1.2)
    plt.figure()
    plt.xlim([-1.5,2.5])
    plt.ylim([-1.5,1.5])
    for i in range(150):     
        plt.plot(x[i,0],x[i,1],col[y[i]]+'o')
    
    #plt.plot(np.dot(lr.w,x.T)+lr.b)
            
            
            
            
    