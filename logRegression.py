# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:59:38 2018

@author: asus
"""
import numpy as np
def loadData():
    dataSet=[]
    labelSet=[]
    fr=open('testSet.txt')
    for lines in fr.readlines():
        lineArr=lines.strip().split();
        dataSet.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelSet.append(int(lineArr[2]))
    return dataSet,labelSet
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def gradientDescent(dataSet,labelSet):
    dataMat=np.mat(dataSet)
    labelMat=np.mat(labelSet).transpose()
    #print(np.shape(labelSet))
    m,n=np.shape(dataMat);
    itr=500
    alpha=0.001
    weights=np.ones((1,n))
   # print(np.shape(weights))
    for i in range(itr):   
        h=sigmoid(dataMat*weights.T)  #矩阵相乘，np.multiply表示矩阵点乘
        err=h-labelMat
        weights=weights-alpha*(err.T*dataMat)
    return weights

def StoGradientDescent(dataSet,labelSet,maxIter=150):
    m,n=np.shape(dataSet)
    alpha=0.001
    weights=np.ones(n)
    for i in range(maxIter):
        #dataIndex=range(m)
        for j in range(m):
            randIndex=int(np.random.uniform(0,m))
            h=sigmoid(sum(dataSet[randIndex]*weights))
            err=(h-labelSet[randIndex])
            weights=weights-alpha*dataSet[randIndex]*err
            #del(dataIndex[randIndex])
    return weights

def batchGradientDescent(dataSet,labelSet,maxIter=300):
    m,n=np.shape(dataSet)
    alpha=0.001
    weights=np.ones(n)
    for i in range(0, maxIter):
        hypothesis = np.dot(dataSet, weights)
        loss = hypothesis - labelSet
        gradient = np.dot(dataSet.T, loss) / m
        weights = weights - alpha * gradient
    return weights

    

def plotBestLine(weights):
    import matplotlib.pyplot as plt
    dataSet,labelSet=loadData()
    dataArr=np.array(dataSet)
    labelArr=np.array(labelSet)
    m=np.shape(dataArr)[0]
    x1Record=[];y1Record=[]
    x2Record=[];y2Record=[]
    for i in range(m):
        if int(labelArr[i])==1:
            x1Record.append(dataArr[i,1]),y1Record.append(dataArr[i,2])
        else:
            x2Record.append(dataArr[i,1]),y2Record.append(dataArr[i,2])
            
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(x1Record,y1Record,s=30,c='red',marker='o')
    ax.scatter(x2Record,y2Record,s=30,c='green',marker='o')
    
    x=np.arange(-3.0,3.0,0.1).reshape(1,60)
   # print(np.shape(x))
   # print(np.shape(weights[0]))
    y=(-weights.T[0]-weights.T[1]*x)/weights.T[2]
    ax.plot(x,y,'bo')
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
    
if __name__=='__main__':
    dataMat,labelMat=loadData()
    dataArr = np.array(dataMat)
    weights = batchGradientDescent(dataArr,labelMat)
    plotBestLine(weights)


        
        
