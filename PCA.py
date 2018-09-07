# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 16:22:50 2018

@author: asus
"""



import numpy as np
def loadDataSet(fileName,delim='\t'):
    fr=open(fileName)
    stringArr=[line.strip().split(delim) for line in fr.readlines()]
    #datArr=[map(lambda x: float(x),line) for line in stringArr]
    datArr=[[float(stringArr[i][j]) for j in range(len(stringArr[i]))] for i in range(len(stringArr))]
   
    return np.mat(datArr)

def pca(dataMat,topNfeat=9999999):
    meanVals=np.mean(dataMat,axis=0)
    zeroMean=dataMat-meanVals
    covMat=np.cov(zeroMean,rowvar=0)
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))
    eigValInd=np.argsort(eigVals)
    eigValInd=eigValInd[:-(topNfeat+1):-1]
    redEigVects=eigVects[:,eigValInd]
    lowdDataMat=zeroMean*redEigVects
    reconMat=(lowdDataMat*redEigVects.T)+meanVals
    return lowdDataMat,reconMat


if __name__=='__main__':
  from numpy import *
  import matplotlib
  import matplotlib.pyplot as plt

  
  dataMat = loadDataSet('testSet.txt')
  lowDMat, reconMat = pca(dataMat, 1)

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(dataMat[:,0].tolist(), dataMat[:,1].tolist(), marker='^', s=90)
  ax.scatter(reconMat[:,0].tolist(), reconMat[:,1].tolist(), marker='o', s=50, c='red')
  plt.show()