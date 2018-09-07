# -*- coding: utf-8 -*-
"""
Created on Thu May 24 19:32:44 2018

@author: asus
"""
import numpy as np
def loadDataSet(filename): #读取数据
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat #返回数据特征和数据类别

def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(np.random.uniform(0,m))
    return j

def clipAlpha(alpha,H,L):
    if alpha>H:
        alpha=H
    if alpha<L:
        alpha=L
    return alpha


    
def SMO(dataMat, classlabels, C,toler, maxIter):
    dataMatrix=np.mat(dataMat)
    labelMat=np.mat(classlabels).transpose
    b=0
    m,n=dataMatrix.shape()
    alphas=np.mat(np.zeros(m,1))
    itr=0
    while(itr<maxIter):
        alphaPairsChanged=0
        for i in range(m):
            fxi=float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            ei=fxi-float(labelMat[i])
            if ((labelMat[i]*ei<-toler) and (alphas[i]<C)) or \
            ((labelMat[i]*ei>toler) and (alphas[i]>0)):
                j=selectJrand(i,m)
                fxj=float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                ej=fxj-float(labelMat[j])
                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()
                if labelMat[i]!=labelMat[j]:
                    L=np.max(0,alphas[j]-alphas[i])
                    H=np.min(C,C+alphas[j]-alphas[i])
                else:
                    L=np.max(0,alphas[j]+alphas[i]-C)
                    H=np.min(C,alphas[j]+alphas[i])
                if L==H:
                    print("L=H")
                    continue
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T\
                -dataMatrix[i,:]*dataMatrix[i,:].T\
                -dataMatrix[j,:]*dataMatrix[j,:].T
                if(eta>=0):
                    print("eta>=0")
                    continue
                alphas[j]-=labelMat[j]*(ei-ej)/eta
                alphas[j]=clipAlpha(alphas[j],H,L)
                if(abs(alphas[j]-alphaJold)<0.00001):
                    print("alpha not moving enough")
                    continue
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                b1=b-ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T\
                -labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T\
                -labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if(alphas[i]>0) and (alphas[i]<C): 
                    b=b1
                elif(alphas[j]>0) and (alphas[j]<C): 
                    b=b2
                else:
                    b=(b1+b2)/2.0
                alphaPairsChanged+=1
        if(alphaPairsChanged==0):
            itr+=1
        else:
            itr=0
    return b, alphas
                    
                
                
                
                    
        
    
    