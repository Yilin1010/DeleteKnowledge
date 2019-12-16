
import numpy as np
from numpy import double, float64
from NN import nnff, nnbp, nnapplygrads, nneval, nnupdatefigures
from time import time
from NN import Collections,NeuralNetwork
import matplotlib.pyplot as plt
from matplotlib.pyplot import ioff

def reversennapplygrads(nn):
  dW = None;
  for i in range(0,nn.N-1):
      dW = nn.DW[i]  
      dW = np.dot(nn.LearningRate,dW)       
      nn.W[i] = nn.W[i] + dW
  return nn 

def nndelete(nn, xi, train_x, train_y, opts,val_x,val_y,figureNo,ifepochs):             

    deleted_x=train_x[xi]
    deleted_y=train_y[xi]


    if ifepochs == 1:
      numepochs = 5 # opts["numepochs"]
    else:
      numepochs = 1
    
    batchsize = opts["batchsize"]
    numepochs = opts["numepochs"]
    loss = Collections.Loss(numepochs)  
    
    L = np.zeros(shape=(numepochs*1,1),dtype=np.float64)
    
    nn = nnff.nnff(nn, deleted_x, deleted_y)
# xiyi nnbp
    n = nn.N
    d = [None for i in range(n)]

    if nn.Output == "sigm":
        temp2 = 1- nn.A[n-1]    
        d[n-1] = - np.multiply(nn.E,np.multiply(nn.A[n-1],temp2))

    for i in range(n-2,0,-1):
        # Derivative of the activation function
        if nn.ActivationFunction == "sigm":
            temp1 = 1
            temp = np.subtract(temp1,nn.A[i]) 
            d_act = np.multiply(nn.A[i],temp)
        elif nn.ActivationFunction == 'tanh_opt':
            t1 = 1.7159*(2/3)
            t2 = 1/(1.7159*1.7159)
            d_act = t1*(1-(t2*np.power(nn.A[i],2)))    
          
        # Backpropagate first derivatives
        if (i+1) == (n-1): #In this case in d[n-1] there is no bias term to be removed
            d[i] = np.multiply(np.dot(d[i+1],nn.W[i]),d_act)
        else: # In this case in d[i] the bias term has to be removed
            temp = d[i+1][:,1:]
            d[i] = np.multiply(np.dot(temp,nn.W[i]),d_act)
            
    
    # Allocate space for activation, error and loss
    # Since matlab allows structs to have cells/arrays etc on the fly, 
    # This is not possible with python. We declare members and then use them    
    #nn.A[0] = np.copy(temp);       
    
    for i in range (0,n-1):
        if (i+1) == (n-1):
            term = d[i+1]
            nn.DW[i] = np.dot(term.T,nn.A[i])/d[i+1].shape[0]
        else:
            term = d[i+1][:,1:]            
            nn.DW[i] = np.dot(term.T,nn.A[i])/d[i+1].shape[0]
            
## xiyi 
        nn = reversennapplygrads(nn)
        
        L[n] = nn.L         
        # toc = time()
        # t = toc-tic
# 2        
        # str_pref = ""
        # if opts["validation"] == 1:
        #     loss = nneval.nneval(nn,loss,train_x,train_y,val_x,val_y,i)
        #     str_pref = (":Full batch train mse = {0},val mse = {1}".format(loss.Training.E[i],loss.Validity.E[i]))
        # else:
        #     loss = nneval.nneval(nn,loss,train_x,train_y,val_x,val_y,i)
        #     str_pref = " Full-batch train err = {0}".format(loss.Training.E[i])
        #     nn.LearningRate = nn.LearningRate*nn.ScalingLearningRate
        # if "plot" in opts and opts["plot"]== 1 and opts["numepochs"] > 1:    
        #    nnupdatefigures.nnupdatefigures(nn, figureNo, loss, opts, i)
               
        # print("epoch " , i+1 ,"/" , opts["numepochs"], ". Took " ,t, " seconds. error on training set is " , L[0] ,str_pref)
        
        nn.LearningRate = nn.LearningRate * nn.ScalingLearningRate;

    return nn,L

  