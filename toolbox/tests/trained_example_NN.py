# 新的nn.W
import sys
sys.path.append('/content/drive/My Drive/DataDeletion/toolbox')
from data import importMat
from util import zscore
from util import normalize
from NN import NeuralNetwork,nntrain,nntest
# ,nndelete
import numpy as np
def trained_NN(nn):
  dataSet = importMat.loadMatFile()
  if dataSet is not None:
      keys = dataSet.keys()
      for key in keys:
          if key == "train_x":
              train_x = dataSet[key]
              train_x = train_x/255.0
          elif key == "train_y":
              train_y = dataSet[key]
              train_y = train_y
          elif key == "test_x":
              test_x = dataSet[key]
              test_x = test_x/255.0
          elif key == "test_y":
              test_y = dataSet[key]
              test_y = test_y            
          
  train_x, mu, sigma = zscore.zscore(train_x)
  test_x =  normalize.normalize(test_x,mu,sigma)
      
  # ex1 vanilla nerual net
  # np.random.seed(1) # Setting the random seed so that the weights are generated same as in matlab code - default is twister algorithm        
  # nn = NeuralNetwork.NN(np.array([784, 100 ,10]))
  
  # numepochs = Number of full sweep through data
  # batchsize = Mean gradient step over this many samples
  options = {"numepochs":5,"batchsize":100,"plot":1}
  figureNo = 1
  val_x = None
  val_y = None

  nn,L = nntrain.nntrain(nn,train_x,train_y,options,val_x,val_y,figureNo)
  er,bad = nntest.nntest(nn,test_x,test_y)
  assert (er < 0.08),"Too big error"
  
  print("train_model_er",er) 
  print("train finish") 

  return nn,train_x,train_y,options,val_x,val_y,figureNo,test_x,test_y 

  