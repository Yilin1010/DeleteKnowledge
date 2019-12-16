
import sys
sys.path.append('/content/drive/My Drive/DataDeletion/toolbox')
from data import importMat
from util import zscore
from util import normalize
from NN import NeuralNetwork , nnpredict
import numpy as np

def confusion_matrix(nn,x,y):
  numlabel=y.shape[1]
  cn = np.zeros([numlabel,numlabel])
  totallabel= np.zeros(numlabel)  
  actual = y.argmax(axis=1)   
  predicted = nnpredict.nnpredict(nn,x)

  for i,j in zip(actual,predicted):
    cn[i-1][j-1] +=1
    totallabel[i-1] +=1
  for i in range(numlabel):
    cn[i] /=totallabel[i]
  return cn