import sys
sys.path.append('/content/drive/My Drive/DataDeletion/toolbox')
from data import importMat
from util import zscore
from util import normalize
from NN import NeuralNetwork
import numpy as np
# cos
from sklearn.metrics.pairwise import cosine_similarity

def similarity(opt,x,y):
  option = {
      0:print("none")
      1:cos_sim(x,y),
      2:eudclidean_distance(x,y),    
  }
  return option[opt]

# def cos_sim(x,y):

#   if type(x)==list:
#     size = len(x)
#   else:
#     size = np.size(x)
#   cos = np.zeros(size)

#   # for i in range(0,size):
#   #   cos[i]=cosine_similarity(x[i],y[i])

#   for i in range(0,size):
#     dot = np.dot(x[i].T,y[i])
#     normx = np.linalg.norm(x[i])
#     normy = np.linalg.norm(y[i])
#     cos[i] = dot/(normx*normy)

#   print("cosine_similarity")
#   return cos

def eudclidean_distance(x,y):
  
  # size = np.size(x.W)
  # dist = 0
  # for i in range(0, size):
  #   dist += np.sqrt(sum( pow(a-b,2) for a,b in np.nditer([x.W[i],y.W[i]]))) 
  # # return 1/(1+dist)
  # return dist

  if type(x)==list:
    size = len(x)
  else:
    size = np.size(x)
  dist = 0
  for i in range(0, size):
    dist += np.sqrt(sum( pow(a-b,2) for a,b in np.nditer([x[i],y[i]],order='F'))) 
  # return 1/(1+dist)
  print("eudclidean_distance")
  return dist
