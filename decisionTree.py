import pandas as pd
from node import Node
from math import log2


# calculate the entropy for the split in the dataset
def entropy(target_column, data_set):
  bits = 0
  classes = y.unique() #get all unique values from a given column

  for classe in classes:
    class_prob = prob_of_class(target_column, classe, data_set)
    bits -= class_prob * log2(class_prob)
  return bits

def prob_of_class(target_column, class_name, data_set):
  #get all unique values from a given column
  y = data[target_column]
  prob = y.value_counts()[class_name]/len(y)
  #values = y.unique()
  #pd_series = pd.Series(y)
  pass

def decisionTreeClassifier(trainingData, attributes):
  pass

def testfun():
    print("relou\n")
