import pandas as pd
from node import Node
from math import log2


# calculate the entropy for the split in the dataset
def entropy(target_column, data_set):
  bits = 0
  classes = y.unique()  # get all unique values from a given column

  for classe in classes:
    class_prob = prob_of_class(target_column, classe, data_set)
    bits -= class_prob * log2(class_prob)

  return bits


def prob_of_class(target_column, class_name, data_set):
  # get all unique values from a given column
  y = data[target_column]
  prob = y.value_counts()[class_name]/len(y)
  # values = y.unique()
  # pd_series = pd.Series(y)
  pass

#select the attribute that will get the best split to a node
#def selectBestAttribute(target_column, training_rdata):
#  current_entropy = entropy(target_column, training_data)

def isOnlyOneClass(target_column, training_data):
  y = training_data[target_column]
  classes = y.unique()
  num_classes = len(classes)
  
  if (num_classes > 1):
    return False
  else:
    return True

def getMainClass(target_colum):
  pass


def decisionTreeClassifier(training_data, attributes, target_column):
  node = Node()

  #if all examples in the current dataset belong to the same class, return the node as a leaf
  if isOnlyOneClass(target_column, training_data):
    node.is_leaf = True
    node.label = 


def testfun():
  print("relou\n")
