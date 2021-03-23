import pandas as pd
from node import Node
from math import log2
from utils import attributeSelection
from utils import entropy


def isOnlyOneClass(training_data, target_column):
  y = training_data[target_column]
  classes = y.unique()
  num_classes = len(classes)
  
  if (num_classes > 1):
    return False
  else:
    return True

def getMajorityClass(training_data, target_colum):
  y = training_data[target_colum]
  #get most frequent element in a pandas series
  return (y.mode().iloc[0])


def decisionTreeClassifier(training_data, attributes, target_column):
  node_entropy = entropy(target_column, training_data)
  node = Node()

  #if all examples in the current dataset belong to the same class, return the node as a leaf
  if isOnlyOneClass(training_data, target_column):
    node.is_leaf = True
    node.label = getMajorityClass(training_data, target_column)
    return node
  #if the attribute list is empty, return the majority class of the training data
  elif (len(attributes) == 0):
    return getMajorityClass(training_data, target_column)

  #select attribute with best split criteria
  else:
    best_attribute = attributeSelection(training_data, attributes, target_column, node_entropy)
    attributes.remove(best_attribute) #remove the used attribute from attributes list
    #TODO
  

def testfun():
  print("relou\n")
