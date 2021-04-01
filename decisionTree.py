import pandas as pd
from node import Node
from math import log2
from utils import *

# The verification of whether all nodes belong to the same class.
# Used on the tree generation.
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

# We call "base case" the case on which all nodes belong to the same class
# or the attribute list is empty
def isBaseCase(training_data, target_column, attributes):
  if (isOnlyOneClass(training_data, target_column)) or (len(attributes) == 0):
    return True
  else:
    return False

def decisionTreeClassifier(training_data, attributes, target_column):
  node = Node()
  node_entropy = entropy(target_column, training_data)

  # If all examples in the current dataset belong to the same class
  # or the attribute list is empty, return the node as a leaf
  if isBaseCase(training_data, target_column, attributes):
    node.is_leaf = True
    node.label = getMajorityClass(training_data, target_column)
    return node
  # Select attribute with best split criteria (information gain)
  else:
    best_attribute = attributeSelection(training_data, attributes, target_column, node_entropy)
    node.attribute = best_attribute
    attributes.remove(best_attribute) # remove used attribute from attributes list
    # for each unique value of the selected attribute, 
    attribute_values = get_unique_values(training_data, best_attribute)
    for attribute_value in attribute_values:
      value_sub_set = training_data[training_data[best_attribute] == attribute_value]
      if len(value_sub_set) > 0:
        node.branches.append(decisionTreeClassifier(value_sub_set, attributes, target_column))
      else:
        ##TODO
        pass
