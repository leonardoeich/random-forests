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

# check if all values are of an attribute are in the current data
def all_values_exist(current_values, possible_values):
  exists = True
  for value in possible_values:
    if value not in current_values:
      exists = False
  return exists

def print_tree(node, level=0):
  node_rep = node.print_node()
  print('\t' * level + node_rep)
  for child in node.children:
    print_tree(child, level+1)
  
# Make a prediction with a decision tree
def predict(node, row, attributes, data_dict, data_types):
  if node.is_leaf:
    return node.label
  else:
    #get the index of the attribute used by the node to split the tree
    attribute_index = attributes.index(node.attribute)
    #get the attribute value in the node we're trying to predict
    attribute_value = row[attribute_index]

    if is_attribute_numeric(node.attribute, data_types):
      child_index = 0 if (attribute_value <= node.numeric_value) else 1
    else:
      child_index = data_dict[node.attribute].index(attribute_value)
    #go to the node's son the matches attribute value
    return predict(node.children[child_index], row, attributes, data_dict, data_types)


def decisionTreeClassifier(training_data, attributes, target_column, data_dict, data_types, attribute_value=None):
  node = Node()
  node.entropy = entropy(target_column, training_data)
  node.attribute_value = attribute_value

  # If all examples in the current dataset belong to the same class
  # or the attribute list is empty, return the node as a leaf
  if isBaseCase(training_data, target_column, attributes):
    node.is_leaf = True
    node.label = getMajorityClass(training_data, target_column)
  # Select attribute with best split criteria (information gain)
  else:
    best_attribute = attributeSelection(training_data, attributes, target_column, node.entropy, data_types)
    node.attribute = best_attribute
    # remove used attribute from attributes list
    attributes.remove(best_attribute)
    
    if is_attribute_numeric(best_attribute, data_types):
      mean = training_data[best_attribute].mean()
      node.numeric_value = mean
      partition_less = training_data[training_data[best_attribute] <= mean]
      partition_greater = training_data[training_data[best_attribute] > mean]
      node.children.append(decisionTreeClassifier(partition_less, attributes, target_column, data_dict, data_types, mean))
      node.children.append(decisionTreeClassifier(partition_greater, attributes, target_column, data_dict, data_types, mean))
    else:
      attribute_values = get_unique_values(training_data, best_attribute)
      #check if all attributes values are represented
      if all_values_exist(attribute_values, data_dict[best_attribute]):
        for value in attribute_values:
          value_sub_set = training_data[training_data[best_attribute] == value]
          node.children.append(decisionTreeClassifier(value_sub_set, attributes, target_column, data_dict, data_types, value))
      else:
        node.is_leaf = True
        node.label = getMajorityClass(training_data, target_column)
      
  return node