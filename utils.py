import pandas as pd
import random
from math import sqrt
from math import log2

# Get the unique values of an attribute, and it's used to create the child nodes.
def get_unique_values(dataset, target_column):
  y = dataset[target_column]
  unique_values = y.unique()
  return unique_values

# Calculates the class' probability.
# This information is used on the entropy calculation.
def prob_of_class(target_column, class_name, data_set):
  # get all unique values from a given column
  y = data_set[target_column]
  classes = y.unique()  # get all unique values from a given column

  prob = y.value_counts()[class_name]/len(y)
  # values = y.unique()
  # pd_series = pd.Series(y)
  return prob

# Calculate the entropy for the split in the dataset
def entropy(target_column, data_set):
  bits = 0
  y = data_set[target_column]
  classes = y.unique()  # get all unique values from a given column

  for classe in classes:
    class_prob = prob_of_class(target_column, classe, data_set)
    bits -= class_prob * log2(class_prob)

  return bits

# Bootstrap for selecting subsets from the training dataset.
# Each bootstrap is applied on the training phase of a given tree.
def bootstrap(data):
  # the sample size will be the number of rows in the df
  sampleSize = data.shape[0]
  # replace=True determines that the same row can be selected more than once
  boot = data.sample(n=sampleSize, replace=True)

  return boot

# The information gain.
# Used in the best attribute selection.
def information_gain(current_entropy, new_entropy):
  return current_entropy - new_entropy

def is_attribute_numeric(attribute, data_types):
  is_numeric = True if (data_types[attribute] == "num") else False
  return is_numeric

#def calculate_numeric_info():

#def calculate_categoric_info():

def calculate_attribute_info(attribute, training_data, data_types, target_column):
  attribute_info = 0
  data_length = len(training_data)
  
  if is_attribute_numeric(attribute, data_types):
    mean = training_data[attribute].mean()
    partition_less = training_data[training_data[attribute] <= mean]
    partition_greater = training_data[training_data[attribute] > mean]
    attribute_info += (len(partition_less)/data_length) * entropy(target_column, partition_less)
    attribute_info += (len(partition_greater)/data_length) * entropy(target_column, partition_greater)
  else:
    #get each possible value for the given attribute
    attribute_values = training_data[attribute].unique()
    #for each possible value of a given attribute, calculate its entropy
    for value in attribute_values:
      #get all rows where the current attribute is equal to the current value
      value_partition = training_data[training_data[attribute] == value]
      #multiply entropy of current attribute value for the partition weight
      attribute_info += (len(value_partition)/data_length) * entropy(target_column, value_partition)

  return attribute_info
  


# Selection of the best attribute considering its information gain value.
# The attribute with the highest entropy is the one selected to be the current node.
def attributeSelection(training_data, attributes_list, target_column, current_entropy, data_types):
  # randomly selecting the square root of #attributes
  m = sqrt(len(attributes_list))
  header = random.sample(attributes_list, round(m))
  data_length = len(training_data)
  information_gain_values = []

  #go through each attribute and calculate the entropy
  for attribute in header:
    '''
    #get each possible value for the given attribute
    attribute_values = training_data[attribute].unique()

    attribute_info = 0
    #for each possible value of a given attribute, calculate its entropy
    for value in attribute_values:
      #get all rows where the current attribute is equal to the current value
      value_partition = training_data[training_data[attribute] == value]
      #multiply entropy of current attribute value for the partition weight
      attribute_info += (len(value_partition)/data_length) * entropy(target_column, value_partition)
    '''

    #save information gain of the current attribute
    #information_gain_values.append(information_gain(current_entropy, attribute_info))
    attribute_info = calculate_attribute_info(attribute,training_data,data_types,target_column)
    attribute_gain = information_gain(current_entropy, attribute_info)
    information_gain_values.append(attribute_gain)

  #print(header[entropyValues.index(min(entropyValues))])

  #return the attribute that maximizes the information gain
  return header[information_gain_values.index(max(information_gain_values))]
