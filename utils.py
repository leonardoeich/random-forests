import pandas as pd
import random
from math import sqrt

def get_unique_values(dataset, target_column):
  y = training_data[target_column]
  unique_values = y.unique()
  return unique_values

def prob_of_class(target_column, class_name, data_set):
  # get all unique values from a given column
  y = data_set[target_column]
  classes = y.unique()  # get all unique values from a given column

  prob = y.value_counts()[class_name]/len(y)
  # values = y.unique()
  # pd_series = pd.Series(y)
  return prob

# calculate the entropy for the split in the dataset
def entropy(target_column, data_set):
  bits = 0
  y = data_set[target_column]
  classes = y.unique()  # get all unique values from a given column

  for classe in classes:
    class_prob = prob_of_class(target_column, classe, data_set)
    bits -= class_prob * log2(class_prob)

  return bits
  

def bootstrap(data):
  # the sample size will be the number of rows in the df
  sampleSize = data.shape[0]
  # replace=True determines that the same row can be selected more than once
  boot = data.sample(n=sampleSize, replace=True)

  return boot

def information_gain(current_entropy, new_entropy):
  return current_entropy - new_entropy

def attributeSelection(training_data, attributes_list, target_column, current_entropy):
  # randomly selecting the square root of #attributes
  m = sqrt(len(attributes_list))
  header = random.sample(attributes_list, round(m))
  data_length = len(training_data)
  information_gain_values = []
  
  #go through each attribute and calculate the entropy 
  for attribute in header:
    #get each possible value for the given attribute
    unique_values = training_data[attribute].unique()

    attribute_info = 0
    #for each possible value of a given attribute, calculate its entropy
    for value in unique_values:
      #get all rows where the current attribute is equal to the current value
      value_partition = training_data[training_data[attribute] == value]
      #multiply entropy of current attribute value for the partition weight
      attribute_info += (len(value_partition)/data_length) * entropy(target_column, value_partition)

    #save information gain of the current attribute
    information_gain_values.append(information_gain(current_entropy, attribute_info))

  #print(header[entropyValues.index(min(entropyValues))])
  
  #return the attribute that maximizes the information gain
  return header[information_gain_values.index(max(information_gain_values))]
