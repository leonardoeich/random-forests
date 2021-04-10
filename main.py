import sys
import pandas as pd
from node import Node
from math import log2
from math import sqrt
import random
import numpy as np
import random
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


# this is for importing the decisionTree file and
# be able to execute its functions from here
from decisionTree import *
from utils import bootstrap
from utils import attributeSelection

def main():
  #this is used to make sure all results are reproducible
  np.random.seed(42)
  # 1. get data ready
  #if (len(sys.argv) != 3):
    #print("Missing arguments! Execute:\npython3 main.py <file_name> <separator> <attribute_type_file> <separator>")
    #print("Missing arguments! Execute:\npython3 main.py data /t data_attributes ;")
  
  #    exit()

  #finput = sys.argv[1]
  #types_file = sys.argv[2]

  #data = pd.read_csv("./data/dadosBenchmark_validacaoAlgoritmoAD.csv", sep = ';')
  #attribute_types = pd.read_csv("./data/dadosBenchmark_validacaoAlgoritmoAD_types.csv", sep="\t")
  data = pd.read_csv("./data/wine_recognition.tsv", sep = '\t')
  attribute_types = pd.read_csv("./data/wine_recognition_types.tsv", sep="\t")

  # Create X (features matrix)
  #X = data.drop("Joga", axis = 1)

  data_dict = {}
  data_types = {}
  columns = list(data.columns.values)
  for attribute in columns:
    data_dict[attribute] = list(data[attribute].unique())
    data_types[attribute] = attribute_types[attribute].unique()[0]
  
  
  

  #data = pd.read_csv(finput, sep = ';')
  #data = pd.read_csv("./data/wine_recognition.tsv", sep = ';')
  #print(data)
  #boot = bootstrap(data)
  #print(boot)

  # given the generated boot, now we need to sample m attributes
  # based on information gain to train our DT
  #attributeSelection(boot)

  # Create X (features matrix)
  #X = data.drop("Joga", axis = 1)
  # create Y (labels)
  #y = data['target'] 
  ''''
  columns.remove('target')
  tree = decisionTreeClassifier(data, columns, 'target', data_dict, data_types, 10)
  a = [[1,2,3,4], [1,2,3,4], [1,2,3,4], [1,2,3,4]]
  b = a[:2] + a[5:]
  print(a)
  print(b)
  '''
  folds = generate_k_folds(data, 'target', 10)
  for i in range(10):
    print(folds[i])
  
main()
