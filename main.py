import sys
import pandas as pd
from node import Node
from math import log2
from math import sqrt
import random
import numpy as np

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


# this is for importing the decisionTree file and
# be able to execute its functions from here
from decisionTree import *
from utils import bootstrap
from utils import attributeSelection

def main():
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
  #y = data["Joga"]
  #columns.remove('Joga')
  tree = decisionTreeClassifier(data, columns, 'target', data_dict, data_types)
  print_tree(tree)
  

  #testes = [['Ensolarado', 'Quente', 'Alta', 'Falso'],
  #          ['Ensolarado', 'Fria', 'Normal', 'Falso']]
  
  #a = data_dict['Tempo']
  #print(a)
  #print(data)
  #result = predict(tree, testes[1], list(data.columns.values), data_dict)
  #print(result)
  #print(data["1"])
  
main()
