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
from decisionTree import isOnlyOneClass
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

  data = pd.read_csv("./data/dadosBenchmark_validacaoAlgoritmoAD.csv", sep = ';')
  attribute_types = pd.read_csv("./data/dadosBenchmark_validacaoAlgoritmoAD_types.csv", sep="\t")

  
  data_dict = {}
  data_types = {}
  columns = list(data.columns.values)
  for attribute in columns:
    data_dict[attribute] = data[attribute].unique()
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

  #print(list(X.columns.values))
  #lista = y.unique()
  #print(y.value_counts()["Nao"])

  #for item in lista:
      #print(item)
      #print('aaa')
  #list(my_dataframe.columns.values)


  #df.word.value_counts()['myword']
  print(data)
  print(data_types)
  #print(data["Joga"].value_counts())
  #print(y.mode().iloc[0])
  #print(data["Tempo"].mode().iloc[0])
  #print(y.mode().iloc[0])
  #print ("testeee")
  #df1 = data[data['Joga'] == "Sim"]
  #print(df1)
  #print (isOnlyOneClass(df1, "Joga"))

  #print(data['Joga'].value_counts()[:1].index.tolist()[0])
  #print(data.columns.values.tolist())
  
  # randomly selecting the square root of #attributes
  #attributes = list(data.columns)
  #attributes.remove('Temperatura')
  #print(attributes)
  #attributes.remove('Temperatura')
  #print(data_dict)




main()
