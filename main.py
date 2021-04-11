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
from utils import *
from randomForest import *

def main():
  #this is used to make sure all results are reproducible
  np.random.seed(42)
  
  # 1. get data ready
  if (len(sys.argv) != 8):  
    print("Missing arguments! Execute:\npython3 main.py <file_name> <$'separator'> <attribute_type_file> <$'separator'> <target> <ntrees> <max_depth>")
    exit()
  
  data_file = sys.argv[1]
  data_sep = sys.argv[2]
  types_file = sys.argv[3]
  types_sep = sys.argv[4]
  target = sys.argv[5]
  ntrees = int(sys.argv[6])
  max_depth = int(sys.argv[7])

  #read files
  data = pd.read_csv(data_file, sep = data_sep)
  attribute_types = pd.read_csv(types_file, sep = types_sep)
  
  data_dict = {}
  data_types = {}
  columns = list(data.columns.values)
  for attribute in columns:
    data_dict[attribute] = list(data[attribute].unique())
    data_types[attribute] = attribute_types[attribute].unique()[0]
  
  ac = randomForest(ntrees, data, columns, target, data_dict, data_types, max_depth, 10)
  print("MEAN ACCURACY = " + str(ac))
  

main()
