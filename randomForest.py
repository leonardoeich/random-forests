import numpy as np
from decisionTree import *
from utils import *


def randomForest(n_trees, training_data, attributes, target_column, data_dict, data_types, max_depth, k=10):
  folds = generate_k_folds(training_data, attributes, k)
  for i in range(k):
    decisionTrees = []
    test_data = folds[i].copy()
    train_partitions = test_data[:i] + test_data[i+1:]
    train_data = concatenate_partitions(train_partitions)
    
    for n in range (n_trees):
      boot_data = bootstrap(train_data)
      decision_tree = decisionTreeClassifier(boot_data, attributes.copy(), target_column, data_dict, data_types, max_depth)
      decisionTrees.append(decision_tree)
  

    #decisionTrees.append()
  #return decisionTrees

