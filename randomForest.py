import numpy as np
from decisionTree import *
from utils import *


def randomForest(n_trees, data, attributes, target_column, data_dict, data_types, max_depth, k=10):
  #split data into k partitions 
  folds = generate_k_folds(data, target_column, k)
  accuracy_list = []
  mean_accuracy = 0
  #for each partition i, train n trees using k-1 partitions and test it with partition i 
  for i in range(k):
    test_data = folds[i].copy()
    train_partitions = folds[:i] + folds[i+1:]
    train_data = concatenate_partitions(train_partitions)
  
    #create n trees 
    random_forest = createRandomForest(n_trees, train_data, attributes.copy(), target_column, data_dict, data_types, max_depth, k)

    #get labels predicted by n_trees
    predicted_labels = ensemble(random_forest, test_data.values.tolist(), attributes, target_column, data_dict, data_types)
    #get accuracy for the current forest
    forest_accuracy = accuracy(predicted_labels, test_data[target_column].tolist())

    print("ACCURACY AT PARTITION " + str(i) + ": " + str(forest_accuracy))
    accuracy_list.append(forest_accuracy)
    mean_accuracy += forest_accuracy
  return (mean_accuracy/len(accuracy_list))

#creates a list of decision trees(a Random Forest)
def createRandomForest(n_trees, train_data, attributes, target_column, data_dict, data_types, max_depth, k=10):
  #create n trees
  decision_trees = []
  for n in range (n_trees):
    boot_data = bootstrap(train_data)
    decision_tree = decisionTreeClassifier(boot_data, attributes.copy(), target_column, data_dict, data_types, max_depth)
    decision_trees.append(decision_tree)
  return decision_trees


def ensemble(random_forest, test_data, attributes, target_column, data_dict, data_types):
  predicted_labels = [] #labels predicted by the votes of the forest
  for row in test_data:
    forest_labels = [] #labels predicted by each tree
    for tree in random_forest:
      prediction = predict(tree, row, attributes, data_dict, data_types)
      forest_labels.append(prediction)
    #forest prediction is the most voted label
    forest_prediction = most_frequent(forest_labels)
    predicted_labels.append(forest_prediction)
  return predicted_labels


def accuracy(predicted_labels, real_labels):
  correct_predictions = 0
  for i in range(len(predicted_labels)):
    if (predicted_labels[i] == real_labels[i]):
      correct_predictions += 1
  return (correct_predictions/len(predicted_labels))