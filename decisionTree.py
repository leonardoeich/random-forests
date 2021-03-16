import pandas as pd
from node import Node 
from math import log2

# 1. get data ready
data = pd.read_csv("data/dadosBenchmark_validacaoAlgoritmoAD.csv", sep = ';')
#print(data)

# Create X (features matrix)
X = data.drop("Joga", axis = 1)

# create Y (labels)
y = data["Joga"]

	
# calculate the entropy for the split in the dataset
def entropy(target_column, data_set):
  bits = 0
  classes = y.unique() #get all unique values from a given column

  for classe in classes:
    class_prob = prob_of_class(target_column, classe, data_set)
    bits -= class_prob * log2(class_prob)
  return bits
  
def prob_of_class(target_column, class_name, data_set):
  #get all unique values from a given column
  y = data[target_column]
  prob = y.value_counts()[class_name]/len(y)
  #values = y.unique()
  #pd_series = pd.Series(y)
  pass



def decisionTreeClassifier(trainingData, attributes):
  pass

#print(list(X.columns.values))
lista = y.unique()
#print(y.value_counts()["Nao"])

for item in lista:
  print(item)
  print('aaa')
#list(my_dataframe.columns.values)


#df.word.value_counts()['myword']
