import pandas as pd
from node import Node
from math import log2
# this is for importing the decisionTree file and
# be able to execute its functions from here
from decisionTree import testfun

def main():

    # and here I'm simply adding an exemple for you to see,
    # but feel free to remove it
    testfun()

    # 1. get data ready
    data = pd.read_csv("./data/dadosBenchmark_validacaoAlgoritmoAD.csv", sep = ';')
    #print(data)

    # Create X (features matrix)
    X = data.drop("Joga", axis = 1)

    # create Y (labels)
    y = data["Joga"]

    #print(list(X.columns.values))
    lista = y.unique()
    #print(y.value_counts()["Nao"])

    for item in lista:
        #print(item)
        print('aaa')
    #list(my_dataframe.columns.values)


    #df.word.value_counts()['myword']

main()
