import pandas as pd
import random
from math import sqrt
from decisionTree import entropy

def bootstrap(data):
    # the sample size will be the number of rows in the df
    sampleSize = data.shape[0]
    # replace=True determines that the same row can be selected more than once
    boot = data.sample(n=sampleSize, replace=True)

    return boot

def attributeSelection(data):
    # we depend on the entropy calculation, so for now I'll try
    # to work it around
    header = list(data.columns)
    m = sqrt(len(header))
    # randomly selecting the square root of #attributes
    header = random.sample(header, round(m))
    print(header)

    entropyValues = [2, 4]
    #for i in range(0, round(m)):
        #entropyValues[i] = entropy(header[i], data)

    print(header[entropyValues.index(max(entropyValues))])
