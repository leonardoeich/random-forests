import pandas as pd
import random

def bootstrap(data):
    # the sample size will be the number of rows in the df
    sampleSize = data.shape[0]
    # replace=True determines that the same row can be selected more than once
    boot = data.sample(n=sampleSize, replace=True)

    return boot
