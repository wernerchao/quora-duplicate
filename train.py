# training
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pal = sns.color_palette()

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

print('done')