'''
Date: 2022-07-19 15:33:48
LastEditors: ZSudoku
LastEditTime: 2022-07-19 15:37:12
FilePath: \Pistachio_DeepLearning\main.py
'''
from fileload import *
import numpy as np
import pandas as pd

data = pd.read_excel(io = dataset,sheet_name= 'Pistachio_28_Features_Dataset')
