'''
Date: 2022-10-04 15:19:03
LastEditors: ZSudoku
LastEditTime: 2022-10-04 18:13:54
FilePath: \Pistachio_DeepLearning\testGPU.py
'''
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))
