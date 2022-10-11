'''
Date: 2022-09-28 15:04:05
LastEditors: ZSudoku
LastEditTime: 2022-10-04 19:18:05
FilePath: \Pistachio_DeepLearning\diyModels.py
'''
# %%
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# coding=utf-8
# get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from keras.layers.core import Flatten

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Embedding, SpatialDropout1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout,LSTM,BatchNormalization,Conv1D,MaxPool1D,Reshape,Input
from tensorflow.keras.optimizers import SGD
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
import re
import torch
from keras.utils.vis_utils import plot_model


# %%
dataset = pd.read_csv('./Pistachio_Dataset/Pistachio_28_Features_Dataset/Pistachio_28_Features_Dataset.csv') 
cols = ['Class','Area', 'Perimeter', 'Major_Axis', 'Minor_Axis', 'Eccentricity',
        'Eqdiasq', 'Solidity', 'Convex_Area', 'Extent', 'Aspect_Ratio',
        'Roundness', 'Compactness', 'Shapefactor_1', 'Shapefactor_2',
        'Shapefactor_3', 'Shapefactor_4', 'Mean_RR', 'Mean_RG', 'Mean_RB',
        'StdDev_RR', 'StdDev_RG', 'StdDev_RB', 'Skew_RR', 'Skew_RG', 'Skew_RB',
        'Kurtosis_RR', 'Kurtosis_RG', 'Kurtosis_RB']
# dataset = dataset.loc[:,cols]
# d = {'Class': dataset['Class'].value_counts().index, 'count': dataset['Class'].value_counts()}
# df_Class = pd.DataFrame(data=d).reset_index(drop=True)
# df_Class

dataset.sample(10)

# %%
dataset = shuffle(dataset)
Lisclass = dataset['Class'].values
LisClassNp = []
for i in range(len(Lisclass)):
    if (Lisclass[i] == 'Kirmizi_Pistachio'):
        LisClassNp.append([0,1])
    else:
        LisClassNp.append([1,0])
Y = np.array(LisClassNp)
dataset = dataset.drop(columns='Class')
X = dataset.values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
#拆分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=40)
X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])

# %%
# d = {'Class': dataset['Class'].value_counts().index, 'count': dataset['Class'].value_counts()}
# df_Class = pd.DataFrame(data=d).reset_index(drop=True)
# df_Class 


# %%
#X.dtype = 'int64'
# Y.dtype = 'int64'
# X = X.astype(int)

# X = X.astype(int)

# %%
print(X.shape)
print(Y.shape)

# %%
import os,sys
import shutil

shutil.rmtree('models')  
os.mkdir('models')

# %%
# import tensorflow as tf
# def metric_precision(y_true,y_pred):
#     TP=tf.reduce_sum(y_true/tf.round(y_pred))
#     TN=tf.reduce_sum((1-y_true)(1-tf.round(y_pred)))
#     FP=tf.reduce_sum((1-y_true)tf.round(y_pred))
#     FN=tf.reduce_sum(y_true(1-tf.round(y_pred)))
#     precision=TP/(TP+FP)
#     return precision

# %%
def accsorce(Y_test,yhat):
    sorce = 0
    for i in range(len(Y_test)):
        sorce += roc_auc_score(Y_test[i], yhat[i])

    return sorce/len(Y_test)

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=40)
X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])

# %%
print(type(X_train))

# %%
# X_train = torch.from_numpy(X_train)
# Y_train = tuple(map(tuple, Y_train))

# %%
# def Input():
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=40)
#     X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
#     X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
#     X = np.array(X)
#     return torch.from_numpy(X)

# %%
class my_model(Model):
    def __init__(self, input_shape, **kwargs):
        super().__init__(**kwargs) 
        
        the_units = 128
        self.CNN   = Conv1D(the_units,1,padding="valid",activation="relu")
        self.POLL  = MaxPool1D(1)
        self.FLATT = Flatten()
        self.LSTM0 = LSTM(the_units,return_sequences=True,
                        kernel_initializer=tf.keras.initializers.glorot_normal(seed=2),bias_initializer=tf.keras.initializers.Zeros())
        self.LSTM1 = LSTM(the_units,return_sequences=True)
        self.LSTM2 = LSTM(the_units)
        self.BAT   = BatchNormalization()
        self.DROP  = Dropout(rate=0.3)
        self.DENS  = Dense(8, activation='relu')
        self.OUT   = Dense(2, activation='softmax')
        self.resha = Reshape((1, 28))
        
        # 模型的参数
        self.input_layer = tf.keras.layers.Input(input_shape)
        self.out = self.call(self.input_layer)
        
        super().__init__( inputs=self.input_layer,outputs=self.out, **kwargs)
    
    # 初始化模型的参数
    def build(self):
        self._is_graph_network = True
        self._init_graph_network(inputs=self.input_layer,outputs=self.out)
    
    def call(self, inputdata, **kwargs):
                
        #x1 = self.LSTM0(inputdata)
        print('inputdata',inputdata)
        x1 = self.CNN(inputdata)
        #x1 = self.LSTM0(inputdata)
        x1 = self.POLL(x1)
        c = self.DROP(x1)
        #c = self.BAT(x1)

        c = self.LSTM1(c)
        c = self.DROP(c)
        c = self.BAT(c)
        
        c = self.LSTM1(c)
        c = self.DROP(c)
        c = self.BAT(c)
        
        # # d = self.LSTM2(c)
        # # d = self.BAT(d)
        # c = self.LSTM1(c)
        # c = self.DROP(c)
        # c = self.BAT(c)
                        
        d = self.DENS(c)
        d = self.DROP(d)
        
        d = self.FLATT(d)
        out = self.OUT(d)
        return out
    # AFAIK: The most convenient method to print model.summary() 
    # similar to the sequential or functional API like.
    def build_graph(self):
        #x = Input(shape=(dim))
        x = Input()
        return Model(inputs=[x], outputs=self.call(x))

mymodel2 = my_model((1,28), name='MyModelMLP')
# mymodel2(X_train)
mymodel2.summary()


# %%
plot_model(mymodel2, to_file='mymodel2.png', show_shapes=True, 
                        show_layer_names=True, rankdir='TB', dpi=100, expand_nested=True)


# %%
Epoch = 100  # 模型迭代的次数
Batch_Size = 64  # 批量训练的样本的个数
Out_Class = 2  # 输出的类别的个数,0-9共10类
# 模型编译
#mYMODEL = my_model((1,28),name="mYMODEL")
#mYMODEL(X_train)
mymodel2.compile(loss='binary_crossentropy',#categorical_crossentropy', binary_crossentropy
                optimizer='adam', metrics=['categorical_accuracy'])  

# train_label_cate = tf.keras.utils.to_categorical(train_labels, 10)
# testlabels = tf.keras.utils.to_categorical(test_labels, 10)
checkpoint_path = "./models/cp-{categorical_accuracy:.5f}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建一个回调，保证验证数据集准确率最大
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                monitor='categorical_accuracy',
                                                mode='max',
                                                verbose=2,
                                                save_best_only=True)
#                                              filepath=filepath,
#                                             save_weights_only=False,
#                                             monitor='categorical_accuracy',
#                                             mode='max',
#                                             save_best_only=True)

# 动态更改学习率：在模型的回调中使用
def scheduler(epoch):  # 根据epoch动态更改学习率的参数
    if epoch < 10:
        return 0.13
    else:
        return 0.13 * tf.math.exp(0.1 * (10 - epoch))
    
lr_back = tf.keras.callbacks.LearningRateScheduler(scheduler)


# %%
mymodel2.fit(X_train,Y_train, batch_size=Batch_Size, 
            epochs=Epoch, verbose=2, validation_split=0.1, 
            callbacks=cp_callback)
            #callbacks=[cp_callback, lr_back])


# %%
# 评估模型,按照模型最后的参数计算
test_loss, test_acc = mymodel2.evaluate(X_test, Y_test)

print('测试数据集成本：{:.8f},准确率{:.8f}%%'. format(test_loss, 100*test_acc))


# %%
best_para = tf.train.latest_checkpoint(checkpoint_dir)
print('最优的参数文件：', best_para)
#best_para = './models\cp-0.92987.ckpt'
mymodel2.load_weights(best_para)

predict_loss, predict_acc = mymodel2.evaluate(X_test, Y_test)
print('使用训练后的参数','成本:', predict_loss, '准确率', predict_acc)

# %%
# del mymodel2
# # 参数加载
# # 新的模型结构保持一致。
# model_new = my_model((1,28))
# #mymodel2 = my_model((1,28), name='MyModelMLP')
# # 需要经过编译，参数也要和原来的一致
# model_new.compile(loss='binary_crossentropy',#categorical_crossentropy', binary_crossentropy
#                 optimizer='adam', metrics=['categorical_accuracy'])  
# # checkpoint_path = "./models/cp-{categorical_accuracy:.5f}.ckpt"
# # checkpoint_dir = os.path.dirname(checkpoint_path)
# # 加载已经训练好的参数
# best_para = tf.train.latest_checkpoint(checkpoint_dir)
# print('最优的参数文件：', best_para)
# model_new.load_weights(best_para)
# predict_loss, predict_acc = model_new.evaluate(X_test, Y_test)
# print('使用训练后的参数','成本:', predict_loss, '准确率', predict_acc)


# %%
# class my_model(Model):
#     def __init__(self, dim):
#         super(my_model, self).__init__()
#         the_units = 128
#         self.CNN   = Conv1D(16,5,padding="same",activation="relu",input_shape=(dim))
#         self.POLL  = MaxPool1D(2)
#         self.FLATT = Flatten()
#         self.LSTM1 = LSTM(the_units,return_sequences=True)
#         self.LSTM2 = LSTM(the_units)
#         self.BAT   = BatchNormalization()
#         self.DROP  = Dropout(rate=0.1)
#         self.DENS  = Dense(the_units, activation='relu')
#         self.OUT   = Dense(2, activation='sigmoid')
#         self.resha = Reshape((None,1, 128))
    
#     def call(self, inputs):
#         x = self.CNN(inputs)
#         x = self.POLL(x)
#         x = self.FLATT(x)
#         x = self.DENS(x)
        
#         a = self.resha(x)
        
#         b = self.LSTM1(a)
#         b = self.DROP(a)
#         b = self.BAT(b)
        
#         c = self.LSTM2(b)
#         c = self.DROP(c)
#         c = self.BAT(c)
        
#         d = self.DENS(c)
#         d = self.DROP(d)
#         return self.OUT(d)
    
#     # AFAIK: The most convenient method to print model.summary() 
#     # similar to the sequential or functional API like.
#     # def build_graph(self):
#     #     x = Input(shape=(dim))
#     #     return Model(inputs=[x], outputs=self.call(x))

# # dim = (124,124,3)
# # model = my_model((dim))
# # model.build((None, *dim))
# # model.build_graph().summary()
# dim = (None,28,1)
# model = my_model(dim)
# model.build(input_shape=dim)
# model.summary()

# model = tf.keras.utils.plot_model(model=model,
#         show_shapes=True, to_file='model.png')

# %%

# #定义模型
# import random
# from string import printable
# from tokenize import Double
# sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)

# # lstm_layers = [1,2,3,4,5]
# # dense_layers = [1,2,3,4,5,6]
# # units = [16,32,64,128]
# # dropout = [0.05,0.1,0.15,0.25]
# # Batch_size = [32,64,128]
# # optimizer = ['adam',sgd]
# lstm_layers = [2]
# dense_layers = [4]
# units = [256]
# dropout = [0.05]
# Batch_size = [64]
# optimizer = ['adam']
# for the_batch_size in Batch_size:
#     for the_dropout in dropout:
#         for the_optimizer in optimizer:
#             for the_dense_layers in dense_layers:
#                 for the_lstm_layers in lstm_layers:
#                     for the_units in units:
#                         sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#                         X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=40)
#                         X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
#                         X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
#                         model = Sequential()
#                         # model.build(input_shape=(277,277,2))
#                         #print(model.summary())
#                         #model.add(SpatialDropout1D(0.2))
#                         print(X_train.shape[0],X_train.shape[1],X_train.shape[2])
#                         print('*************')
#                         model.add(Conv1D(16,5,padding="same",activation="relu",input_shape=(X_train.shape[2],1)))
#                         model.add(MaxPool1D(2))
#                         model.add(Flatten())
#                         model.add(Dense(the_dense_layers,activation='relu'))
#                         #model.add(LSTM(the_units ,input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences = True))
#                         model.add(LSTM(the_units,return_sequences=True))
#                         model.add(Dropout(the_dropout))
#                         model.add(BatchNormalization())
#                         # #第二层
#                         for i in range(the_lstm_layers):
#                             model.add(LSTM(the_units,return_sequences=True))
#                             model.add(Dropout(the_dropout))
#                             model.add(BatchNormalization())

#                         model.add(LSTM(the_units))
#                         model.add(Dropout(the_dropout))
#                         model.add(BatchNormalization())
#                         #全连接层
#                         for i in range(the_dense_layers):
#                             model.add(Dense(the_dense_layers,activation='relu'))
#                             model.add(Dropout(the_dropout))
                            
#                         # model.add(Flatten()) 
                        
                        
#                         model.add(Dense(2, activation='softmax'))
                        
#                         #sgd = SGD(learning_rate=0.01, momentum=0.9 , decay=0.1, nesterov=False)
                        
#                         # learning_rate = 0.1
#                         # decay = 0.001
#                         # epochs = 50
#                         # batch_size = 64
                        
                        
#                         model.compile(  loss='binary_crossentropy',#categorical_crossentropy', binary_crossentropy
#                                         optimizer=the_optimizer, metrics=['categorical_accuracy'])
#                         print(model.summary())

#                         epochs = 100
#                         batch_size = the_batch_size
#                         if(the_optimizer == sgd):
#                             the_optimizer = 'sgd'
#                         filepath = './models/{categorical_accuracy:.4f}_{epoch:02d}_'+f'dropout_{the_dropout}_batch_size_{the_batch_size}_optimizer_{the_optimizer}_dense_layers_{the_dense_layers}_lstm_layers_{the_lstm_layers}_unit_{the_units}.h5'
#                         checkpoint = ModelCheckpoint(
#                                             filepath=filepath,
#                                             save_weights_only=False,
#                                             monitor='categorical_accuracy',
#                                             mode='max',
#                                             save_best_only=True)
#                         history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
#                                             #callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
#                                             callbacks=checkpoint)

# %%
#model.save('mymodel.h5')

# %%
# from tensorflow.keras.models import load_model
# best_model = load_model('./models/0.9557_78_dropout_0.05pbatch_size_32optimizer_adamdense_layers_1_lstm_layers_2_unit_128.h5')

# %%
# dataset = pd.read_csv('test.csv') 
# cols = ['Class','Area', 'Perimeter', 'Major_Axis', 'Minor_Axis', 'Eccentricity',
#         'Eqdiasq', 'Solidity', 'Convex_Area', 'Extent', 'Aspect_Ratio',
#         'Roundness', 'Compactness', 'Shapefactor_1', 'Shapefactor_2',
#         'Shapefactor_3', 'Shapefactor_4', 'Mean_RR', 'Mean_RG', 'Mean_RB',
#         'StdDev_RR', 'StdDev_RG', 'StdDev_RB', 'Skew_RR', 'Skew_RG', 'Skew_RB',
#         'Kurtosis_RR', 'Kurtosis_RG', 'Kurtosis_RB']
# # dataset = dataset.loc[:,cols]
# dataset.sample(10)
# dataset = shuffle(dataset)
# Lisclass = dataset['Class'].values
# LisClassNp = []
# for i in range(len(Lisclass)):
#     if (Lisclass[i] == 'Kirmizi_Pistachio'):
#         LisClassNp.append([0,1])
#     else:
#         LisClassNp.append([1,0])
# Y = np.array(LisClassNp)
# dataset = dataset.drop(columns='Class')
# X = dataset.values
# # from sklearn.preprocessing import MinMaxScaler
# # scaler = MinMaxScaler(feature_range=(0, 1))
# # X = scaler.fit_transform(X)
# #进行预测 make a prediction
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.90, random_state=random.randint(10,100))
# X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
# X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
# # print ('train_x.shape, train_y.shape, test_x.shape, test_y.shape')
# # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# yhat = best_model.predict(X_test)
# # print(yhat.shape)
# print(yhat)
# print(Y_test)
# sorce = 0
# for i in range(len(Y_test)):
#     sorce += roc_auc_score(Y_test[i], yhat[i])

# print('ACC:',sorce/len(Y_test))
# #拆分训练集和测试集
# # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=random.randint(10,100))
# # X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
# # X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])

# %%


# %%
# print(Y_test.shape)
# print(yhat.shape)
# # Y_test = Y_test.reshape(430)
# # yhat = yhat.reshape(430)
# # Y_test = list(Y_test)
# # yhat = list(yhat)

# %%
# print(yhat)

# %%








