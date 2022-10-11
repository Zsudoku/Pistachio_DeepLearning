'''
Date: 2022-10-03 21:06:14
LastEditors: ZSudoku
LastEditTime: 2022-10-10 22:11:55
FilePath: \Pistachio_DeepLearning\diyLSTM.py
'''
# %%
import matplotlib.pyplot as plt
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
from tensorflow.keras.layers import Dropout,LSTM,BatchNormalization,Conv1D,MaxPool1D,Reshape,Input,Conv2D
from tensorflow.keras.optimizers import SGD
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
import re
import torch
from keras.utils.vis_utils import plot_model
import math 
import os
import scipy.io as io
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import keras
import gc
print('=============================',tf.__version__) # 注意version前后各有两个下划线   2.4.0
print('=============================',keras.__version__)
#tf.executing_eagerly()

# %%
dataset = pd.read_csv('./Pistachio_Dataset/Pistachio_28_Features_Dataset/Pistachio_28_Features_Dataset.csv') 
cols = ['Class','Area', 'Perimeter', 'Major_Axis', 'Minor_Axis', 'Eccentricity',
        'Eqdiasq', 'Solidity', 'Convex_Area', 'Extent', 'Aspect_Ratio',
        'Roundness', 'Compactness', 'Shapefactor_1', 'Shapefactor_2',
        'Shapefactor_3', 'Shapefactor_4', 'Mean_RR', 'Mean_RG', 'Mean_RB',
        'StdDev_RR', 'StdDev_RG', 'StdDev_RB', 'Skew_RR', 'Skew_RG', 'Skew_RB',
        'Kurtosis_RR', 'Kurtosis_RG', 'Kurtosis_RB']
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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.001, random_state=40)
X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
#X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])

X_train = tf.cast(X_train, dtype='float32')
#X_test = tf.cast(X_test, dtype='float32')
Y_train = tf.cast(Y_train, dtype='float32')
# Y_test = tf.cast(Y_test, dtype='float32')

# %%
batch_size = 64 
sequence_lenth = 1 #每个句子的长度
input_size = 28    #每个时序的单独向量，代表每个字的含义
output_size = 128
x = X_train
#x = tf.random.uniform((batch_size,sequence_lenth,input_size))

# %%
#LSTM'S INOPUT: [batch_size,sequence_length,output_size]
#LSTM'S OUTPUT1: [batch_size,sequence_length,output_size]
#       OUTPUT2:[batch_size,output_size]

# %%
class CustomLSTM(tf.keras.layers.Layer):
    '''
    #LSTM'S INOPUT: [batch_size,sequence_length,output_size]
    #LSTM'S OUTPUT1: [batch_size,sequence_length,output_size]
    #       OUTPUT2:[batch_size,output_size]
    '''
    
    def __init__(self,output_size,return_sequences=False,cause_error = False):
        super(CustomLSTM,self).__init__()
        self.output_size = output_size
        self.return_sequences = return_sequences
        self.cause_error = cause_error
    def build(self,input_shape):
        super(CustomLSTM,self).build(input_shape)
        input_size = int(input_shape[-1])
        #self.wf = self.add_weight('wf', shape=(input_size,self.output_size),initializer='random_normal',trainable=True)
        self.wi = self.add_weight('wi', shape=(input_size,self.output_size),initializer='random_normal',trainable=True)
        self.wo = self.add_weight('wo', shape=(input_size,self.output_size),initializer='random_normal',trainable=True)
        self.wc = self.add_weight('wc', shape=(input_size,self.output_size),initializer='random_normal',trainable=True)

        #self.uf = self.add_weight('uf', shape=(self.output_size,self.output_size),initializer='random_normal',trainable=True)
        #self.ui = self.add_weight('ui', shape=(self.output_size,self.output_size),initializer='random_normal',trainable=True)
        #self.uo = self.add_weight('uo', shape=(self.output_size,self.output_size),initializer='random_normal',trainable=True)
        #self.uc = self.add_weight('uc', shape=(self.output_size,self.output_size),initializer='random_normal',trainable=True)

        #self.bf = self.add_weight('bf', shape=(1,self.output_size),initializer='random_normal',trainable=True)
        self.bi = self.add_weight('bi', shape=(1,self.output_size),initializer='random_normal',trainable=True)
        self.bo = self.add_weight('bo', shape=(1,self.output_size),initializer='random_normal',trainable=True)
        self.bc = self.add_weight('bc', shape=(1,self.output_size),initializer='random_normal',trainable=True)
    def call(self, x):
        Para = []
        sequence_outputs = []
        for i in range(sequence_lenth):
            if i == 0:
                xt = x[:, 0 ,:]
                # para1 = tf.matmul(xt,self.wf) + self.bf
                # para2 = tf.matmul(xt,self.wi) + self.bi
                # para3 = tf.matmul(xt,self.wo) + self.bo
                # para4 = tf.matmul(xt,self.wc) + self.bc
                # ft = tf.sigmoid(para1)
                # it = tf.sigmoid(para2)
                # ot = tf.sigmoid(para3)
                # cht = tf.tanh(para4)
                # ct = it * cht         
                # ht = ot * tf.tanh(ct)
                #ft = tf.sigmoid(tf.matmul(xt,self.wf) + self.bf)
                # it = tf.sigmoid(tf.matmul(xt,self.wi) + self.bi)
                # ot = tf.sigmoid(tf.matmul(xt,self.wo) + self.bo)
                it = tf.nn.sigmoid(tf.matmul(xt,self.wi) + self.bi)
                ot = tf.nn.relu(tf.matmul(xt,self.wo) + self.bo)
                cht = tf.tanh(tf.matmul(xt,self.wc) + self.bc)
                ct = it * cht
                ht = ot * tf.tanh(ct)
                
                #print('叫爸爸')                                                  晶哥铁马
                # cht = tf.tanh(tf.matmul(xt,self.wc) + self.bc)
                # ct = it * cht
                # ht = ot * tf.tanh(ct)
                # print('************************************')
                # #tf.executing_eagerly()
                # #tf.compat.v1.enable_eager_execution()
                # with tf.compat.v1.Session() as sess:
                #     print (para1.eval())
                # # b = para1.numpy()
                # # print(b)
                # print('************************************')
                # para1 = para1.eval()
                # para2 = para2.eval()
                # para3 = para3.eval()
                # para4 = para4.eval()
                # ct = ct.eval()
                # print(para1)
                # np.savetxt('para1.txt',para1)
                # np.savetxt('para2.txt',para2)
                # np.savetxt('para3.txt',para3)
                # np.savetxt('para4.txt',para4)
                # np.savetxt('ct.txt',ct)
                #io.savemat('save.mat',{'result1':result1})
            # else:
            #     xt = x[:, i ,:]
            #     para1 = tf.matmul(xt,self.wf) + tf.matmul(ht,self.uf) + self.bf
            #     para2 = tf.matmul(xt,self.wi) + tf.matmul(ht,self.ui) + self.bi
            #     para3 = tf.matmul(xt,self.wo) + tf.matmul(ht,self.uo) + self.bo
            #     para4 = tf.matmul(xt,self.wc) + tf.matmul(ht,self.uc) + self.bc
            #     ft = tf.sigmoid(para1)
            #     it = tf.sigmoid(para2)
            #     ot = tf.sigmoid(para3)
            #     cht = tf.tanh(para4)
            #     ct = ft * ct + it * cht
            #     ht = ot * tf.tanh(ct)
                #Para.append([para1,para2,para3,para4,ct])
                #Note.write(f'para1:,{para1}\npara2:{para2}\npara3:{para3}\npara4:{para4}\nct:{ct}')
            sequence_outputs.append(ht)
        
        #Note.write(str(Para))
        #Note.close()
        sequence_outputs = tf.stack(sequence_outputs)
        sequence_outputs = tf.transpose(sequence_outputs,(1, 0, 2))
        if self.return_sequences:
            return sequence_outputs
        return sequence_outputs[:, -1 ,:]

# %%
class my_model(Model):
    def __init__(self, input_shape, **kwargs):
        super().__init__(**kwargs) 

        
        #input_shape = tf.reshape(input_shape)
        #self.graph = tf.Graph()
        the_units = 128
        self.CNN   = Conv1D(the_units,1,padding="valid",activation="relu")
        self.POLL  = MaxPool1D(1)
        self.FLATT = Flatten()
        self.LSTM0 = CustomLSTM(output_size=the_units,return_sequences=True)
                        #kernel_initializer=tf.keras.initializers.glorot_normal(seed=2),bias_initializer=tf.keras.initializers.Zeros())
        self.LSTM1 = CustomLSTM(output_size=the_units,return_sequences=True)
        self.LSTM2 = CustomLSTM(output_size=the_units)
        self.BAT   = BatchNormalization()
        self.DROP  = Dropout(rate=0.05)
        self.DENS  = Dense(8, activation='relu')
        self.OUT   = Dense(2, activation='softmax')
        #self.resha = Reshape((1, 28))
        
        # 模型的参数
        self.input_layer = tf.keras.layers.Input(input_shape)
        #print('self.input_layer',hasattr(self.input_layer, '_keras_history'))
        self.out = self.call(self.input_layer)
        #print('self.out',hasattr(self.out, '_keras_history'))
        
        super().__init__( inputs=self.input_layer,outputs=self.out, **kwargs)
    
    # 初始化模型的参数
    def build(self):
        self._is_graph_network = True
        self._init_graph_network(inputs=self.input_layer,outputs=self.out)
    
    def call(self, inputdata, **kwargs):
                
        ##x1 = self.LSTM0(inputdata)
        ##print('inputdata',inputdata)
        #x1 = self.CNN(inputdata)
        # ##x1 = self.LSTM0(inputdata)
        # x1 = self.POLL(x1)
        #c = self.DROP(x1)
        ##c = self.BAT(x1)

        c = self.LSTM1(inputdata)
        #print(hasattr(c, '_keras_history'))
        c = self.DROP(c)
        #print(hasattr(c, '_keras_history'))
        c = self.BAT(c)
        #print(hasattr(c, '_keras_history'))
        # c = self.LSTM1(c)
        # c = self.DROP(c)
        # c = self.BAT(c)
        
        d = self.LSTM2(c)
        #print(hasattr(c, '_keras_history'))
        d = self.DROP(d)
        #d = self.BAT(d)
        #print(hasattr(d, '_keras_history'))
        #d = self.BAT(d)
        # c = self.LSTM1(c)
        # c = self.DROP(c)
        # c = self.BAT(c)
                        
        d = self.DENS(d)
        #print(hasattr(d, '_keras_history'))
        d = self.DROP(d)
        #print(hasattr(d, '_keras_history'))
        
        #d = self.FLATT(d)
        #print(hasattr(d, '_keras_history'))
        out = self.OUT(d)
        #print(hasattr(d, '_keras_history'))
        return out
    # AFAIK: The most convenient method to print model.summary() 
    # similar to the sequential or functional API like.
    def build_graph(self):
        #x = Input(shape=(dim))
        x = Input()
        return Model(inputs=[x], outputs=self.call(x))


def get_k_fold_data(k, i, X, y):
    '''它返回第i折交叉验证时所需要的训练和验证数据。'''
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = tf.concat([X_train, X_part], 0)
            y_train = tf.concat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs,batch_size):
    '''在 ? 折交叉验证中我们训练 ? 次并返回训练和验证的平均误差'''
    train_l_sum, valid_l_sum = 0, 0
    
    for i in range(k):
        X_train2, y_train2, X_valid, y_valid = get_k_fold_data(k, i, X_train, y_train)
        # 这里的*表示传入的参数是一个元组（tuple），对应的**传入的参数是一个字典
        net =  my_model((1,28), name='MyModelMLP')
        #net2 =  my_model((1,28), name='MyModelMLP')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                monitor='categorical_accuracy',
                                                mode='max',
                                                verbose=2,
                                                save_best_only=True)
        net.compile(loss='binary_crossentropy',#categorical_crossentropy', binary_crossentropy
                optimizer='adam', metrics=['categorical_accuracy'],run_eagerly=True)  
        history = net.fit(X_train2, y_train2,batch_size=batch_size,
                                        epochs=num_epochs, verbose=2,validation_data=(X_valid, y_valid), validation_split=0.1, 
                                        validation_freq=10,callbacks=cp_callback)
        train_ls = history.history['categorical_accuracy']
        valid_ls = history.history['val_categorical_accuracy']
        train_l_sum += max(train_ls)
        valid_l_sum += max(valid_ls)
        #   history包含以下几个属性：
        # 训练集loss： loss
        # 测试集loss： val_loss
        # 训练集准确率： sparse_categorical_accuracy
        # 测试集准确率： val_sparse_categorical_accuracy
        # acc = history.history['sparse_categorical_accuracy']
        # val_acc = history.history['val_sparse_categorical_accuracy']
        # if i == 0:
        #     d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
        #                 range(1, num_epochs + 1), valid_ls,
        #                 ['train', 'valid'])
        print('fold %d, train acc %f, valid acc %f'
            % (i, train_ls[-1], valid_ls[-1]))
        del net
        gc.collect()
        #ctypes.pythonapi._Py_Dealloc(ctypes.py_object(net))
        
    return train_l_sum / k, valid_l_sum / k
def ave_list(lis):
    sum = 0
    for i in range(len(lis)):
        sum += lis[i]
    return sum / len(lis)
checkpoint_path = "./models/cp-{categorical_accuracy:.5f}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
train_acc = []
valid_acc = []
for i in range(10):
    t,v = k_fold(10, X_train, Y_train, 100,128)
    train_acc.append(t)
    valid_acc.append(v)
print('训练集：',train_acc)
print('验证集:',valid_acc)

print('训练集平均，验证集平均：',ave_list(train_acc),ave_list(valid_acc))
#print('************\n训练集和验证集精度：',k_fold(10, X_train, Y_train, 100,128))
# mymodel2 = my_model((1,28), name='MyModelMLP')
# # mymodel2(X_train)
# mymodel2.summary()


# # %% [markdown]
# # 

# # %%
# Epoch = 100  # 模型迭代的次数
# Batch_Size = 128  # 批量训练的样本的个数
# Out_Class = 2  # 输出的类别的个数,0-9共10类
# # 模型编译
# #mYMODEL = my_model((1,28),name="mYMODEL")
# #mYMODEL(X_train)
# mymodel2.compile(loss='binary_crossentropy',#categorical_crossentropy', binary_crossentropy
#                 optimizer='adam', metrics=['categorical_accuracy'],run_eagerly=True)  

# # train_label_cate = tf.keras.utils.to_categorical(train_labels, 10)
# # testlabels = tf.keras.utils.to_categorical(test_labels, 10)
# checkpoint_path = "./models/cp-{categorical_accuracy:.5f}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# # 创建一个回调，保证验证数据集准确率最大
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                 save_weights_only=True,
#                                                 monitor='categorical_accuracy',
#                                                 mode='max',
#                                                 verbose=2,
#                                                 save_best_only=True)
# #                                              filepath=filepath,
# #                                             save_weights_only=False,
# #                                             monitor='categorical_accuracy',
# #                                             mode='max',
# #                                             save_best_only=True)

# # 动态更改学习率：在模型的回调中使用
# def scheduler(epoch):  # 根据epoch动态更改学习率的参数
#     if epoch < 10:
#         return 0.13
#     else:
#         return 0.13 * tf.math.exp(0.1 * (10 - epoch))
    
# lr_back = tf.keras.callbacks.LearningRateScheduler(scheduler)


# # %%
# # Note = open('LSTM-para.txt',mode='w')
# # Note.close()
# mymodel2.fit(X_train,Y_train, batch_size=Batch_Size, 
#             epochs=Epoch, verbose=2, #validation_split=0.1, 
#             callbacks=cp_callback)
#             #callbacks=[cp_callback, lr_back])


# # %%
# # 评估模型,按照模型最后的参数计算
# test_loss, test_acc = mymodel2.evaluate(X_test, Y_test)

# print('测试数据集成本：{:.8f},准确率{:.8f}%%'. format(test_loss, 100*test_acc))


# best_para = tf.train.latest_checkpoint(checkpoint_dir)
# print('最优的参数文件：', best_para)
# #best_para = './models\cp-0.92987.ckpt'

# mymodel2.load_weights(best_para)

# predict_loss, predict_acc = mymodel2.evaluate(X_test, Y_test)
# print('使用训练后的参数','成本:', predict_loss, '准确率', predict_acc)




# %% [markdown]
# 使用其他数据集去做验证
# 可行性分析，会拿几种进行对比
# 数据分析与处理
# 在输入到模型之前，有一个优化器，该优化器是对指标进行排序的，是一个固定的模式
# 顺序优化的优化器。基于优化器，针对大数据，使用LSTM对所有指标进行分类（重要性分类），对几类的数据做相应优化，
# 越不重要越先输入还是越重要越先输入。重要性体现在什么地方。
# 
# 优化器-> LSTM（对内部超参数进行优化）
# 找到其他LSTM的改进，进行实验。最近三年来新的LSTM改进。
# 
# 改进——>.0本身的改进
#     -->基本原理的改进，逻辑门的改进

# %%
# from pickletools import optimize


# model = tf.keras.Sequential([
#     CustomLSTM(output_size=128),
#     tf.keras.layers.Dense(2, activation='softmax')
# ])
# model.compile(loss='binary_crossentropy',#categorical_crossentropy', binary_crossentropy
#                 optimizer='adam', metrics=['categorical_accuracy'])  

# %%
# model.fit(X_train,Y_train,batch_size=64)


