'''
Date: 2022-10-11 20:04:55
LastEditors: ZSudoku
LastEditTime: 2022-11-24 14:06:52
FilePath: \Pistachio_DeepLearning\DiyLSTMmod.py
'''
# coding=utf-8
# from distutils.command.config import config
from keras.layers.core import Flatten
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout,BatchNormalization,Conv1D,MaxPool1D,Input,GRU,Bidirectional,LSTM
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import keras
import gc
tf.random.set_seed(123)
tf.autograph.set_verbosity(0)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore') 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 要放在 import tensorflow as tf 前面才会起作用 ！！！
import tensorflow as tf
X_train = []
batch_size = 128
#sequence_lenth = 7 #每个句子的长度 葡萄干
sequence_lenth = 16
input_size = 1    #每个时序的单独向量，代表每个字的含义
output_size = 128
x = X_train
checkpoint_path = "./models/{categorical_accuracy:.5f}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

class CustomLSTM(tf.keras.layers.Layer):
    '''
    #LSTM'S INOPUT: [batch_size,sequence_length,output_size]
    #LSTM'S OUTPUT1: [batch_size,sequence_length,output_size]
    #       OUTPUT2:[batch_size,output_size]
    '''
    
    def __init__(self,output_size,return_sequences=False,cause_error = False,**kwargs):
        super(CustomLSTM,self).__init__(**kwargs)
        self.output_size = output_size
        self.return_sequences = return_sequences
        self.cause_error = cause_error
    def build(self,input_shape):
        super(CustomLSTM,self).build(input_shape)
        input_size = int(input_shape[-1])
        self.wf = self.add_weight('wf', shape=(input_size,self.output_size),initializer='random_normal',trainable=True)#''random_normal''
        self.wi = self.add_weight('wi', shape=(input_size,self.output_size),initializer='random_normal',trainable=True)
        self.wo = self.add_weight('wo', shape=(input_size,self.output_size),initializer='random_normal',trainable=True)
        self.wc = self.add_weight('wc', shape=(input_size,self.output_size),initializer='random_normal',trainable=True)

        self.uf = self.add_weight('uf', shape=(self.output_size,self.output_size),initializer='random_normal',trainable=True)
        self.ui = self.add_weight('ui', shape=(self.output_size,self.output_size),initializer='random_normal',trainable=True)
        self.uo = self.add_weight('uo', shape=(self.output_size,self.output_size),initializer='random_normal',trainable=True)
        self.uc = self.add_weight('uc', shape=(self.output_size,self.output_size),initializer='random_normal',trainable=True)

        self.bf = self.add_weight('bf', shape=(1,self.output_size),initializer='random_normal',trainable=True)
        self.bi = self.add_weight('bi', shape=(1,self.output_size),initializer='random_normal',trainable=True)
        self.bo = self.add_weight('bo', shape=(1,self.output_size),initializer='random_normal',trainable=True)
        self.bc = self.add_weight('bc', shape=(1,self.output_size),initializer='random_normal',trainable=True)
        #keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=123)
        #keras.initializers.Orthogonal(gain=1.0, seed=123)
        #keras.initializers.Constant(value=0)

    def call(self, x):
        Para = []
        sequence_outputs = []
        for i in range(sequence_lenth):
            if i == 0:
                xt = x[:, 0 ,:]
                ft = tf.sigmoid(tf.matmul(xt,self.wf) + self.bf)
                it = tf.sigmoid(tf.matmul(xt,self.wi) + self.bi)
                ot = tf.sigmoid(tf.matmul(xt,self.wo) + self.bo)
                it = tf.nn.sigmoid(tf.matmul(xt,self.wi) + self.bi)
                ot = tf.nn.sigmoid(tf.matmul(xt,self.wo) + self.bo)
                cht = tf.tanh(tf.matmul(xt,self.wc) + self.bc)
                ct = it * cht
                ht = ot * tf.tanh(ct)                  
                cht = tf.tanh(tf.matmul(xt,self.wc) + self.bc)
                ct = it * cht
                ht = ot * tf.tanh(ct)
            else:
                xt = x[:, i ,:]
                para1 = tf.matmul(xt,self.wf) + tf.matmul(ht,self.uf) + self.bf
                para2 = tf.matmul(xt,self.wi) + tf.matmul(ht,self.ui) + self.bi
                para3 = tf.matmul(xt,self.wo) + tf.matmul(ht,self.uo) + self.bo
                para4 = tf.matmul(xt,self.wc) + tf.matmul(ht,self.uc) + self.bc
                ft = tf.sigmoid(para1)
                it = tf.sigmoid(para2)
                ot = tf.sigmoid(para3)
                cht = tf.tanh(para4)
                ct = ft * ct + it * cht
                ht = ot * tf.tanh(ct)
            sequence_outputs.append(ht)
        sequence_outputs = tf.stack(sequence_outputs)
        sequence_outputs = tf.transpose(sequence_outputs,(1, 0, 2))
        if self.return_sequences:
            return sequence_outputs
        return sequence_outputs[:, -1 ,:]
    
    # def get_config(self):
    #     config =  super().get_config().copy()
    #     config.update({
    #         'output_size' : self.output_size,
    #         'return_sequences' : self.return_sequences,
    #         'cause_error':self.cause_error
    #         })
    #     return config
class my_model(Model):
    def __init__(self, input_shape, **kwargs):
        super().__init__(**kwargs) 

        
        #input_shape = tf.reshape(input_shape)
        #self.graph = tf.Graph()
        the_units = 128
        self.CNN   = Conv1D(the_units,1,padding="valid",activation="relu")
        self.POLL  = MaxPool1D(1)
        self.FLATT = Flatten()
        self.GRU   = GRU(units=the_units,return_sequences=True)
        self.GRU2  = GRU(units=the_units)
        self.BILSTM = Bidirectional(LSTM(the_units,return_sequences=True))
        self.BILSTM2 = Bidirectional(LSTM(the_units))
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

        #c = self.LSTM1(x1)
        c = self.LSTM1(inputdata)
        #c = self.GRU(inputdata)
        #c = self.BILSTM(inputdata)
        c = self.DROP(c)
        c = self.BAT(c)
        
        d = self.LSTM2(c)
        #d = self.GRU2(c)
        #d = self.BILSTM2(c)
        d = self.DROP(d)
        
                        
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
        net =  my_model((sequence_lenth,input_size), name='MyModelMLP')
        #net2 =  my_model((1,28), name='MyModelMLP')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                monitor='categorical_accuracy',
                                                mode='max',
                                                verbose=0,
                                                save_best_only=True)
        net.compile(loss='categorical_crossentropy',#categorical_crossentropy', binary_crossentropy
                optimizer='adam', metrics=['categorical_accuracy'],run_eagerly=True)  
        history = net.fit(X_train2, y_train2,batch_size=batch_size,
                                        epochs=num_epochs, verbose=0,validation_data=(X_valid, y_valid), validation_split=0.1, 
                                        validation_freq=10,callbacks=cp_callback)
        train_ls = history.history['categorical_accuracy']
        valid_ls = history.history['val_categorical_accuracy']
        train_l_sum += max(train_ls)
        valid_l_sum += max(valid_ls)
        #   history包含以下几个属性：
        # 训练集loss： loss
        # 测试集loss： val_loss
        # 训练集准确率： categorical_accuracy
        # 测试集准确率： val_categorical_accuracy
        # acc = history.history['categorical_accuracy']
        # val_acc = history.history['val_categorical_accuracy']
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