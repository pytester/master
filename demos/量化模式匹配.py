#encoding:utf8
import tensorflow as tf
import numpy as np  
import matplotlib.pyplot as plt
import json
import sys

class DATA:
    def __init__(self):
        self.c = None
        self.data = None
        self.ma20 = None
    def save(self):
        import tushare as ts
        data = ts.get_hist_data('000001')
        data_save = {}
        for key in data:
            data_save[key] = list(data[key])
            data_save[key].reverse()
        with open('000001.json', 'w') as fp:
            json.dump(data_save, fp)
        #data.to_json('000001.json')

    def load(self):
        if self.data == None:
            with open('000001.json') as fp:
                self.data = json.load(fp)
        self.__parse_data()

    def __parse_data(self):
        data = self.data
        if self.c == None:
            #self.c = [d['close'] for d in data]
            self.c = data['close']

        if self.ma20 == None:
            #self.ma20 = [d['ma20'] for d in data]
            self.ma20 = data['ma20']

    def show(self):
        data = self.load()        
        plt.plot(self.c)
        plt.plot(self.ma20)
        plt.show()
class PatternNetwork:

    in_muti = 3
    def __init__(self, c_len):# c_len = 243 => 3 ^ 5
        self.seg_lens = []
        self.c_len = c_len
        seg_len = float(c_len)
        # 3^4, 3^3, 3^2
        while seg_len > self.in_muti * 3:
            self.seg_lens.append(int(seg_len))              # close length
            seg_len = seg_len / self.in_muti
        self.data = []
        self.mark = []
        self.np_data = None
        self.np_ret  = None
    # ret: 1, 0, -1
    def append_data(self, c, v, ret, mark):
        #if len(c) != self.c_len or len(v) != self.v_len:
        self.data.append((c[-self.c_len:], v[-self.c_len:], ret))
        self.mark.append(mark)
    def get_mark(self, idx):
        return self.mark[idx]
    def mk_numpy(self):
        col_len = sum(self.seg_lens)  # np_data colume length
        row_len = len(self.data)
        np_data = np.zeros((row_len, col_len * 2))
        np_ret  = np.zeros((row_len, 2))
        for i in xrange(0, row_len):
            idx = 0
            for seg_len in self.seg_lens:
                np_data[i, idx:(idx + seg_len)] = self.data[i][0][-seg_len:]
                idx1 = idx + col_len
                np_data[i, idx1:(idx1 + seg_len)] = self.data[i][1][-seg_len:]
                idx += seg_len
            ret = self.data[i][2]
            if ret > 0:
                np_ret[i, 0] = 1
            elif ret < 0:
                np_ret[i, 1] = 1
        # np_data 行为样本的数目,列为一个样本的长度
        self.np_data = np_data
        self.np_ret  = np_ret

    def mk_network(self):

        np_data_holder = tf.placeholder(tf.float32, [None, self.np_data.shape[1]], name = 'np_data')
        # 一维数组, 长度是len(np_data) * 周期数 * 2
        pattern_layer  = self.pattern_layer(np_data_holder, self.np_data)
        # 3倍数量中间层
        middle_layer_len = int(pattern_layer.shape[1]) * 3
        middle_layer   = self.normal_layer(pattern_layer, 
                                            middle_layer_len, 
                                            tf.nn.sigmoid)
        out_layer      = self.normal_layer(middle_layer, self.np_ret.shape[1], tf.nn.sigmoid)
        return np_data_holder, out_layer
    #定义隐藏层
    @staticmethod
    def normal_layer(inputs, out_len, activation_function=None):
        in_len = int(inputs.shape[1])
        #Weights=tf.Variable(tf.zeros([in_len, out_len]))  #权值  
        Weights=tf.Variable(tf.fill([in_len, out_len], 1 / in_len))  #权值  
        #Weights=tf.Variable(tf.random_normal([in_len,out_len]))  #权值  
        #Weights=tf.Variable(tf.ones([in_len,out_len])*0.1)  #权值  
        biases=tf.Variable(tf.zeros([1, out_len]))# + 0.1) #偏置  
        Wx_plus_b=tf.matmul(inputs, Weights) + biases  #z=wx+b  
        if activation_function is None:
            outputs=Wx_plus_b  
        else:  
            outputs=activation_function(Wx_plus_b)  
        return outputs
    
    # 返回行数为:1, 列数为:len(self.np_data) * len(self.in_sigm_len) * 2的数组
    # 因为每行np_data, 由不同周期的数据组成, 并且包含close与vol
    # 1 - sum((x-w)^2, ...) / seg_len, 1是可变变量
    @staticmethod
    def pattern_layer(inputs, np_data):
        distance = tf.square(inputs - tf.constant(np_data))
        # 每个分块分开计算
        sum_matrix = np.zeros([np_data.shape[1], len(self.seg_lens) * 2])
        seg_idx = 0
        col_idx = 0
        for _ in xrange(2):
            for seg_len in self.seg_lens:
                sum_matrix[seg_idx:(seg_idx + seg_len), col_idx] = 1 / seg_len
                seg_idx += seg_len
                col_idx += 1
        self.pattern_patrix = tf.matmul(distance, sum_matrix)
        # out = [x,x,x,x,] 变成一维
        pattern_line = tf.reshape(tf.transpose(self.pattern_matrix), [-1])
        pattern_out  = tf.Variable(tf.ones([1, pattern_line.shape[0]])) - pattern_line
        return tf.nn.relu(pattern_out)

    def init(self):
        np_ret_holder = tf.placeholder(tf.float32, [None, self.np_ret.shape[1]], name = 'np_ret')
        np_data_holder, out_layer = self.mk_network()
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(out_layer - np_ret_holder),  
                                    reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(self.loss)#梯度下降优化器,减少误差，学习效率0.1  

        init = tf.global_variables_initializer()  
        self.sess = tf.Session()  
        self.sess.run(init)

    def run(self):
        self.sess.run(self.train_step, feed_dict = {'np_data': self.np_data, 'np_ret': self.np_ret})

    def loss(self):
        loss = sess.run(self.loss,feed_dict = {'np_data': self.np_data, 'np_ret': self.np_ret}) #输出误差  
        return loss

def MA(data, N):
    
class PatternSaver:
    pass
if __name__ == '__main__':
    #make up some real data  
    data = DATA()
    #data.save()
    data.load()
    #data.show()
    #sys.exit(0)
#if 0:
    #x_ = []
    #y_ = []
    start_idx = 20
    x_data = np.zeros((len(data.c) - start_idx, start_idx), dtype=np.float32)
    y_data = np.zeros((len(data.c) - start_idx, 1),dtype=np.float32)
    for idx in range(start_idx, len(data.c)):
        x_data[idx - start_idx] = data.c[idx - start_idx:idx]
        y_data[idx - start_idx] = data.ma20[idx]
        #x_.append(data.c[idx - 40:idx])
        #y_.append(data.ma20[idx])
