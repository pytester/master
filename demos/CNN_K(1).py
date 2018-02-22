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

#定义隐藏层  
def add_layer(inputs, in_size, out_size, activation_function=None):  
    #Weights=tf.Variable(tf.zeros([in_size, out_size]))  #权值  
    #Weights=tf.Variable(tf.fill([in_size, out_size], 1 / in_size))  #权值  
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))  #权值  
    #Weights=tf.Variable(tf.ones([in_size,out_size])*0.1)  #权值  
    biases=tf.Variable(tf.zeros([1, out_size]) + 0.001) #偏置  
    Wx_plus_b=tf.matmul(inputs, Weights)# + biases  #z=wx+b  
    if activation_function is None:  
        outputs=Wx_plus_b  
    else:  
        outputs=activation_function(Wx_plus_b)  
    return outputs, Weights, biases

def add_square_layer(inputs, in_size, out_size):  
    #a=tf.Variable(tf.random_normal([1, in_size]))
    b=tf.Variable(tf.ones([1, in_size]) * 0.5)
    #c=tf.ones([1, in_size])

    f = 1 - tf.square(inputs - b)

    return tf.matmul(tf.nn.relu(f), tf.ones([in_size, out_size])), b

if __name__ == '__main__':
    #make up some real data  
    data = DATA()
    #data.save()
    data.load()
    #data.show()
    #sys.exit(0)

    #train_step所要输入的值  
    
    size = [(20, 20), #long price
    		#(20, 10), #long vol
    		#(20, 10), #short price
            #(20, 10), #short vol
    		]
    x_data = np.zeros([1, size[0][0]])
    tmp = np.array(data.c[200:200+size[0][0]])
    x_data[0] = tmp / tmp.max()
    #ys=tf.placeholder(tf.float32,[None, 1], name = 'ys')
    xs=tf.placeholder(tf.float32,[None, size[0][0]], name = 'xs')
    #short_xs=tf.placeholder(tf.float32,[None, short_dis], name = 'short_xs')

    ###建立第一,二次隐藏层layer  
    ###add_layer(inputs,in_size,out_size,activation_function=None)
    prediction, b = add_square_layer(xs, size[0][0], size[0][1])

    #prediction, w2, b2 = add_layer(l1, l1_cnt, 1, activation_function=None)
      
    #创建损失函数  
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(prediction - size[0][0]),  
                   reduction_indices=[1]))
    #loss=tf.reduce_sum(tf.square(prediction - ys))
    train_step=tf.train.GradientDescentOptimizer(0.0001).minimize(loss)#梯度下降优化器,减少误差，学习效率0.1  
      
    #important step  
    init=tf.global_variables_initializer()  
    sess=tf.Session()  
    sess.run(init)  

    #绘图部分  
    fig=plt.figure()  
    ax=fig.add_subplot(1,1,1)  
    #ax.scatter(data.c[start_idx:],y_data)
    
    #ax.plot(x_xais, y_data)
    #ax.plot(x_xais, data.c[start_idx:])
    ax.plot(range(0, len(x_data[0])), x_data[0])
    b1 = sess.run(b)
    print('########################################################')
    lost_value = sess.run(loss,feed_dict={xs:x_data}) #输出误差  
    print('[0]=%f'%lost_value)
    print('x_data=', x_data)
    print('b1=', b1)
    lines = None
    for i in range(20000):
        sess.run(train_step,feed_dict={xs:x_data})  
        if i%50==0:
            lost_value = sess.run(loss,feed_dict={xs:x_data}) #输出误差  
            print('[%d]=%f'%(i, lost_value))
            if np.isnan(lost_value) or np.isinf(lost_value):
                print('lost is nan')
                break
    
    prediction_value=sess.run(prediction,feed_dict={xs:x_data})
    b1 = sess.run(b)
    print('b=', b1) 
    print('prediction=', prediction_value)
    lines=ax.plot(range(0, len(x_data[0])), b1[0],'r', lw=5)
    plt.show()