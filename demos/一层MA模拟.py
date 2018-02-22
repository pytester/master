#encoding:utf8
#import tensorflow as tf
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
        fig=plt.figure()  
        ax=fig.add_subplot(2,1,1)  
        #ax.scatter(data.c[start_idx:],y_data)
        
        #ax.plot(x_xais, y_data)
        #ax.plot(x_xais, data.c[start_idx:])
        ax.plot(self.c)
        plt.plot(self.ma20)
        ax=fig.add_subplot(2,1,2)
        k,d,j = self.KDJ()  
        ax.plot(k)
        ax.plot(d)
        ax.plot(j)
        plt.show()

    def KDJ(self, N=30,M1=20,M2=20):
        data = self.data
        datelen=len(self.c)  
        array=np.array(data)  
        kdjarr=[]
        cclose = data['close']
        low = data['low']
        high = data['high']
        K = []
        D = []
        J = []
        for i in range(datelen):  
            if i-N<0:  
                b=0  
            else:  
                b=i-N+1  

            # 2: high, 3:low, 4: close
            min_low = float(min(low[b:i+1]))
            max_high = float(max(high[b:i+1]))
            rsv=(float(cclose[i])-min_low)/(max_high-min_low)*100  
            if i==0:  
                k=rsv  
                d=rsv  
            else:  
                k=1/float(M1)*rsv+(float(M1)-1)/M1*float(K[-1])  
                d=1/float(M2)*k+(float(M2)-1)/M2*float(D[-1])  
            j=3*k-2*d  
            #kdjarr.append(list((rsvarr[-1,0],rsv,k,d,j))) 
            K.append(k)
            D.append(d)
            J.append(j)
        return K, D, J 
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

if __name__ == '__main__':
    #make up some real data  
    data = DATA()
    #data.save()
    data.load()
    data.show()
    sys.exit(0)
#if 0:
    #x_ = []
    #y_ = []
    start_idx = 30
    x_data = np.zeros((len(data.c) - start_idx, start_idx), dtype=np.float32)
    y_data = np.zeros((len(data.c) - start_idx, 1),dtype=np.float32)
    _, _, j = data.KDJ()
    for idx in range(start_idx, len(data.c)):
        x_data[idx - start_idx] = data.c[idx - start_idx:idx]
        y_data[idx - start_idx] = j[idx]
        #x_.append(data.c[idx - 40:idx])
        #y_.append(data.ma20[idx])
    #x_data = np.array(x_)
    #y_data = np.array(y_).resharp((len(y_), 1))
    
    #train_step所要输入的值  
    ys=tf.placeholder(tf.float32,[None, 1], name = 'ys')
    size = [(start_idx, 20),
    		#(20, 10),
    		#(20, 10),
    		]
    xs=tf.placeholder(tf.float32,[None, size[0][0]], name = 'xs')

    ###建立第一,二次隐藏层layer  
    ###add_layer(inputs,in_size,out_size,activation_function=None)
    l1_mtx = []
    l1_cnt = 0
    for sz in size:#[-sz[0]:]
        #l1_mtx.append(
        l1, w1, b1 = add_layer(xs, sz[0], sz[1], activation_function=None)#, activation_function=tf.nn.relu, tf.nn.sigmoid))#激励函数(activation_function)ReLU  )
        l1_cnt += sz[1]
    #l1 = tf.concat(0, l1_mtx)
    prediction = l1
    #prediction, w2, b2 = add_layer(l1, l1_cnt, 1, activation_function=None)
      
    #创建损失函数  
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys),  
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
    ax.plot(range(0, len(data.c)), data.ma20)
    ax.plot(range(0, len(data.c)), data.c)
    #ax.scatter(data.ma20[start_idx:],y_data)  
    #sys.exit(0)
    #学习1000步  
    x_xais = range(start_idx, len(data.c))
    #print(sess.run([w1, b1, w2, b2]))
    w11, b11 = sess.run([w1, b1])
    print('########################################################')
    lost_value = sess.run(loss,feed_dict={xs:x_data,ys:y_data}) #输出误差  
    print('lost = %f'%lost_value)
    print('w1=', w11)
    print('b1=', b11)
    lines = None
    for i in range(2000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})  
        if i%500==0:
            lost_value = sess.run(loss,feed_dict={xs:x_data,ys:y_data}) #输出误差  
            print('%d = %f'%(i, lost_value))
            if np.isnan(lost_value) or np.isinf(lost_value):
                break

    
    prediction_value=sess.run(prediction,feed_dict={xs:x_data})
     
    lines=ax.plot(x_xais, prediction_value,'r', lw=5)
    w11, b11 = sess.run([w1, b1])
    print(w11)
    plt.show()