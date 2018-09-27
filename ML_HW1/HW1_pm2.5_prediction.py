
# coding: utf-8

# In[2]:


import tensorflow as tf
import keras 
import numpy as np
import pandas as pd


# In[3]:


# data read
data=pd.read_csv('data/train.csv')
pm2_5=data[data['test']=='PM2.5'].ix[:,3:]


# In[4]:


# pre process
tempxlist=[]
tempylist=[]
for i in range(15):
    tempx=pm2_5.iloc[:,i:i+9]        
    tempx.columns=np.array(range(9))
    tempy=pm2_5.iloc[:,i+9]     
    tempy.columns=['1']
    tempxlist.append(tempx)
    tempylist.append(tempy)
# x 
xdata=pd.concat(tempxlist)     
x=np.array(xdata,dtype='f')
print(x.shape[0])
# y
ydata=pd.concat(tempylist)     
ydata.head()
y=(np.array(ydata,dtype='f'))
y = np.reshape(y, [-1,1])
print(y.dtype)

# dimension
x_dim = x.shape[0]


# In[5]:


'''
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras import regularizers

model = Sequential()
model.add(Dense(100,input_dim=9)) # 1 fully connected input 9 parameters
model.add(Activation('linear'))
model.add(Dense(1))
model.add(Activation('relu'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, batch_size=30, epochs=100)


model = Sequential()
model.add(Dense(1,input_dim=9)) # 1 fully connected input 9 parameters
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, batch_size=30, epochs=100)
'''


# In[6]:


# testing x
testing_data = pd.read_csv( 'data/test_X.csv' )
pm25_testing =testing_data[testing_data['test']=='PM2.5'].ix[:,2:]
print(pm25_testing.shape)

testing_x = np.array(pm25_testing,dtype='f')
print(testing_x)

# testing y
testing_y_file = pd.read_csv( 'ans.csv' )
temp_y = testing_y_file['value']
testing_y = np.array(temp_y,dtype='f')
testing_y = np.reshape(testing_y, [-1,1])
print(testing_y.shape)


# In[7]:


'''
# eval
score = model.evaluate(testing_x, testing_y)
print(score)
'''


# In[8]:


# tensorflow
with tf.name_scope('inputs'):
    X = tf.placeholder(tf.float32, [None,9])
    Y = tf.placeholder(tf.float32, [3600,1])

with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        Weights = tf.Variable(tf.random_uniform([9,1], -1.0, 1.0), name='Weight')
    with tf.name_scope('biases'):
        biases = tf.Variable(tf.zeros([9]), name='b')
    
    with tf.name_scope('Wx_plus_b'):
        y_ = tf.matmul(X, Weights)
        y_ = tf.add(y_, tf.transpose(biases))
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y_ - Y))
with tf.name_scope('train'):
    training_step = tf.train.AdagradOptimizer(0.01).minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

for epoch in range(1000000):
    sess.run(training_step,feed_dict={X: x, Y: y})
    if epoch % 10000 == 0:
        print('weight', sess.run(Weights))
        print(' ')
        print('biases', sess.run(biases))
        print(' ')
        
writer.close()


        


# In[23]:


pre_y = sess.run(y_,feed_dict={X: testing_x})
for i in range( 0,201):
    print('testing: ',testing_y[i][0])
    print('predicted: ', pre_y[i][1])
    print('')
    

mse = tf.reduce_mean(tf.square(pre_y - testing_y))
print("MSE: %.4f" % sess.run(mse)) 


# In[20]:




