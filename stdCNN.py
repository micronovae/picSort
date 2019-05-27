# -*- coding: utf-8 -*-
import tensorflow as tf
import cv2
import os
import numpy as np

def getList(file_path):
    
    print('getList is ok')
    
    DataToPros = os.listdir(file_path+'/')
    
    return DataToPros

def getFile(file_path,file_list):
    
    print('getFile is ok')
    
    data = []
    label = []
    for i in file_list:
        data.append(file_path+'/'+i)
        if file_path == 'pics_modal/rumor_pics':
            label.append(0)
        else:
            label.append(1)
            
    return data,label

def proList():
    
    print('proList is ok')

    file_path_rumor = "pics_modal/rumor_pics"
    file_path_truth = "pics_modal/truth_pics"
    list_rumor = getList(file_path_rumor)
    list_truth = getList(file_path_truth)
    
    rd,rl = getFile(file_path_rumor,list_rumor[:20])
    td,tl = getFile(file_path_truth,list_truth[:20])
    
    prodata = rd+td
    prolabel = rl+tl
    
    temp = np.array([prodata,prolabel])
    temp = temp.T
    np.random.shuffle(temp)
    
    prodata = list((temp[:,0]))
    prolabel = list((temp[:,1]))
    prolabel = [int(float(i)) for i in prolabel]

    return prodata, prolabel

def getBat(data_list,label_list,width,height,batSize,capacity):
    
    print('getBat is ok')

    image = []
    
    for i in data_list:
        
        temp = tf.read_file(str(i))
        temp = tf.image.decode_jpeg(temp,channels = 3)
        temp = tf.image.resize_image_with_crop_or_pad(
          temp,width,height)
        temp = tf.image.per_image_standardization(temp)
        image.append(temp)
        
    label = tf.cast(label_list,tf.int32)
    image = tf.cast(image,tf.float32)

    return image,label

def init_weights(shape):
    res = tf.Variable(tf.random_normal(shape,stddev = 0.01))
    return res

def norm(x,lsize = 4):
    return tf.nn.lrn(x,depth_radius = lsize,bias = 1,
                      alpha = 0.001/9.0,beta = 0.75)

def conv2d(x, w, b):
    x = tf.nn.conv2d(x,w,strides = [1,1,1,1],padding = "SAME")
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)  

def max_pool_2x2(x):
    return tf.nn.max_pool(x, 
      ksize = [1, 2, 2, 1],strides = [1, 2, 2, 1], padding = 'SAME')

def loss(logits,label_batches):
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=label_batches)
    loss = tf.reduce_mean(cross_entropy)
    
    return loss
 
def get_accuracy(logits,labels):
    
    acc = tf.nn.in_top_k(logits,labels,1)
    acc = tf.cast(acc,tf.float32)
    acc = tf.reduce_mean(acc)
    
    return acc

def mymodel(data):

    # init w      
    weights = {
      "w1":init_weights([3,3,3,16]),
      "w2":init_weights([3,3,16,128]),
      "w3":init_weights([3,3,128,256]),
      "w4":init_weights([4096,4096]),
      "wo":init_weights([4096,2])}
     
    # init biases
    biases = {
     "b1":init_weights([16]),
     "b2":init_weights([128]),
     "b3":init_weights([256]),
     "b4":init_weights([4096]),
     "bo":init_weights([2])
     }


    layer1 = conv2d(data,weights["w1"],biases["b1"])
    layer2 = max_pool_2x2(layer1)
    layer2 = tf.nn.lrn(layer2)
    layer3 = conv2d(layer2,weights["w2"],biases["b2"])
    layer4 = max_pool_2x2(layer3)
    layer4 = tf.nn.lrn(layer4)
    layer5 = conv2d(layer4,weights["w3"],biases["b3"])
    layer6 = max_pool_2x2(layer5)
    layer6 = tf.reshape(layer6,[-1,weights["w4"].get_shape().as_list()[0]])
    layer7 = tf.nn.relu(tf.matmul(layer6,weights["w4"])+biases["b4"])
    softmax = tf.add(tf.matmul(layer7,weights["wo"]),biases["bo"])
    
    return softmax

def training(loss,lr):
    train_op = tf.train.RMSPropOptimizer(lr,0.9).minimize(loss)
    return train_op

def cnn_run(data_bat,label_bat):
    log_dir = '/Users/yanghang/Desktop/Dataset/picSort/'
    p = mymodel(data_bat)
    cost = loss(p,label_bat)
    train_op = training(cost,0.01)
    acc = get_accuracy(p,label_bat)
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    try:
        for step in np.arange(20):
            print(step)
            _,train_acc,train_loss = sess.run([train_op,acc,cost])
            print("loss:{} accuracy:{}".format(train_loss,train_acc))
            if step % 100 == 0:
                check = os.path.join(log_dir,"model.ckpt")
                saver.save(sess,check,global_step = step)
                
    except tf.errors.OutOfRangeError:
        print('Wrong!!!')

    sess.close()

    
def main():
    
    data, label = proList()
    data_bat, label_bat = getBat(data,label,32,32,5,64)
    cnn_run(data_bat,label_bat)

if __name__ == 'main':
    main()













