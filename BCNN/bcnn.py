from __future__ import print_function
import tensorflow as tf
import numpy as np
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import os
from tflearn.data_utils import shuffle
from skimage import transform

import pickle 
from tflearn.data_utils import image_preloader
import h5py
import math
import logging
import random
import time
import scipy




def random_flip_right_to_left(image_batch):
    result = []
    for n in range(image_batch.shape[0]):
        if bool(random.getrandbits(1)):
            result.append(image_batch[n][:,::-1,:])
        elif bool(random.getrandbits(1)):
            result.append(image_batch[n][::-1,:,:])    
        else:
            result.append(image_batch[n])
    return result

def random_crop(image_batch):
    result = []
    for n in range(image_batch.shape[0]):
        start_x = random.randint(0,19)
        start_y = random.randint(0,19)
        result.append(scipy.misc.imresize(image_batch[n][start_y:start_y+224,start_x:start_x+224,:],(224,224,3)))
    return np.array(result)

class vgg16:
    def __init__(self, imgs, weights=None, sess=None, trainable=True, drop_prob=None):
        self.imgs = imgs
        self.last_layer_parameters = []     
        self.parameters = []                
        self.convlayers(trainable)          
        self.fc_layers()                    
        self.weight_file = weights           
        self.drop_prob=drop_prob       
        #self.load_weights(weights, sess)


    def convlayers(self,trainable):
        
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean
            print('Adding Data Augmentation')
            

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainable=trainable, name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                   trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainable=trainable, name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64],  dtype=tf.float32),
                                   trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainable=trainable,  name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                   trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainable=trainable,  name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1),  trainable=trainable, name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                   trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=trainable,  name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                   trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1),  trainable=trainable,  name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                   trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=trainable,  name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                  trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=trainable,   name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                  trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=trainable,  name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                  trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1),  trainable=trainable, name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                  trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=trainable,  name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                  trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=trainable,  name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                  trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        self.InnerPro = tf.einsum('ijkm,ijkn->imn',self.conv5_3,self.conv5_3)
        self.InnerPro = tf.reshape(self.InnerPro,[-1,512*512])
        self.InnerPro = tf.divide(self.InnerPro,14.0*14.0)  
        self.ySsqrt = tf.multiply(tf.sign(self.InnerPro),tf.sqrt(tf.abs(self.InnerPro)+1e-12))
        self.zL2 = tf.nn.l2_normalize(self.ySsqrt, dim=1)




    def fc_layers(self):

        with tf.name_scope('fc') as scope:

            fcw = tf.get_variable('weights', [512*512, 200], 
                      initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            fcb = tf.Variable(tf.constant(1.0, shape=[200], dtype=tf.float32),
                      name='biases', trainable=True)
            self.fcl = tf.nn.bias_add(tf.matmul(self.zL2, 
                        tf.contrib.layers.dropout(fcw,self.drop_prob)), 
                        tf.contrib.layers.dropout(fcb,self.drop_prob))
            self.last_layer_parameters += [fcw, fcb]
            self.parameters += [fcw, fcb]

    def load_weights(self, sess):
        #saver=tf.train.Saver(self.parameters)
        #save_path="save"
        #saver.restore(sess,save_path)
        weights = np.load(self.weight_file)
        #return 
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            removed_layer_variables = ['fc6_W','fc6_b','fc7_W','fc7_b','fc8_W','fc8_b']
            if not k in removed_layer_variables:
                print(k)
                print("",i, k, np.shape(weights[k]))
                sess.run(self.parameters[i].assign(weights[k]))

if __name__ == '__main__':

    train_data = h5py.File('new_train.h5', 'r')
    val_data = h5py.File('new_val.h5', 'r')
    

    print('Input data read complete')

    X_train, Y_train = train_data['X'], train_data['Y']
    X_val, Y_val = val_data['X'], val_data['Y']
    valsize = len(X_val)
    valsize = int(valsize*0.1)
    X_val, Y_val = shuffle(X_val, Y_val)
    X_val, Y_val = X_val[0:32], Y_val[0:32]    
    print("Data shapes -- (train, val)", X_train.shape, X_val.shape)
    X_train, Y_train = shuffle(X_train, Y_train)
    
    
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    target = tf.placeholder("float", [None, 200])
    drop_prob = tf.placeholder("float")
    isFineTune=False
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess, isFineTune, drop_prob)

    
    
    print('VGG network created')
    
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vgg.fcl, labels=target))
    learning_rate_wft = tf.placeholder(tf.float32, shape=[])
    learning_rate_woft = tf.placeholder(tf.float32, shape=[])
    
    if isFineTune:
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(loss)
    else: optimizer = tf.train.MomentumOptimizer(learning_rate=0.9, momentum=0.9).minimize(loss)
    ###optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(vgg.fcl,1), tf.argmax(target,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())
    
    vgg.load_weights(sess)
    
    batch_size = 32


    print('Starting training')

    lr = 1.0
    finetune_step = -1
    base_lr = 1.0
    break_training_epoch = 15
    best_accuracy = 0.0
    for epoch in range(100):
        avg_cost = 0.
        total_batch = int(len(X_train)/batch_size)
        X_train, Y_train = shuffle(X_train, Y_train)
        



        for i in range(total_batch):
            batch_xs, batch_ys = X_train[i*batch_size:i*batch_size+batch_size], Y_train[i*batch_size:i*batch_size+batch_size]
            
            batch_xs = random_crop(batch_xs)
            batch_xs = random_flip_right_to_left(batch_xs)

            start = time.time()
            [cp,cost,opt] = sess.run([correct_prediction,loss,optimizer], feed_dict={imgs: batch_xs, target: batch_ys, drop_prob:0.8})
            if i%20==0:
                print('Last layer training, time to run optimizer for batch size:', batch_size,'is --> ',time.time()-start,'seconds',"loss:",cost,"correct_prediction",cp)

            if i % 100 == 0:
                #print ('Learning rate: ', (str(lr)))
                if epoch <= finetune_step:
                    print("Training last layer of BCNN_DD")

                print("Epoch:", '%03d' % (epoch+1), "Step:", '%03d' % i,"Loss:", str(cost))
                print("Training Accuracy:", sess.run([correct_prediction,accuracy],feed_dict={imgs: batch_xs, target: batch_ys, drop_prob:0.8}))

                val_batch_size = 32
                total_val_count = len(X_val)
                correct_val_count = 0
                val_loss = 0.0
                total_val_batch = int(total_val_count/val_batch_size)
                for i in range(total_val_batch):
                    batch_val_x, batch_val_y = X_val[i*val_batch_size:i*val_batch_size+val_batch_size], Y_val[i*val_batch_size:i*val_batch_size+val_batch_size]
                    val_loss += sess.run(loss, feed_dict={imgs: batch_val_x, target: batch_val_y, drop_prob:1.0})

                    pred = sess.run(num_correct_preds, feed_dict = {imgs: batch_val_x, target: batch_val_y, drop_prob:1.0})
                    correct_val_count+=pred
                acc = 100.0*correct_val_count/(1.0*total_val_count)
                if acc>best_accuracy:
                    best_accuracy = acc
                    saver=tf.train.Saver(vgg.parameters)
                    save_path="save"
                    saver.save(sess,save_path)
                print("##############################")
                print("Validation Loss:", val_loss)
                print("correct_val_count,", correct_val_count, "total_val_count", total_val_count)
                print("Validation Data Accuracy:", acc)
                print("##############################")

        

    test_data = h5py.File('new_test.h5', 'r')
    X_test, Y_test = test_data['X'], test_data['Y']
    total_test_count = len(X_test)
    correct_test_count = 0
    test_batch_size = 10
    total_test_batch = int(total_test_count/test_batch_size)
    for i in range(total_test_batch):
        batch_test_x, batch_test_y = X_test[i*test_batch_size:i*test_batch_size+test_batch_size], Y_test[i*test_batch_size:i*test_batch_size+test_batch_size]
        
        pred = sess.run(num_correct_preds, feed_dict = {imgs: batch_test_x, target: batch_test_y, drop_prob:1.0})
        correct_test_count+=pred

    print("##############################")
    print("correct_test_count,",correct_test_count," total_test_count",  total_test_count)
    print("accuracy :", 100.0*correct_test_count/(1.0*total_test_count))
    print("##############################")



