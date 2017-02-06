# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 20:53:38 2017

@author: Boris Musarais
"""
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

global layer_counter
layer_counter={}

# Create some wrappers for simplicity
def conv1d(x, W, b, strides=1, name=""):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv1d(x, W, stride=strides, padding='SAME', name=name)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def conv2d(x, W, b, strides=1, name=""):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME', name=name)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def conv3d(x, W, b, strides=1, name=""):
    # Conv3D wrapper, with bias and relu activation
    x = tf.nn.conv3d(x, W, strides=[1, strides, strides, strides, 1], padding='SAME', name=name)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool1d(x, k=2, name=""):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, 1], padding='SAME', name=name)
    
def maxpool2d(x, k=2, name=""):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)
    
def maxpool3d(x, reduction_indices=[3], name=""):
    # MaxPool3D wrapper
    return tf.reduce_max(x, reduction_indices=reduction_indices, keep_dims=True, name=name)

class NNOperation:
    def __init__(self, _type, shape=[], bias=[0], param2=0.001 / 9.0, param3=0.75, name="", steps=0):
        self._type = _type
        
        if self._type=="wx+b":
            self._type="local"
        
        self.shape = shape
        self.bias = bias
        self.param2 = param2
        self.param3 = param3
        self.name = name
        self.steps = steps
        
        global layer_counter
        
        if layer_counter.__contains__(name)==False:
            layer_counter[name] = 0
        
        layer_counter[name] += 1

    def getGraph(self, graph):
        global layer_counter
        if len(self.name)==0:
            self.name = self._type
        
        if layer_counter.__contains__(self.name)==False:
            layer_counter[self.name] = 0

        finalName = self._type+"_"+str(layer_counter[self.name])

        obj = None
        if self._type=="relu":
            obj = tf.nn.relu(graph, name=finalName)
        elif self._type=="norm":
            obj = tf.nn.lrn(graph, self.shape, self.bias, self.param2, self.param3, name=finalName)
        elif self._type=="reshape":
            obj = tf.reshape(graph, shape=self.shape, name=finalName)
        elif self._type=="dropout":
            obj = tf.nn.dropout(graph, self.shape, name=finalName)
        elif self._type=="conv1d":
            obj = conv1d(graph, tf.Variable(tf.random_normal(self.shape)), tf.Variable(tf.random_normal([self.shape[2]])), name=finalName)
        elif self._type=="conv2d":
            obj = conv2d(graph, tf.Variable(tf.random_normal(self.shape)), tf.Variable(tf.random_normal([self.shape[3]])), name=finalName)
        elif self._type=="conv3d":
            obj = conv3d(graph, tf.Variable(tf.random_normal(self.shape)), tf.Variable(tf.random_normal([self.shape[4]])), name=finalName)
        elif self._type=="maxpool1d":
            obj = maxpool1d(graph, self.shape, name=finalName)
        elif self._type=="maxpool2d":
            obj = maxpool2d(graph, self.shape, name=finalName)
        elif self._type=="maxpool3d":
            obj = maxpool3d(graph, self.shape, name=finalName)
        elif self._type=="local" or self._type=="wx+b":
            obj = tf.add(tf.matmul(graph, tf.Variable(tf.random_normal(self.shape))), tf.Variable(tf.random_normal(self.bias)), name=finalName)
        elif self._type=="LSTM":
        
            # Define a lstm cell with tensorflow
            cell = tf.nn.rnn_cell.LSTMCell(self.shape)
            #cell = DropoutWrapper(cell, output_keep_prob=dropout)
            #cell = MultiRNNCell([cell] * num_layers)
        
            # Get lstm cell output
            outputs, states = tf.nn.dynamic_rnn(cell, graph, dtype=tf.float32)
        
            # Linear activation, using rnn inner loop last output
            obj =  tf.matmul(outputs[-1], tf.Variable(tf.random_normal([self.shape, self.bias]))) + tf.Variable(tf.random_normal([self.bias]), name=finalName)

        else:
            obj = graph
            print(self._type+" is not implemented")
        print(finalName+": "+str(obj.get_shape()))
            
        return obj
        
class ConvNet:
    def __init__(self, loader, learning_rate = 0.001, training_iters = 200000, batch_size = 128, display_step = 10, dropout=0.75, n_steps=-1):
        global layer_counter
        layer_counter = {}
        
        self.loader = loader
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.batch_size = batch_size
        self.display_step = display_step
        
        # Network Parameters
        if n_steps<=0:
            self.n_input = loader.getImageBytes() # MNIST data input (img shape: 28*28)
        else:
            self.n_input = int(np.sqrt(loader.getImageBytes()))
            
        self.n_classes = loader.LABELS_COUNT # MNIST total classes (0-9 digits)
        self.img_width = int(loader.getImageWidth())
        self.final_img_width = int((loader.getImageWidth()/2)/2)
        self.dropout = dropout # Dropout, probability to keep units
        self.acc_log = []
        self.loss_log = []
        self.n_steps = n_steps
    
    def getDefaultModel(self):
        layers = []
        layers.append(NNOperation("reshape", [-1, self.img_width, self.img_width, 1]))
        layers.append(NNOperation("conv2d", [5, 5, 1, 32]))
        layers.append(NNOperation("maxpool2d", 2))
        layers.append(NNOperation("conv2d", [5, 5, 32, 64]))
        #layers.append(NNOperation("norm", 4, 1, 0.001 / 9.0, 0.75))
        layers.append(NNOperation("maxpool2d", 2))
        layers.append(NNOperation("reshape", [-1, self.final_img_width*self.final_img_width*64]))
        layers.append(NNOperation("local", [self.final_img_width*self.final_img_width*64, 1024], [1024]))
        layers.append(NNOperation("relu"))
        layers.append(NNOperation("dropout", self.dropout))
        layers.append(NNOperation("local", [1024, self.n_classes], [self.n_classes]))
        return layers
        
    def getDefaultLSTM_Model(self):
            layers = []
            layers.append(NNOperation("LSTM", [128, self.n_classes], [self.n_classes], steps=self.n_input))
            return layers
        
    def BuildGraph(self, graph, layers):
        if layers == None:
            layers = self.getDefaultModel()
        
        graph = layers[0].getGraph(graph)
        
        l=1
        while l<len(layers):
            graph = layers[l].getGraph(graph)
            l+=1
            
        return graph
        
    def Run(self, layers=None, save_path="", restore_path="", input_as_label=False):
        
        # tf Graph input
        x = tf.placeholder(tf.float32, [None, self.n_input], name="input_x")
        if self.n_steps>0:
            x = tf.placeholder("float", [None, self.n_steps, int(self.n_input*self.n_input/self.n_steps)], name="input_x")
        elif input_as_label:
         x = tf.placeholder(tf.float32, [None, self.n_input-self.n_classes], name="input_x")
        
        y = tf.placeholder(tf.float32, [None, self.n_classes], name="classes_y")
        keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        
        print("x: "+str(x.get_shape()))
        print("y: "+str(y.get_shape()))
        
        
        # Construct model
        pred = self.BuildGraph(x, layers)
        
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        
        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        # Initializing the variables
        self.init = tf.global_variables_initializer()
        
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(self.init)
            
            if len(restore_path)>0 and os.path.exists(restore_path+".meta"):
                 # Restore model weights from previously saved model
                saver = tf.train.import_meta_graph(restore_path+'.meta')
                load_path=saver.restore(sess, tf.train.latest_checkpoint('temp'))
                print ("Model restored from file: %s" % restore_path)
            
            step = 1
            
            if self.training_iters>300:
                # Keep training until reach max iterations
                while step * self.batch_size < self.training_iters:
                    
                    batch = []
                    
                    if(self.n_steps>0):
                        batch = self.loader.getNextTimeBatch(self.batch_size, n_steps=self.n_steps)
                    else:
                        batch = self.loader.getNextBatch(self.batch_size)
                    self.debug_batch = batch
                    inputs = batch[0]
                    labels = batch[1]                  

                    if input_as_label:
                        
                        inputs = []
                        labels = []

                        for i in range(len(batch[0])):
                            if len(batch[0][i])!=self.n_input:
                                raise Exception("Input length mismtach expected: "+str(self.n_input)+" recieved: "+str(len(batch[0][i])))
                            
                            tmp_x = batch[0][i][0:self.n_input-self.n_classes]
                            tmp_y = batch[0][i][self.n_input-self.n_classes:self.n_input]

                            inputs.append(tmp_x)
                            labels.append(tmp_y)
                         
                    # Run optimization op (backprop)
                    sess.run(optimizer, feed_dict={x: inputs, y: labels, keep_prob: self.dropout})
                    if step % self.display_step == 0:
                        # Calculate batch loss and accuracy
                        loss, acc = sess.run([cost, accuracy], feed_dict={x: inputs,
                                                                          y: labels,
                                                                          keep_prob: 1.})
                        print ("Iter " + str(step*self.batch_size) + ", Minibatch Loss= " \
                        "{:.6f}".format(loss) + ", Training Accuracy= " \
                        "{:.5f}".format(acc))
                        self.acc_log.append(acc)
                        self.loss_log.append(loss)
                    step += 1
            
            batch = self.loader.getTestBatch(50)
            
            if len(save_path)>0:
                # Save model weights to disk
                s_path = saver.save(sess, save_path)
                print ("Model saved in file: %s" % s_path)
                
            result = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1], keep_prob: 1.}) 
            print("input: "+str(batch[1][0])+"result: "+str(result))
            # Calculate accuracy for 256 mnist test images
            #print ("Testing Accuracy:"+\
            #    sess.run(accuracy, feed_dict={x: batch[0],
            #                                  y: batch[1],
            #                                  keep_prob: 1.}) )

    
    def Predict(self, input_v, restore_path, layers=None):
        
        # tf Graph input
        x = tf.placeholder(tf.float32, [self.n_input], name="input_x")
        if self.n_steps>0:
            x = tf.placeholder("float", [None, self.n_steps, int(self.n_input*self.n_input/self.n_steps)], name="input_x")
        y = tf.placeholder(tf.float32, [None, self.n_classes], name="classes_y")
        keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        
        pred = self.BuildGraph(x, layers)
        
         # Initializing the variables
        self.init = tf.global_variables_initializer()
        
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(self.init)
            
            if len(restore_path)>0 and os.path.exists(restore_path+".meta"):
                 # Restore model weights from previously saved model
                saver = tf.train.import_meta_graph(restore_path+'.meta')
                load_path=saver.restore(sess, tf.train.latest_checkpoint('temp'))
                print ("Model restored from file: %s" % load_path)                
                
                result = sess.run(pred, feed_dict={x: input_v}) 
                sess.close()
                return result
            else:
                print ("Not found: " + restore_path)
        return None
	
    def Generate(self, input_v, restore_path, layers=None, iterations=10, multiplier=1):
        
        generation_result = []
		
        # tf Graph input
        x = tf.placeholder(tf.float32, [self.n_input], name="input_x")
        if self.n_steps>0:
            x = tf.placeholder("float", [None, self.n_steps, int(self.n_input*self.n_input/self.n_steps)], name="input_x")
        y = tf.placeholder(tf.float32, [None, self.n_classes], name="classes_y")
        keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        
        pred = self.BuildGraph(x, layers)
        
         # Initializing the variables
        self.init = tf.global_variables_initializer()
        
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(self.init)
            
            if len(restore_path)>0 and os.path.exists(restore_path+".meta"):
                 # Restore model weights from previously saved model
                saver = tf.train.import_meta_graph(restore_path+'.meta')
                load_path=saver.restore(sess, tf.train.latest_checkpoint('temp'))
                print ("Model restored from file: %s" % load_path)                
                
                for i in range(iterations):
                    result = sess.run(pred, feed_dict={x: input_v}) 
                    
                    if self.n_steps>0:
                        input_v.pop(0)
                        input_v.append(list(result[0]))
                    else:
                        
                        for r in range(len(result[0])):
                            result[0][r]*=multiplier
                        
                        input_v = input_v + list(result[0])
                        input_v = input_v[self.n_classes:]
                    generation_result = generation_result + list(result[0])
                    if i%100==0:
                        print("status: "+str(i)+"/"+str(iterations)+" "+str(input_v[0])+" "+str(len(generation_result)))
            else:
                print ("Not found: " + restore_path)
        sess.close()
        return generation_result
                
    def Plot(self):
        plt.plot(self.acc_log)
        plt.ylabel="Accuracy"
        plt.show()
        plt.plot(self.loss_log, color="red")
        plt.ylabel="Loss"
        plt.show()