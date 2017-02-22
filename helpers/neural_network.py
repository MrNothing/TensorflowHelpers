# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 20:53:38 2017

@author: Boris Musarais
"""
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from helpers.sound_tools import Encoder
import random as rand

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

    
class RNNOperation:
    def __init__(self, cells=[32], n_classes=None, dropout=0.75, name=""):
        self._type = "RNN"
        self.bias = n_classes
        self.cells = cells
        self.dropout = dropout
        self.name = name
        
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
        
        cells = []
        for cell_n_hidden in self.cells:
            t_cell = tf.nn.rnn_cell.RNNCell(cell_n_hidden, state_is_tuple=True)
            t_cell = tf.nn.rnn_cell.DropoutWrapper(t_cell, output_keep_prob=self.dropout)
            cells.append(t_cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            
        val, _ = tf.nn.dynamic_rnn(cell, graph, dtype=tf.float32)
        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)
        weight = tf.Variable(tf.truncated_normal([self.cells[len(self.cells)-1], self.bias]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.bias]))
        obj = tf.matmul(last, weight) + bias

        print(finalName+": "+str(self.cells)+" => "+str(obj.get_shape()))

        return obj
    
    
class BIRNNOperation:
    def __init__(self, cells=[32], n_classes=None, dropout=0.75, name="", cell_type="RNN"):
        self._type = "BI_Directionnal_RNN"
        self.bias = n_classes
        self.cells = cells
        self.dropout = dropout
        self.name = name
        self.cell_type = cell_type
        
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
        
        cellConstructor = tf.nn.rnn_cell.RNNCell
        if self.cell_type=="LSTM":
            cellConstructor = tf.nn.rnn_cell.BasicLSTMCell
        elif self.cell_type=="GRU":
            cellConstructor = tf.nn.rnn_cell.GRUCell
            
        cells = []
        for cell_n_hidden in self.cells:
            t_cell = cellConstructor(cell_n_hidden)
            cells.append(t_cell)
        lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        
        cells = []
        for cell_n_hidden in self.cells:
            t_cell = cellConstructor(cell_n_hidden)
            cells.append(t_cell)
        lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

            
        val, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, graph, dtype=tf.float32)
        self.state = state
        
        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)
        weight = tf.Variable(tf.truncated_normal([self.cells[len(self.cells)-1], self.bias]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.bias]))
        obj = tf.matmul(last, weight) + bias

        print(finalName+": "+str(self.cells)+" => "+str(obj.get_shape()))

        return obj
        
class GRUOperation:
    def __init__(self, cells=[32], n_classes=None, dropout=0.75, name="", batch_size=128):
        self._type = "GRU"
        self.bias = n_classes
        self.cells = cells
        self.dropout = dropout
        self.name = name
        self.batch_size = batch_size
        
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
        
        cells = []
        for cell_n_hidden in self.cells:
            t_cell = tf.nn.rnn_cell.GRUCell(cell_n_hidden)
            t_cell = tf.nn.rnn_cell.DropoutWrapper(t_cell, output_keep_prob=self.dropout)
            cells.append(t_cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            
        val, self.state = tf.nn.dynamic_rnn(cell, graph, dtype=tf.float32)
        
        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)
        weight = tf.Variable(tf.truncated_normal([self.cells[len(self.cells)-1], self.bias]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.bias]))
        obj = tf.matmul(last, weight) + bias

        print(finalName+": "+str(self.cells)+" => "+str(obj.get_shape()))

        return obj
    
# cells = [512, 128, 32]
class LSTMOperation:
    def __init__(self, cells=[32], n_classes=None, dropout=0.75, name=""):
        self._type = "LSTM"
        self.bias = n_classes
        self.cells = cells
        self.dropout = dropout
        self.name = name
        
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
        
        cells = []
        for cell_n_hidden in self.cells:
            t_cell = tf.nn.rnn_cell.LSTMCell(cell_n_hidden, state_is_tuple=True)
            t_cell = tf.nn.rnn_cell.DropoutWrapper(t_cell, output_keep_prob=self.dropout)
            cells.append(t_cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            
        val, _ = tf.nn.dynamic_rnn(cell, graph, dtype=tf.float32)
        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)
        weight = tf.Variable(tf.truncated_normal([self.cells[len(self.cells)-1], self.bias]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.bias]))
        obj = tf.matmul(last, weight) + bias

        print(finalName+": "+str(self.cells)+" => "+str(obj.get_shape()))

        return obj
    
class NNOperation:
    def __init__(self, _type, shape=[], bias=[0], param2=0.001 / 9.0, param3=0.75, name="", steps=0, n_input=0):
        self._type = _type
        
        if self._type=="wx+b":
            self._type="local"
        
        self.shape = shape
        self.bias = bias
        self.param2 = param2
        self.param3 = param3
        self.name = name
        self.steps = steps
        self.n_input = n_input
        
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
            
            num_hidden = self.shape
            
              # Define a lstm cell with tensorflow
            cell = tf.nn.rnn_cell.LSTMCell(self.shape,state_is_tuple=True)
           
            val, _ = tf.nn.dynamic_rnn(cell, graph, dtype=tf.float32)
            val = tf.transpose(val, [1, 0, 2])
            last = tf.gather(val, int(val.get_shape()[0]) - 1)
            weight = tf.Variable(tf.truncated_normal([num_hidden, self.bias]))
            bias = tf.Variable(tf.constant(0.1, shape=[self.bias]))
            obj = tf.matmul(last, weight) + bias
            
            # Permuting batch_size and n_steps
            #graph = tf.transpose(graph, [1, 0, 2])
            # Reshaping to (n_steps*batch_size, n_input)
            #graph = tf.reshape(graph, [-1, self.n_input])
            # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            #graph = tf.split(0, self.steps, graph)
        
            # Define a lstm cell with tensorflow
            #lstm_cell = tf.nn.rnn_cell.LSTMCell(self.shape, forget_bias=1.0)
        
            # Get lstm cell output
            #outputs, states = tf.nn.rnn(lstm_cell, graph, dtype=tf.float32)
        
            # Linear activation, using rnn inner loop last output
            #obj =  tf.matmul(outputs[-1], tf.Variable(tf.random_normal([self.shape, self.bias]))) + tf.Variable(tf.random_normal([self.bias]), name=finalName)

        else:
            obj = graph
            print(self._type+" is not implemented")
        print(finalName+": "+str(obj.get_shape()))
            
        return obj
        
class ConvNet:
    def __init__(self, 
                 loader, 
                 learning_rate = 0.001, 
                 learning_decrease = 0,
                 training_iters = 200000, 
                 batch_size = 128, 
                 display_step = 10, 
                 dropout=0.75, 
                 n_steps=-1, 
                 save_step = 10000,
                 decay_step = 100000,
                 decay_rate = 0.95):
        global layer_counter
        layer_counter = {}
        
        self.learning_decrease = 1-learning_decrease
        self.loader = loader
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.batch_size = batch_size
        self.display_step = display_step
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.save_step = save_step
        
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
        
    def Run(self, layers=None, save_path="", restore_path="", input_as_label=False, x=None, y=None):
        
        # tf Graph input
        if x==None:
            x = tf.placeholder(tf.float32, [None, self.n_input], name="input_x")
            if self.n_steps>0:
                x = tf.placeholder("float", [None, self.n_steps, int((self.n_input*self.n_input)/self.n_steps)], name="input_x")
            elif input_as_label:
             x = tf.placeholder(tf.float32, [None, self.n_input-self.n_classes], name="input_x")
        
        if y==None:
            y = tf.placeholder(tf.float32, [None, self.n_classes], name="classes_y")
            
        keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        
        print("x: "+str(x.get_shape()))
        print("y: "+str(y.get_shape()))
        
        
        # Construct model
        pred = self.BuildGraph(x, layers)
        
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
       
        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        global_step = tf.Variable(0)
        
        learning_rate = tf.train.exponential_decay(
          self.learning_rate,               # Base learning rate.
          global_step*self.batch_size,            # Current index into the dataset.
          self.decay_step,           # Decay step.
          self.decay_rate,                             # Decay rate.
          staircase=True)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
        
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
                load_path=saver.restore(sess, tf.train.latest_checkpoint(restore_path))
                print ("Model restored from file: %s" % restore_path)
            
            step = 1
            
            self.labels_log = []
            
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

                    self.labels_log.append(self.max_index(labels[0]))              

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
                        "{:.5f}".format(acc) + ", Learning Rate= " + str(learning_rate.eval()))
                        self.acc_log.append(acc)
                        self.loss_log.append(loss)
                    if step % self.save_step == 0:
                         if len(save_path)>0:
                            # Save model weights to disk
                            s_path = saver.save(sess, save_path+"/model")
                            print ("Model saved in file: %s" % s_path)
                            
                    step += 1
                    
            batch = []

            if len(save_path)>0:
                # Save model weights to disk
                s_path = saver.save(sess, save_path+"/model")
                print ("Model saved in file: %s" % s_path)
            
            if(self.n_steps==-1):
               batch = self.loader.getTestBatch(50)
               result = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1], keep_prob: 1.}) 
               print("test result: "+str(result))
            
            sess.close()
    
    def Predict(self, input_v, restore_path, layers=None, x=None):
        
        # tf Graph input
        if x==None:
            x = tf.placeholder(tf.float32, [self.n_input], name="input_x")
            if self.n_steps>0:
                x = tf.placeholder("float", [None, self.n_steps, int((self.n_input*self.n_input)/self.n_steps)], name="input_x")
        
        pred = self.BuildGraph(x, layers)
        
         # Initializing the variables
        self.init = tf.global_variables_initializer()
        
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(self.init)
            
            if len(restore_path)>0 and os.path.exists(restore_path+"/model.meta"):
                 # Restore model weights from previously saved model
                #saver = tf.train.import_meta_graph(restore_path+'/model.meta')
                load_path=saver.restore(sess, restore_path+"/model")
                print ("Model restored from file: %s" % load_path)                
                
                if self.n_steps>0:
                    result = sess.run(pred, feed_dict={x: [input_v]}) 
                    sess.close()
                    return result
                else:
                    result = sess.run(pred, feed_dict={x: input_v}) 
                    sess.close()
                    return result
            else:
                print ("Not found: " + restore_path)
        return None
	
    def Generate(self, 
                 input_v, 
                 restore_path, 
                 layers=None, 
                 x=None, 
                 iterations=10, 
                 multiplier=1, 
                 use_sample_state=False, 
                 sample_state_offset=0, 
                 display_step=100, 
                 sample_length=15, 
                 epsilon = 0.01,
                 label_offset=0):
        
        generation_result = []
        for v in input_v:
            generation_result+=v
		
        # tf Graph input
        if x==None:
            x = tf.placeholder(tf.float32, [self.n_input], name="input_x")
            
            if self.n_steps>0:
                x = tf.placeholder("float", [None, self.n_steps, int((self.n_input*self.n_input)/self.n_steps)], name="input_x")
        
        #y = tf.placeholder(tf.float32, [None, self.n_classes], name="classes_y")
        #keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        
        pred = self.BuildGraph(x, layers)
        
         # Initializing the variables
        self.init = tf.global_variables_initializer()
        
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(self.init)
            
            if len(restore_path)>0 and os.path.exists(restore_path+"/model.meta"):
                # Restore model weights from previously saved model
                #saver = tf.train.import_meta_graph(restore_path+'/model.meta')
                load_path=saver.restore(sess, restore_path+"/model")
                print ("Model restored from file: %s" % load_path)           
                #load_path=saver.restore(sess, restore_path+'/model.ckpt.data-1000-00000-of-00001')
                
                sample_state = sample_state_offset
                last_val=[]
                for o in range(label_offset):
                    last_val.append(0.5)
                
                for i in range(iterations):
                    result = sess.run(pred, feed_dict={x: [input_v]}) 
                    r_sample_state = sample_state
                    
                    if self.loader.one_hot!=-1:
                        next_val = self.max_index(result[0])/self.loader.one_hot
                        generation_result.append(next_val)
                        
                        if self.loader.uLawEncode!=-1:
                            next_val = Encoder.uLawEncode(next_val, self.loader.uLawEncode)
                            
                        next_val += epsilon*rand.uniform(0.0, 1.0)
                        
                        if self.n_steps>0:
                            if label_offset>0:
                                input_v = self.deep_push(input_v, last_val[0])
                            else:
                                input_v = self.deep_push(input_v, next_val)
                            
                            if use_sample_state:
                                for s in range(self.loader.insert_global_input_state):
                                    input_v[s] = [r_sample_state]*len(input_v[1])
                        else:
                            input_v.pop(0)
                            
                            if label_offset>0:
                                input_v.append(last_val[0])
                            else:
                                input_v.append(next_val)
                            
                            if use_sample_state:
                                input_v[0] = r_sample_state

                        last_val.append(next_val)
                        last_val.pop(0)
                        
                    else:
                        result[0]=self.normalize(result[0])
                        
                        if self.n_steps>0:
                            if use_sample_state:
                                res = tf.reshape(result, [-1, 4])
                                input_v = self.fuse(input_v[1+8:len(input_v)],res.eval())
                                input_v.insert(0, [r_sample_state]*len(input_v[1]))
                            else:
                                input_v.pop(0)
                                input_v.append(list(result[0]))
                        else:
                            input_v = input_v + list(result[0])
                            input_v = input_v[self.n_classes:]

                        generation_result = generation_result + list(result[0])

                    if i%display_step==0:
                        print("sample_state: "+str(r_sample_state))
                        if self.n_steps>0:
                            print("status: "+str(i)+"/"+str(iterations)+" "+str(input_v[len(input_v)-1][0:10])+" "+str(len(generation_result)))
                        else:
                            print("status: "+str(i)+"/"+str(iterations)+" "+str(input_v[0:10])+" "+str(len(generation_result)))
                        
                    sample_state+=1/len(self.loader.converter.data)
            else:
                print ("Not found: " + restore_path)
        sess.close()
        return generation_result
        
    def TrainAndGenerate(self, 
                         layers=None, 
                         save_path="", 
                         restore_path="", 
                         input_as_label=False, 
                         x=None,
                         iterations=10, 
                         multiplier=1, 
                         use_sample_state=False, 
                         sample_state_offset=0, 
                         display_step=100, 
                         sample_length=15, 
                         epsilon = 0.01):
        
         # tf Graph input
        if x==None:
            x = tf.placeholder(tf.float32, [None, self.n_input], name="input_x")
            if self.n_steps>0:
                x = tf.placeholder("float", [None, self.n_steps, int((self.n_input*self.n_input)/self.n_steps)], name="input_x")
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
                load_path=saver.restore(sess, tf.train.latest_checkpoint(restore_path))
                print ("Model restored from file: %s" % restore_path)
            
            step = 1
            
            self.labels_log = []
            
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

                    self.labels_log.append(self.max_index(labels[0]))              

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
                    if step % self.save_step == 0:
                         if len(save_path)>0:
                            # Save model weights to disk
                            s_path = saver.save(sess, save_path+"/model")
                            print ("Model saved in file: %s" % s_path)
                            
                    step += 1
                    
            batch = []

            if len(save_path)>0:
                # Save model weights to disk
                s_path = saver.save(sess, save_path+"/model")
                print ("Model saved in file: %s" % s_path)
            
            if(self.n_steps==-1):
               batch = self.loader.getTestBatch(50)
               result = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1], keep_prob: 1.}) 
               print("test result: "+str(result))
            
            ########## GENERATION ##########
            sample_state = sample_state_offset
            generation_result = []

            input_v = self.loader.getNextTimeBatch(1, n_steps=self.n_steps)[0][0]
            
            for i in range(iterations):
                result = sess.run(pred, feed_dict={x: [input_v]}) 
                
                if self.loader.uLawEncode:
                    r_sample_state = Encoder.uLawEncode(sample_state, self.loader.uLawEncode)
                else:
                    r_sample_state = sample_state
                    
                if self.loader.one_hot!=-1:
                    next_val = self.max_index(result[0])/self.loader.one_hot
                    generation_result.append(next_val)
                    
                    if self.loader.uLawEncode!=-1:
                        next_val = Encoder.uLawEncode(next_val, self.loader.uLawEncode)
                        
                    next_val += epsilon*rand.uniform(0.0, 1.0)
                    
                    if self.n_steps>0:
                        input_v = self.deep_push(input_v, next_val)
                    
                        if use_sample_state:
                            for s in range(self.loader.insert_global_input_state):
                                input_v[s] = [r_sample_state]*len(input_v[1])
                    else:
                        input_v.pop(0)
                        input_v.append(next_val)
                        
                        if use_sample_state:
                            input_v[0] = r_sample_state
                else:
                    result[0]=self.normalize(result[0])
                    
                    if self.n_steps>0:
                        if use_sample_state:
                            res = tf.reshape(result, [-1, 4])
                            input_v = self.fuse(input_v[1+8:len(input_v)],res.eval())
                            input_v.insert(0, [r_sample_state]*len(input_v[1]))
                        else:
                            input_v.pop(0)
                            input_v.append(list(result[0]))
                    else:
                        input_v = input_v + list(result[0])
                        input_v = input_v[self.n_classes:]
    
                    generation_result = generation_result + list(result[0])
    
                if i%display_step==0:
                    print("sample_state: "+str(r_sample_state))
                    if self.n_steps>0:
                        print("status: "+str(i)+"/"+str(iterations)+" "+str(input_v[len(input_v)-1][0:10])+" "+str(len(generation_result)))
                    else:
                        print("status: "+str(i)+"/"+str(iterations)+" "+str(input_v[0:10])+" "+str(len(generation_result)))
                    
                sample_state+=1/len(self.loader.converter.data)
                
        sess.close()
        return generation_result
            
            
    def Plot(self):
        plt.plot(self.acc_log)
        plt.ylabel("Accuracy")
        plt.show()
        plt.plot(self.loss_log, color="red")
        plt.ylabel("Loss")
        plt.show()
        plt.plot(self.labels_log, color="green")
        plt.ylabel("Labels repartition")
        
        plt.show()
        
    def normalize(self=None, data=[]):
        _max = 0
        for i in range(len(data)):
            if data[i]>_max:
                _max = abs(data[i])

        for i in range(len(data)):
            data[i] = data[i]/_max
        
        return data
        
    def max_index(self=None, data=[]):
        _max = 0
        _max_index = -1
        for i in range(len(data)):
            if data[i]>_max:
                _max_index = i
                _max = abs(data[i])
                
        return _max_index
        
    def fuse(self, a, b):
        for i in b:
            a.append(i)
        return a
        
    def deep_push(self, input_v, val):
        for i in range(len(input_v)):
            if(i<len(input_v)-1):
                input_v[i].pop(0)
                input_v[i].append(input_v[i+1][0])
            elif (i>=len(input_v)-1):
                input_v[i].pop(0)
                input_v[i].append(val)
        return input_v
        