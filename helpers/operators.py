# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 00:26:11 2017

@author: Boris
"""

import tensorflow as tf
import numpy as np
import random as rand
import os
import pickle

global layer_counter
layer_counter = {}

# Create some wrappers for simplicity
def conv1d(x, W, b, strides=1, name=""):
    # Conv1D wrapper, with bias and relu activation
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
    # MaxPool1D wrapper
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
        self.state = None
        
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
            
        with tf.variable_scope(finalName):
	        val, self.state = tf.nn.dynamic_rnn(cell, graph, dtype=tf.float32)
        
        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)
        weight = tf.Variable(tf.truncated_normal([self.cells[len(self.cells)-1], self.bias]), name=self.name+"_weight")
        bias = tf.Variable(tf.constant(0.1, shape=[self.bias]), name=self.name+"_bias")
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
        
        with tf.variable_scope(finalName):
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
        elif self._type=="local" or self._type=="wx+b" or self._type=="dense":
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
								
class FeatureMap:
    def __init__(self,  
                 extract_length = 16, 
                 path="", 
                 z_size = 3, 
                 samplerate=441*10, 
                 one_hot=0, 
                 future_labels=False,
                 entropy=None,
                 ):
        self.feature_map = self.load_output(path)
        self.init_feature_map = self.feature_map
								
        self.one_hot = one_hot
        self.z_size = z_size
        self.normalize_map()
			
        self.extract_length = extract_length
        self.fixed_size = 0
        self.LABELS_COUNT = z_size
        self.samplerate = samplerate
        self.cache = {}
        self.future_labels = future_labels
        self.entropy = entropy
		
    def getNextFeatureBatch(self, batch_size, n_reccurent_input=0, one_hot=False):
        return self.getNextBatch(batch_size, n_reccurent_input, one_hot)
        
    def getNextBatch(self, batch_size, n_reccurent_input=0, one_hot=False):
        batch = []
        labels = []
        prev_batch = []

        for i in range(batch_size):
            
            start_point = rand.randint(0, len(self.feature_map)-self.extract_length)
    
            if self.cache.__contains__(start_point):
                batch.append(self.cache[start_point][0])
                labels.append(self.cache[start_point][1])
                prev_batch.append(self.cache[start_point][2])
            else:
                if i==0:
                    print("cache: "+str(len(self.cache))+"/"+str(len(self.feature_map)))
                    
                _input = []
                label = None
                _prev = None
                
                if self.one_hot>0:
                    #input
                    batch.append(self.extract_map(start_point, start_point+self.extract_length))
                    
                    #label
                    sample = self.extract_map(start_point+self.extract_length, start_point+self.extract_length+1)[0]
                    label = [0]*self.one_hot*self.z_size             
                    for s in range(len(sample)):
                        index = int(sample[s]*self.one_hot)
                        if(index>self.one_hot-1):
                            index = self.one_hot-1
                        elif(index<0):
                            index=0
                        index+=s*self.one_hot
                        label[index] = 1
                    
                    labels.append(label)                
    
                    if n_reccurent_input>0:
                        _prev = self.extract_map(start_point-n_reccurent_input, start_point)
                        prev_batch.append(_prev)
                else:
                        
                    if self.entropy!=None:
                        _input+=self.get_entropy_map(start_point)
                    
                    _input += self.extract_map(start_point, start_point+self.extract_length)
                    
                    batch.append(_input)
                    if self.future_labels==False:
                        label = self.extract_map(start_point+self.extract_length, start_point+self.extract_length+1)[0]
                        labels.append(label)
                    else:
                        label = self.extract_map(start_point+self.extract_length, start_point+self.extract_length*2)
                        labels.append(label)
                        
                    if n_reccurent_input>0:
                        _prev = self.extract_map(start_point-n_reccurent_input, start_point)
                        prev_batch.append(_prev)
                    
                self.cache[start_point] = [_input, label, _prev]   
                    
        pack = [batch, labels, prev_batch]
        
        return pack
        
    def extract_map(self, start_frame, end_frame, map_source=None):
         extract = []
         i = start_frame
         
         f_map = self.feature_map
         if map_source!=None:
             f_map=map_source
         
         while i<end_frame:
             val = 0
             if i<0:
                 val = f_map[0]
             elif i>=len(f_map):
                 val = f_map[len(f_map)-1]
             else:
                 val = f_map[i]

             extract.append(val)
                
             i+=1
                
         return extract
         
    def get_entropy_map(self, last_start_index, map_source=None):
        size = self.entropy["size"]
        step = self.entropy["step"]
        increase_rate = self.entropy["increase_rate"]
        
        increase_step = 0
        if self.entropy.__contains__("increase_step"):
            increase_step = self.entropy["increase_step"]
        
        max_step = self.entropy["max_step"]
        #differential = self.entropy["differential"]
        
        offset = 0
        sample = []

        for k in range(size):
            small_sample = self.extract_map(last_start_index-offset-step, last_start_index-offset, map_source=map_source)
            
             
            #if differential:
            #    small_sample = np.diff(small_sample)
            
            val = self.eval_entropy(small_sample)
                
            sample.insert(0, val) 
            
            offset += step 
            if step<max_step and k%increase_step==0:
                step+=increase_rate
                
        return sample
        
    def eval_entropy(self, sample):
        ent_sample = []
        for channel in range(len(sample[0])):
            ent_sample.append(self.eval_entropy_local(sample, channel))
        return ent_sample
        
    def eval_entropy_local(self, sample, channel):
        _sum = 0
        for f in range(len(sample)):
            k = sample[f][channel]
            _sum+=abs(k-0.5)
                
        return self.quantify((_sum/len(sample))*3)
               
    def quantify(self, val, fact=256):
        return np.floor(val*fact)/fact
        
    def getImageBytes(self):
         return self.extract_length
									
    def getImageWidth(self):
        return self.extract_length
         
    def getMap(self):
         res = []
         for f in self.feature_map:
             for u in f:
                res.append(u) 
         return res
									
    def get_feature_map(self):
        return self.init_feature_map
								
    def normalize_map(self):
        maxes = [0]*self.z_size
        self.maxes = maxes 
        for f in self.feature_map:
            for channel in range(len(f)):
                if maxes[channel]<abs(f[channel]):
                    maxes[channel] = abs(f[channel])	
				
        new_map = []
        for f in self.feature_map:
            tmp = []
            for channel in range(len(f)):
                v = f[channel]
                mul = maxes[channel]
                tmp.append((v/mul)*0.5+0.5)
            new_map.append(tmp)
												
        self.init_feature_map = self.feature_map
        self.feature_map = new_map
        
    def de_normalize_map(self, feature_map):
        new_map = []
        for f in feature_map:
            tmp = []
            for channel in range(len(f)):
                y = f[channel]
                mul = self.maxes[channel]

                x = mul*((y-0.5)/0.5)

                tmp.append(x)
            new_map.append(tmp)
												
        return new_map
									
    def save_output(data, cache_file):
         with open(cache_file, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            print("data saved in file: "+cache_file)
            
    def load_output(self, cache_file):        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                print("data loaded from file: "+cache_file)
                return data
        else:
            print("file was not found: "+cache_file)
            return None
 
# standard convolution layer
def conv2d_simple(x, inputFeatures, outputFeatures, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("w",[5,5,inputFeatures, outputFeatures], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b",[outputFeatures], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, w, strides=[1,2,2,1], padding="SAME") + b
        return conv

def conv_transpose(x, inputFeatures, outputShape, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
       
        # h, w, out, in
        w = tf.get_variable("w",[5,5, outputShape[-1], inputFeatures], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b",[outputShape[-1]], initializer=tf.constant_initializer(0.0))
    
        dyn_input_shape = tf.shape(x)
        batch_size = dyn_input_shape[0]
        outputShape = tf.pack([batch_size, outputShape[1], outputShape[2], outputShape[3]])
        convt = tf.nn.conv2d_transpose(x, w, output_shape=outputShape, strides=[1,2,2,1])
        return convt
        
def dense(x, inputFeatures, outputFeatures, scope=None, with_w=False, name = ""):
    with tf.variable_scope(scope):
        return tf.add(
                      tf.matmul(x, tf.Variable(tf.random_normal([inputFeatures, outputFeatures]), name=name)),
                      tf.Variable(tf.random_normal([outputFeatures]), name=name+"bias")
                      )

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
        
class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
         self.epsilon = epsilon
         self.momentum = momentum
         self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
         self.name = name
                
    def __call__(self, x, train=True, shape=None, reuse=None):
        
        x = tf.reshape(x, shape)
        shape = x.get_shape()
        print(shape)
        
                            
        if train:
            with tf.variable_scope(self.name):
                self.beta = tf.get_variable("beta", [shape[-1]], initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],initializer=tf.random_normal_initializer(1., 0.02))
                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                    ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)
    
                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

#def bn(x, is_training=True, name="bn"):
    #h2 = tf.contrib.layers.batch_norm(x, 
    #                                  center=True, scale=True, 
    #                                  is_training=is_training,
    #                                  scope=name)
    
#    return h2