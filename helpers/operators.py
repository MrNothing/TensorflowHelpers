# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 00:26:11 2017

@author: Boris
"""

import tensorflow as tf
import numpy as np

global layer_counter
layer_counter = {}

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
 
# standard convolution layer
def conv2d(x, inputFeatures, outputFeatures, name):
    with tf.variable_scope(name):
        w = tf.get_variable("w",[5,5,inputFeatures, outputFeatures], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b",[outputFeatures], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, w, strides=[1,2,2,1], padding="SAME") + b
        return conv

def conv_transpose(x, inputFeatures, outputShape, name):
    with tf.variable_scope(name):
       
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