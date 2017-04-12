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
        
def dense(x, inputFeatures, outputFeatures, scope=None, with_w=False):
    return tf.add(
                      tf.matmul(x, tf.Variable(tf.random_normal([inputFeatures, outputFeatures]))),
                      tf.Variable(tf.random_normal([outputFeatures]))
                      )
        