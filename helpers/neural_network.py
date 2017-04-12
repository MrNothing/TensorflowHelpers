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
import time
from IPython.display import clear_output
import pickle
from helpers.unsupervised import AutoEncoder
from helpers.operators import *

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

class ConvNet:
    def __init__(self, 
                 loader, 
                 learning_rate = 0.001, 
                 learning_decrease = 0,
                 training_iters = 200000, 
                 batch_size = 128, 
                 display_step = 10, 
                 graph_display_step = 100,
                 dropout=0.75, 
                 n_steps=-1, 
                 save_step = 10000,
                 decay_step = 100000,
                 decay_rate = 1):
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
        self.graph_display_step = graph_display_step
        
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
        
        tf.reset_default_graph() 
    
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
        
    def Run(self, 
            layers=None, 
            save_path="", 
            restore_path="", 
            input_as_label=False, 
            x=None, 
            y=None,
            state=None):
        
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
        start_t = time.process_time()
        
        # Launch the graph
        with tf.Session() as sess:
            sess.run(self.init)
            
            if len(restore_path)>0 and os.path.exists(restore_path+"/model.meta"):
                 # Restore model weights from previously saved model
                #saver = tf.train.import_meta_graph(restore_path+'/model.meta')
                load_path=saver.restore(sess, restore_path+"/model")
                print ("Model restored from file: %s" % load_path)  
                #TODO: restore acc_log and loss_log
            
            step = 1
            
            self.labels_log = []
            duration = 0
            gpu_duration = 0
            
            if self.training_iters>300:
                # Keep training until reach max iterations
                while step * self.batch_size < self.training_iters:
                    
                    batch = []
                    _t = time.process_time()
                    
                    if(self.n_steps>0):
                        batch = self.loader.getNextTimeBatch(self.batch_size, n_steps=self.n_steps)
                    else:
                        batch = self.loader.getNextBatch(self.batch_size)
                        
                    duration += time.process_time()-_t
                        
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
                    
                    _t = time.process_time()
                    
                    # Run optimization op (backprop)
                    if state!=None:
                        state, optim = sess.run([state, optimizer], feed_dict={x: inputs, y: labels, keep_prob: self.dropout, initial_state:state})
                    else:
                        sess.run(optimizer, feed_dict={x: inputs, y: labels, keep_prob: self.dropout})
                    
                    gpu_duration += time.process_time()-_t
                    
                    if step % self.save_step == 0:
                         if len(save_path)>0:
                            # Save model weights to disk
                            s_path = saver.save(sess, save_path+"/model")
                            print ("Model saved in file: %s" % s_path)
                    
                    if step % self.display_step == 0:
                        duration = int((duration/self.display_step)*100)/100
                        gpu_duration = int((gpu_duration/self.display_step)*100)/100
                        mins_passed = int((time.process_time()-start_t)/60)
                        # Calculate batch loss and accuracy
                        loss, acc = sess.run([cost, accuracy], feed_dict={x: inputs,
                                                                          y: labels,
                                                                          keep_prob: 1.})
                        print ("Iter " + str(step*self.batch_size) + ", Loss= " \
                        "{:.6f}".format(loss) + ", Accuracy= " \
                        "{:.5f}".format(acc) + ", Lrn Rate= " + str(learning_rate.eval())\
                        +" cpu: " + str(duration) + "s, gpu: " + str(gpu_duration)+"s"\
                        +" time: "+str(mins_passed)+"mins"+" cache: "+str(int(len(self.loader.cache)/(len(self.loader.converter.data))*10000)/100)+"%"
                        )
                        self.acc_log.append(acc)
                        self.loss_log.append(loss)
                        duration = 0
                        gpu_duration = 0
                        
                    if step % self.graph_display_step == 0:
                        clear_output()
                        plt.plot(self.acc_log)
                        plt.ylabel("Accuracy")
                        plt.show()
                        plt.plot(self.loss_log, color="red")
                        plt.ylabel("Loss")
                        plt.show()
                        plt.plot(self.labels_log, color="green")
                        plt.ylabel("Labels repartition")
                        plt.show()
                        
                        
                    
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
        
    def save_output(self, data, cache_file):
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            print("data saved in file: "+cache_file)
            
    def load_output(self, cache_file):        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                return data
                print("data loaded from file: "+cache_file)
        else:
            print("Cache file was not found: "+cache_file)
	
    def GenerateFeatureMap(self, 
                           input_data, 
                           restore_path, 
                           layers=None, 
                           x=None, 
                           feature_type="derivative",
                           display_step = 100,
                           ):
        
        output = []
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
                
                for i in range(len(input_data)):
                    #input data (right side)
                    extract = []
                    for k in range(self.n_input-1):
                        index = k-(self.n_input-1)
                        if index<0:
                            extract.append(0.5)
                        else:
                            extract.append(input_data[index])
                            
                    #entropy data (left side)
                    prev_inputs = input_data[0:len(output)]
                            
                    entropy = self.extract_entropy_summary(prev_inputs, len(prev_inputs)-1)
                           
                    result = sess.run(pred, feed_dict={x: [entropy+extract]}) 
                    output.append(self.max_index(result[0])/self.loader.one_hot)
                    
                    if i%display_step==0:
                        clear_output()
                        plt.plot(output)
                        plt.ylabel("Output")
                        plt.show()
        return output
        
    def GenerateLowLevel(self, 
                           maps, 
                           maps_samplerate,
                           samplerate,
                           restore_path, 
                           layers=None, 
                           x=None, 
                           feature_type="derivative",
                           display_step = 100,
                           initializer = []
                           ):
        
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
        
        output = []
        
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
                
                for i in range(len(maps[0])):
                        
                    bloc_len = int(samplerate/maps_samplerate)
                    
                    for n in range(bloc_len):
                        #input data (right side)
                        tmp_input = initializer+output
                        extract = []
                        for k in range(self.loader.fixed_size-1):
                            index = len(initializer)+len(output)+k-1-self.loader.fixed_size
                            if index<0:
                                extract.append(0.5)
                            else:
                                extract.append(tmp_input[index])
                        
                        for m in maps:
                            feature = m[i]
                            if i<len(maps[0])-2:
                                t = (i*bloc_len+n)/(len(maps[0])*bloc_len)
                                feature = self.lerp(m[i], m[i+1], t)
                                
                            extract = [feature]*(5) + extract
                             
                        if self.n_steps>0:   
                            extract = self.loader.converter.reshapeAsSequence(extract, len(extract))

                        result = sess.run(pred, feed_dict={x: [extract]}) 
                        result = self.max_index(result[0])/self.loader.one_hot
                        output.append(result)
                        self.generation_result = output
                    
                    if i%display_step==0:
                        clear_output()
                        plt.plot(output)
                        plt.ylabel("Output "+str(i)+"/"+str(len(maps[0])))
                        plt.show()
                        
                        plt.plot(output[len(output)-256:], color="red")
                        plt.show()
                        
                        for m in maps:
                            plt.plot(m, color="green")
                        plt.ylabel("map")
                        plt.show()
                        
            return output
            
    def lerp(self, a, b, t):
        return a+(b-a)*t
        
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
                 epsilon = 0,
                 label_offset=0,
                 sample_state_speed=1,
                 start_samples = None,
                 state = None,
                 input_buffer = []
                 ):
        
        self.generation_result = []

        if self.n_steps>0:
              for v in input_v:
                self.generation_result+=v
        else:
            self.generation_result = input_v[1:]
        
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
                    
                debug_input = []
                
                for i in range(iterations):
                    final_input = input_v
                    if len(self.loader.sample_shape)!=0:
                        final_input = start_samples + input_v
                        start_samples = self.extract_summary(input_buffer+self.generation_result, len(input_buffer+self.generation_result)-1)
                    elif self.loader.entropy!=None:
                        final_input = input_v
                        if start_samples!=None:
                            final_input = start_samples + input_v
                        start_samples = self.extract_entropy_summary(input_buffer+self.generation_result, len(input_buffer+self.generation_result)-1)
                    
                    debug_input = final_input
                        
                    result = None
                    
                    if state == None:
                        result = sess.run(pred, feed_dict={x: [final_input]}) 
                    else:
                        result, state = sess.run(pred, feed_dict={x: [final_input], initial_state: state})
                    r_sample_state = sample_state
                    
                    if self.loader.one_hot!=-1:
                        next_val = self.max_index(result[0])/self.loader.one_hot

                        self.generation_result.append(next_val)
                        
                        if self.loader.uLawEncode!=-1:
                            next_val = Encoder.uLawEncode(next_val, self.loader.uLawEncode, False)
                            
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
                                
                            if i<1000:
                                print(next_val)
                            
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

                        self.generation_result = self.generation_result + list(result[0])
                        
                    if i%display_step==0:
                        clear_output()
                        print("sample_state: "+str(r_sample_state))
                        if self.n_steps>0:
                            print("status: "+str(i)+"/"+str(iterations)+" "+str(input_v[len(input_v)-1][0:10])+" "+str(len(self.generation_result)))
                        else:
                            print("status: "+str(i)+"/"+str(iterations)+" "+str(input_v[0:10])+" "+str(len(self.generation_result)))
                        
                        plt.plot(self.generation_result)
                        plt.ylabel("Result")
                        plt.show()
                        plt.plot(debug_input)
                        plt.ylabel("Input")
                        plt.show()
                        
                    sample_state+=(1/len(self.loader.converter.data))*sample_state_speed
            else:
                print ("Not found: " + restore_path)
        sess.close()
        return self.generation_result
        
    def initClones(self):
         self.clones = {}
         self.clone_buffers = {}
 
         for clone_id in self.loader.sample_shape[0]:
             self.clones[clone_id] = []
             self.clone_buffers[clone_id] = []
         
    def pushInClones(self, value):
        for clone_id in self.loader.sample_shape[0]:
            if len(self.clone_buffers[clone_id])>=clone_id/self.loader.samplerate:
                self.clones[clone_id].append(np.average(self.clone_buffers[clone_id]))
                self.clone_buffers[clone_id] = [value]
            else:
                self.clone_buffers[clone_id].append(value)

    def extractFromClones(self):
        samples = []

        start = None

        for clone_id in self.loader.sample_shape[0]:
            if start==None:
                start = len(self.clones[clone_id])-1
            
            sample = self.extractFromClone(clone_id, start-len(self.loader.sample_shape), start, uLawEncode = self.loader.uLawEncode)
            
            if self.n_steps!=-1:
                sample = self.loader.converter.reshapeAsSequence(sample, self.n_steps)
            
            samples = samples + sample
        return samples
            
    def extractFromClone(self, clone_id, start_frame, end_frame, uLawEncode = -1):
         data = self.clones[clone_id]
         
         extract = []
         i = start_frame
         
         while i<end_frame:
             val = 0
             if i<0:
                 val = (0.5)
             elif i>=len(data):
                 val = (0.5)
             else:
                 val = (data[i])
                 
             if uLawEncode!=-1:
                extract.append(Encoder.uLawEncode(val, uLawEncode))
             else:
                extract.append(val)
                
             i+=1
                
         return extract
         
    def extract_entropy_summary(self, data, last_start_index):
        converter  = self.loader.converter
        
        size = self.loader.entropy["size"]
        step = self.loader.entropy["step"]
        increase_rate = self.loader.entropy["increase_rate"]
        max_step = self.loader.entropy["max_step"]
        differential = self.loader.entropy["differential"]
                    
        increase_step = 0
        if self.loader.entropy.__contains__("increase_step"):
            increase_step = self.loader.entropy["increase_step"]
        
        offset = 0
        sample = []
        
        for k in range(size):
            small_sample = self._extract_from_generated(data, last_start_index-offset-step, last_start_index-offset, uLawEncode = self.loader.uLawEncode, multiplier = 0)
            
            if differential:
                small_sample = np.diff(small_sample)
            
            val = 0
            if self.loader.sample_avg>0:
                val = Encoder.avg(small_sample, self.loader.sample_avg)
            else:
                #if self.loader.uLawEncode!=-1:
                val = Encoder.entropy(small_sample)
                #else:
                #    val = Encoder.uLawEncode(Encoder.entropy(small_sample), self.loader.uLawEncode, False)
            
            sample.insert(0, val)     
            
            offset += step 
            if step<max_step and k%increase_step==0:
                step+=increase_rate
                
        #sample = np.flip(sample, 0).tolist()
        if self.n_steps!=-1:
            sample = converter.reshapeAsSequence(sample, self.n_steps)
            
        self.debug_offset = offset
        self.debug_max_step = step
        return sample      
         
    def extract_summary(self, data, last_start_index):
        converter  = self.loader.converter
        offset = 0
        samples = []
        sample_length = len(self.loader.sample_shape)
        for sample_range in self.loader.sample_shape[0]:
            
            sample = []
            for frame_id in range(sample_length):
                if self.loader.use_avg:
                    small_sample = self._extract_from_generated(data, last_start_index-offset-sample_range, last_start_index-offset, uLawEncode = self.loader.uLawEncode)
                    #process small_sample
                    val = np.average(small_sample)
                    sample.append(val)
                else:
                    small_sample = self._extract_from_generated(data, last_start_index-offset-1, last_start_index-offset, uLawEncode = self.loader.uLawEncode)
                    #process small_sample
                    val = small_sample[0]
                    sample.append(val)
                
                offset += sample_range
                
            if self.n_steps!=-1:
                sample = converter.reshapeAsSequence(sample, self.n_steps)
            
            samples = sample + samples
        return samples
        
    def _extract_from_generated(self, data, start_frame, end_frame, uLawEncode = -1, multiplier = 1):
         extract = []
         i = start_frame
         
         while i<end_frame:
             val = 0
             if i<0:
                 val = (0.5)
             elif i>=len(data):
                 val = (0.5)
             else:
                 val = data[i]
                 
             if uLawEncode!=-1:
                extract.append(Encoder.uLawEncode(val, uLawEncode, False))
             else:
                extract.append(val)
                
             i+=1
                
         return extract
        
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
                         epsilon = 0.01, 
                         sample_state_speed=1,
                         ):
        
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
                    
                sample_state+=(1/len(self.loader.converter.data))*sample_state_speed
                
        sess.close()
        return generation_result
            
            
    def Plot(self):
        plt.plot(self.acc_log)
        plt.ylabel("Accuracy")
        plt.show()
        
        plt.plot(self.loss_log, color="red")
        plt.ylabel("Loss")
        plt.show()
        
        plt.scatter(self.labels_log, s=1, color="green")
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
                _max = data[i]
                
        if(_max_index==-1):
            _max_index = int(len(data)/2)
                
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
        