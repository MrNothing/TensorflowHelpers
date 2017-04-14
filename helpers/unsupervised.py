# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 21:57:10 2017
@author: Boris
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import random as rand
from helpers.sound_tools import Encoder
from helpers.operators import *
from IPython.display import clear_output

class FeatureMap:
    def __init__(self,  extract_length = 16):
        self.feature_map = None
        self.extract_length = extract_length
        
    def getNextBatch(self, batch_size, n_reccurent_input=0):
        batch = []
        prev_batch = []
        for i in range(batch_size):
            start_point = rand.randint(0, len(self.feature_map)-self.extract_length)
            batch.append(self.extract_map(start_point, start_point+self.extract_length))
            
            if n_reccurent_input>0:
                prev_batch.append(self.extract_map(start_point-n_reccurent_input, start_point))
                
        return [batch, None, prev_batch]
        
    def extract_map(self, start_frame, end_frame):
         extract = []
         i = start_frame
         
         while i<end_frame:
             val = 0
             if i<0:
                 val = (0.5)
             elif i>=len(self.feature_map):
                 val = (0.5)
             else:
                 val = self.feature_map[i]

             extract.append(val)
                
             i+=1
                
         return extract
         
    def getImageBytes(self):
         return self.extract_length
         
    def getMap(self):
         res = []
         for f in self.feature_map:
             for u in f:
                res.append(u) 
         return res

#requires: a path containing folders with names: auto_encoder_0, auto_encoder_1 ...
class StackedAutoEncoders:
    def __init__(self,
                 encoders,
                 path = "graphs/AbstractNet1",
                 display_step = 100,
                 ):
        self.encoders = encoders
        self.display_step = display_step
        self.path = path
        
        self.feature_maps = {}
        #feature_maps = encoders[0].load_output(self.path+"/feature_maps")
        #if feature_maps!=None:
        #    self.feature_maps = feature_maps
        
    def TrainAll(self):
        _id = 0
        for e in self.encoders:
            print("Layer #"+str(_id))
            if _id!=0:
                e.loader.feature_map = self.feature_maps[_id-1]
            
            #train
            e.Train(save_path=self.path+"/auto_encoder_"+str(_id), restore_path=self.path+"/auto_encoder_"+str(_id), display_step=self.display_step)
            
            #generate feature map from example
            feature_map = e.EncodeSequence(
                        e.loader.getMap(), 
                        restore_path=self.path+"/auto_encoder_"+str(_id),
                        display_step = 1000, 
                        compression_rate = 16  
                        )[0]
            
            self.feature_maps[_id] = feature_map
            
            _id+=1
            
        #self.encoders[0].save_output(self.feature_maps, self.path+"/feature_maps")
            
    def EncodeSequence(self,
                        sequence,
                        compression_rates,
                        display_step = 1000):
        _id = 0
        for e in self.encoders:
            print("Layer #"+str(_id))

            sequence = e.EncodeSequence(
                        sequence, 
                        restore_path=self.path+"/auto_encoder_"+str(_id),
                        display_step = display_step, 
                        compression_rate = compression_rates[_id]  
                        )
            _id+=1
        return sequence
        
    def DecodeSequence(self, 
                       feature_map, 
                       compression_rates,
                       display_step = 1000,
                       ):
        
        _id = len(self.encoders)-1
        for e in reversed(self.encoders):
            print("Layer #"+str(_id))

            feature_map = e.DecodeSequence(
                        feature_map, 
                        restore_path=self.path+"/auto_encoder_"+str(_id),
                        display_step = display_step, 
                        compression_rate = compression_rates[_id]  
                        )
            _id-=1
        return feature_map
        
class AutoEncoder:
    def __init__(self,
                 loader,
                 hidden = [256, 128],
                 decoder_hidden = None,
                 cells = [32],
                 learning_rate = 0.01,
                 batch_size = 128,
                 training_iters = 10000,
                 n_reccurent_input = 0,
                 time_theory = False,
                 advanced_time_theory = False,
                 rnn = False,
                 guess_one_hot = -1,
                 vae = False,
                 n_z = 20,
                 ):
        self.loader = loader
        self.hidden = hidden
        self.cells = cells
        self.n_input = loader.getImageBytes()
        self.batch_size = batch_size
        self.training_iters = training_iters
        self.learning_rate = learning_rate
        self.n_reccurent_input = n_reccurent_input
        self.rnn = rnn
        self.index_step = 256
        self.vae = vae
        self.n_z = n_z
        self.guess_one_hot = guess_one_hot
        
        if decoder_hidden==None:
            decoder_hidden = hidden
        
        self.decoder_hidden = decoder_hidden
		
        if time_theory:
            self.time_theory = time_theory
            self.advanced_time_theory = advanced_time_theory
            self.loader.advanced_time_theory = advanced_time_theory
        else:
            self.time_theory = False
            self.advanced_time_theory = False
        
        tf.reset_default_graph() 
        
    def save_output(self, data, cache_file):
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
            
    # Building the encoder
    def encoder(self, graph):
         print(graph.name+"    "+str(graph.get_shape()))
         
         # Encoder Hidden layer with sigmoid activation #1
         graph = tf.nn.sigmoid(
                               tf.add(
                                      tf.matmul(graph, tf.Variable(tf.random_normal([self.n_input, self.hidden[0]]), name="encode_matmul_start")),
                                      tf.Variable(tf.random_normal([self.hidden[0]]), name="encode_bias_start")
                                      )
                               )
         print(graph.name+"    "+str(graph.get_shape()))
         for index in range(len(self.hidden)-1):
             i = index+1
             graph = tf.nn.sigmoid(
                                   tf.add(
                                          tf.matmul(graph, tf.Variable(tf.random_normal([self.hidden[i-1], self.hidden[i]]), name="encode_matmul_"+str(i))), 
                                          tf.Variable(tf.random_normal([self.hidden[i]]), name="encode_bias_"+str(i))
                                          )
                                   )
             print(graph.name+"    "+str(graph.get_shape()))
         
         return graph
    
    
    # Building the decoder
    def decoder(self, graph):
         print(graph.name+"    "+str(graph.get_shape()))
         # Decoder Hidden layer with sigmoid activation
         first = True
         for index in range(len(self.decoder_hidden)-1):
             reccurent_bonus = 0
             if first:
                 reccurent_bonus = self.n_reccurent_input
                 first = False
             
             i = index+1
             graph = tf.nn.sigmoid(
                               tf.add(
                                      tf.matmul(graph, tf.Variable(tf.random_normal([self.decoder_hidden[len(self.decoder_hidden)-i]+reccurent_bonus, self.decoder_hidden[len(self.decoder_hidden)-i-1]]), name="decode_matmul_"+str(i))), 
                                      tf.Variable(tf.random_normal([self.decoder_hidden[len(self.decoder_hidden)-i-1]]), name="decode_bias_"+str(i))
                                    )
                                )
             print(graph.name+"    "+str(graph.get_shape()))
         graph = tf.nn.sigmoid(
                               tf.add(
                                      tf.matmul(graph, tf.Variable(tf.random_normal([self.decoder_hidden[0], self.n_input]), name="decode_matmul_end")), 
                                      tf.Variable(tf.random_normal([self.n_input]), name="decode_bias_end")
                                    )
                                )
         print(graph.name+"    "+str(graph.get_shape()))
         
         return graph
         
        
    def GRUencoder(self, graph, output, name=""):
        gru = GRUOperation(cells=self.cells, n_classes=output, name=name)
        graph = gru.getGraph(graph)
        return graph
        
    #variational  autoencoder  encoder
    def vae_encoder(self, graph):
        print("vae_encoder")
        i=0
         
        # Encoder Hidden layer with sigmoid activation #1
        graph = tf.nn.sigmoid(
                               tf.add(
                                      tf.matmul(graph, tf.Variable(tf.random_normal([self.n_input, self.hidden[0]]), name="encode_matmul_start")),
                                      tf.Variable(tf.random_normal([self.hidden[0]]), name="encode_bias_start")
                                      )
                               )
        print(graph.name+"    "+str(graph.get_shape()))
        for index in range(len(self.hidden)-1):
             i = index+1
             graph = tf.nn.sigmoid(
                                   tf.add(
                                          tf.matmul(graph, tf.Variable(tf.random_normal([self.hidden[i-1], self.hidden[i]]), name="encode_matmul_"+str(i))), 
                                          tf.Variable(tf.random_normal([self.hidden[i]]), name="encode_bias_"+str(i))
                                          )
                                   )
             print(graph.name+"    "+str(graph.get_shape()))

        w_mean = tf.nn.sigmoid(dense(graph, self.hidden[i], self.n_z, name= "w_mean"), name= "w_mean")
        w_stddev = tf.nn.sigmoid(dense(graph, self.hidden[i], self.n_z, name= "w_stddev"), name ="w_stddev")

        print(w_mean.name+"    "+str(w_mean.get_shape()))
        print(w_stddev.name+"    "+str(w_stddev.get_shape()))
        
        return w_mean, w_stddev

    # variational autoencoder decoder
    def vae_decoder(self, graph):
        print("vae_decoder")
        print(graph.name+"    "+str(graph.get_shape()))
        graph = tf.nn.sigmoid(dense(graph, self.n_z, self.hidden[len(self.hidden)-1], scope='z_matrix', name = "z_matrix"))
        print(graph.name+"    "+str(graph.get_shape()))
        
        for index in range(len(self.hidden)-1):
        
             i = index+1
             graph = tf.nn.sigmoid(
                               tf.add(
                                      tf.matmul(graph, tf.Variable(tf.random_normal([self.hidden[len(self.hidden)-i], self.hidden[len(self.hidden)-i-1]]), name="decode_matmul_"+str(i))), 
                                      tf.Variable(tf.random_normal([self.hidden[len(self.hidden)-i-1]]), name="decode_bias_"+str(i))
                                    )
                                )
             print(graph.name+"    "+str(graph.get_shape()))
        graph = tf.nn.sigmoid(
                               tf.add(
                                      tf.matmul(graph, tf.Variable(tf.random_normal([self.hidden[0], self.n_input]), name="decode_matmul_end")), 
                                      tf.Variable(tf.random_normal([self.n_input]), name="decode_bias_end")
                                    )
                                )
        print(graph.name+"    "+str(graph.get_shape()))

        return graph
        
    def Run(self, 
              save_path="", 
              restore_path="", 
              display_step = 10,
              ):
        return self.Train(restore_path, restore_path, display_step)
    
    #Trains the model
    def Train(self, 
              save_path="", 
              restore_path="", 
              display_step = 10,
              ):
        
        tf.reset_default_graph() 
        
        self.loss_log = []
        self.acc_log = []
        
        x = None
        encoder_op = None
        decoder_op = None
        
        #vae
        z_mean = None
        z_stddev = None
        generation_loss = None
        latent_loss = None
        
        if self.rnn:
            x = tf.placeholder("float", [None, self.n_input], name="x_input")
            encoder_op = self.encoder(x)
            
            if self.n_reccurent_input==0:
                x2 = tf.placeholder("float", [None, self.hidden[len(self.hidden)-1], 1], name="x_input")
            elif self.time_theory:
                self.n_reccurent_input = 2
                x2 = tf.placeholder("float", [None, self.hidden[len(self.hidden)-1]+self.n_reccurent_input, 1], name="x2_input")
            else:
                x2 = tf.placeholder("float", [None, self.hidden[len(self.hidden)-1]+self.n_reccurent_input, 1], name="x2_input")
                
            if self.guess_one_hot>0:
                decoder_op = self.GRUencoder(x2, self.guess_one_hot, "x2_input")
            else:
                decoder_op = self.GRUencoder(x2, self.n_input, "x2_input")
                
        elif self.vae:
             # tf Graph input (only pictures)
            x = tf.placeholder("float", [None, self.n_input], name="x_input")
            
            # Construct model
            z_mean, z_stddev = self.vae_encoder(x)
            samples = tf.random_normal([self.batch_size,self.n_z],0,1,dtype=tf.float32)
            encoder_op = z_mean + (z_stddev * samples)
    
            decoder_op = self.vae_decoder(encoder_op)

        else:
            # tf Graph input (only pictures)
            x = tf.placeholder("float", [None, self.n_input], name="x_input")
            
            # Construct model
            encoder_op = self.encoder(x)
            
            if self.n_reccurent_input==0:
                x2 = tf.placeholder("float", [None, self.hidden[len(self.hidden)-1]], name="x2_input")
            elif self.time_theory:
                self.n_reccurent_input = 2
                x2 = tf.placeholder("float", [None, self.hidden[len(self.hidden)-1]+self.n_reccurent_input], name="x2_input")
            else:
                x2 = tf.placeholder("float", [None, self.hidden[len(self.hidden)-1]+self.n_reccurent_input], name="x2_input")
            
            decoder_op = self.decoder(x2)
        
        # Prediction
        y_pred = decoder_op
        # Targets (Labels) are the input data.
        y_true = None
        
        if self.guess_one_hot>0:
            y_true = tf.placeholder("float", [None, self.guess_one_hot], name="y_true")
        else:
            y_true = x
        
        # Define loss and optimizer, minimize the squared error
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(cost)
        
        if self.vae:
            generation_loss = -tf.reduce_sum(y_true * tf.log(1e-8 + y_pred) + (1-y_true) * tf.log(1e-8 + 1 - y_pred),1)
            latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
            cost = tf.reduce_mean(generation_loss + latent_loss)
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
            
        # Evaluate model
        correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        # Initializing the variables
        init = tf.global_variables_initializer()
        
        step = 1
        
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            
            if len(restore_path)>0 and os.path.exists(restore_path+"/model.meta"):
                 # Restore model weights from previously saved model
                #saver = tf.train.import_meta_graph(restore_path+'/model.meta')
                load_path=saver.restore(sess, restore_path+"/model")
                print ("Model restored from file: %s" % load_path)  
                #TODO: restore acc_log and loss_log
            
            # Training cycle
            while step * self.batch_size < self.training_iters:
                 if self.vae:
                     
                     batch = self.loader.getNextBatch(self.batch_size, n_reccurent_input = 0)[0]
                     
                     _, gen_loss, lat_loss = sess.run((optimizer, generation_loss, latent_loss), feed_dict={x: batch})
                 
                     # Display logs per epoch step
                     if step % display_step == 0:
                        print("step:", '%04d' % (step),
                              "vae cost="+str(np.mean(gen_loss))+" "+str(np.mean(lat_loss))+" cache: "+str(int(len(self.loader.cache)/(len(self.loader.converter.data))*10000)/100)+"%")
                     
                 else:
                     
                    
                     batch=None
                     labels=None
                     past_batch=[]
                     if self.n_reccurent_input<=0:
                         batch = self.loader.getNextBatch(self.batch_size, label_as_first=True)
                         labels = batch[1]
                         batch = batch[0]
                         
                     else:
                         if self.time_theory:
                            batch = self.loader.getNextBatch(self.batch_size, time_theory = True)
                         else:
                            batch = self.loader.getNextBatch(self.batch_size, n_reccurent_input = self.n_reccurent_input)
                         
                         past_batch = batch[2]
                         labels = batch[1]
                         batch = batch[0]
                     
                     # Run optimization op (backprop) and cost op (to get loss value)
                     tmp_output = sess.run(encoder_op, feed_dict={x: batch})
                         
                     fused_input = []
                     
                     if len(past_batch)>0:
                         for i in range(len(past_batch)):
                             fused_input.append(tmp_output[i].tolist()+past_batch[i])
                     else:
                         fused_input = tmp_output
                         
                     if self.rnn:
                         fused_input = np.reshape(fused_input, (self.batch_size, self.hidden[len(self.hidden)-1]+self.n_reccurent_input, 1))
                     
                     c= None
                     acc = None
                     if self.guess_one_hot>0:
                        _, c, acc = sess.run([optimizer, cost, accuracy], feed_dict={x: batch, x2: fused_input, y_true:labels})
                     else:
                        _, c, acc = sess.run([optimizer, cost, accuracy], feed_dict={x: batch, x2: fused_input})
                     # Display logs per epoch step
                     if step % display_step == 0:
                         self.loss_log.append(c)
                         self.acc_log.append(acc)
                         
                         clear_output()
                         plt.plot(self.acc_log)
                         plt.ylabel("Accuracy")
                         plt.show()
                         plt.plot(self.loss_log, color="red")
                         plt.ylabel("Loss")
                         plt.show()
                        
                         
                         print("step:", '%04d' % (step),
                              "cost=", "{:.9f}".format(c)+" acc: "+"{:.5f}".format(acc)+" cache: "+str(int(len(self.loader.cache)/(len(self.loader.converter.data))*10000)/100)+"%")
                    
                 step+=1
        
            print("Optimization Finished!")
            
            if len(save_path)>0:
                    # Save model weights to disk
                    s_path = saver.save(sess, save_path+"/model")
                    print ("Model saved in file: %s" % s_path)
            
            sess.close()
    
	#Generates a feature/sequence index from a raw sequence
    def IndexSequence(self, 
                        sequence,
                        restore_path="",
                        display_step = 10,
                        index_step = 256
                        ):
        
        self.index_step = index_step
        
        tf.reset_default_graph() 
        # tf Graph input (only pictures)
        X = tf.placeholder("float", [None, self.n_input])
        
        # Construct model
        encoder_op = self.encoder(X)
        #decoder_op = self.decoder(encoder_op)
        
        # Initializing the variables
        init = tf.global_variables_initializer()
        
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        
        self.result = []
        self.index = {}
        
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            
            if len(restore_path)>0 and os.path.exists(restore_path+"/model.meta"):
                 # Restore model weights from previously saved model
                #saver = tf.train.import_meta_graph(restore_path+'/model.meta')
                load_path=saver.restore(sess, restore_path+"/model")
                print ("Model restored from file: %s" % load_path)  
                #TODO: restore acc_log and loss_log
            
            for i in range(len(sequence)-self.n_input):
                sample = sequence[i:i+self.n_input]
                code = sess.run(encoder_op, feed_dict={X: [sample]})[0]
                
                code_id = self.getCodeId(code)
                
                if self.index.__contains__(code_id)==False:
                    self.index[code_id]=[[sample], [code]]
                else:
                    self.index[code_id][0].append(sample)
                    self.index[code_id][1].append(code)
                
                if i%display_step==0:
                    print("status: "+str(i)+"/"+str(len(sequence)-self.n_input)+" categories: "+str(len(self.index)))
            
            sess.close()
            
        #we have the encoded sequence!
        self.save_output(self.index, restore_path+"/features_index")
        
        return self.index
        
    #Generates a decoded sequence from an index buffer and a feature map using n-dimentional trees
    def DecodeSequenceWithIndex(self, 
                        feature_maps,
                        restore_path="",
                        display_step = 10,
                        ):
        
        self.index = self.load_output(restore_path+"/features_index")   
        
        self.result = []
        i=1
        for code in feature_maps:
            self.result += self.try_get_sample(code)
            
            if i%display_step==0:
                print("status: "+str(i)+"/"+str(len(feature_maps)))
            
            i+=1
            
        return self.result
            
    def try_get_sample(self, code):
        code_id = self.getCodeId(code)
        if self.index.__contains__(code_id):
            return self.getNearestSubSample(code, self.index[code_id])
        else:
            index = self.getNearestSample(code)
            return self.getNearestSubSample(code, self.index[index])
            
    def getNearestSample(self, code):
        best_dist = 9999999999
        dist = -1
        chosen_index = None
        for i in self.index:
            dist = self.getDistance(code, self.index[i][1][0])
            if dist<best_dist:
                chosen_index = i
                best_dist = dist
        return chosen_index
        
    def getNearestSubSample(self, code, subset):
        best_dist = 9999999999
        chosen_sample = None
        length = len(subset[1])
        for i in range(length):
            dist = self.getDistance(code, subset[1][i])
            if dist<best_dist:
                chosen_sample = subset[0][i]
                best_dist = dist
                
        if chosen_sample==None:
            print("Error!")
        
        return chosen_sample
                
    def getDistance(self, tensorA, tensorB):
        dist = 0
        for i in range(len(tensorA)):
            dist+=abs(tensorA[i]-tensorB[i])
        return dist
        
    def getCodeId(self, code):
        index_step = self.index_step
        code_id = ""
        for i in code:
            code_id += str(np.floor(i*index_step)/index_step)+"_"
        return code_id
            
    #Simple Denoising Method
    def EncodeAndDecode(self, 
                        _input,
                        restore_path="",
              ):
        
        encoder_op = None
        decoder_op = None
        
        # tf Graph input (only pictures)
        x = tf.placeholder("float", [None, self.n_input], name="x_input")
        
        if self.vae:
             # tf Graph input (only pictures)
            x = tf.placeholder("float", [None, self.n_input], name="x_input")
            
            # Construct model
            z_mean, z_stddev = self.vae_encoder(x)
            samples = tf.random_normal([self.batch_size,self.n_z],0,1,dtype=tf.float32)
            encoder_op = z_mean + (z_stddev * samples)
    
            decoder_op = self.vae_decoder(encoder_op)
        else:
            
            # Construct model
            encoder_op = self.encoder(x)
            decoder_op = self.decoder(encoder_op)

        # Initializing the variables
        init = tf.global_variables_initializer()
        
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            
            if len(restore_path)>0 and os.path.exists(restore_path+"/model.meta"):
                 # Restore model weights from previously saved model
                #saver = tf.train.import_meta_graph(restore_path+'/model.meta')
                load_path=saver.restore(sess, restore_path+"/model")
                print ("Model restored from file: %s" % load_path)  
                #TODO: restore acc_log and loss_log
            
            self.result = sess.run(decoder_op, feed_dict={x: _input})
            
            sess.close()
            
        return self.result
        
    #Simple Encode Method
    def Encode(self, 
                        _input,
                        restore_path="",
                        n_input = None,
                        ):
        
        # tf Graph input (only pictures)
        X = tf.placeholder("float", [None, self.n_input])
        if n_input!=None:
            X = tf.placeholder("float", [None, n_input])
        
        # Construct model
        encoder_op = self.encoder(X)
        #decoder_op = self.decoder(encoder_op)

        # Initializing the variables
        init = tf.global_variables_initializer()
        
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            
            if len(restore_path)>0 and os.path.exists(restore_path+"/model.meta"):
                 # Restore model weights from previously saved model
                #saver = tf.train.import_meta_graph(restore_path+'/model.meta')
                load_path=saver.restore(sess, restore_path+"/model")
                print ("Model restored from file: %s" % load_path)  
                #TODO: restore acc_log and loss_log
            
            self.result = sess.run(encoder_op, feed_dict={X: _input})
            
            sess.close()
            
        return self.result
        
    #Simple Decode Method
    def Decode(self, 
                        _input,
                        restore_path=""):
        
        # tf Graph input (only pictures)
        X = tf.placeholder("float", [None, self.hidden[len(self.hidden)-1]], name="x2_input")
        
        # Construct model
        #encoder_op = self.encoder(X)
        decoder_op = self.decoder(X)

        # Initializing the variables
        init = tf.global_variables_initializer()
        
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            
            if len(restore_path)>0 and os.path.exists(restore_path+"/model.meta"):
                 # Restore model weights from previously saved model
                #saver = tf.train.import_meta_graph(restore_path+'/model.meta')
                load_path=saver.restore(sess, restore_path+"/model")
                print ("Model restored from file: %s" % load_path)  
                #TODO: restore acc_log and loss_log
            
            self.result = sess.run(decoder_op, feed_dict={X: _input})
            
            sess.close()
            
        return self.result
        
    """
        returns N channels of features from a sequence based on the last layer of features
    """
    def EncodeSequence(self, 
                        sequence,
                        restore_path="",
                        display_step = 10,
                        compression_rate = 1
                        ):
        
        tf.reset_default_graph() 
        # tf Graph input (only pictures)
        X = tf.placeholder("float", [None, self.n_input])
        
        encoder_op = None
        
        if self.vae:
        
            # Construct model
            z_mean, z_stddev = self.vae_encoder(X)
            samples = tf.random_normal([self.batch_size,self.n_z],0,1,dtype=tf.float32)
            encoder_op = z_mean + (z_stddev * samples)
            
            print(encoder_op.name+"    "+str(encoder_op.get_shape()))
        else:
            # Construct model
            encoder_op = self.encoder(X)
            #decoder_op = self.decoder(encoder_op)
            
            

        # Initializing the variables
        init = tf.global_variables_initializer()
        
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        
        self.result = []
        
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            
            if len(restore_path)>0 and os.path.exists(restore_path+"/model.meta"):
                 # Restore model weights from previously saved model
                #saver = tf.train.import_meta_graph(restore_path+'/model.meta')
                load_path=saver.restore(sess, restore_path+"/model")
                print ("Model restored from file: %s" % load_path)  
                #TODO: restore acc_log and loss_log
            
            for i in range(len(sequence)-self.n_input):
                sample = sequence[i:i+self.n_input]
                if i%compression_rate == 0:
                    self.result.append(sess.run(encoder_op, feed_dict={X: [sample]})[0])
                        
                if i%display_step==0:
                    print("status: "+str(i)+"/"+str(len(sequence)-self.n_input))
            
            sess.close()
            
        return self.result
        
    """
        returns a low level decoded sequence from a feature map
    """
    def DecodeSequence(self, 
                        feature_maps,
                        restore_path="",
                        display_step = 10,
                        compression_rate = 1,
                        gaussian_mixture = False,
                        receptive_field = 0,
                        offset = 0,
                        ):
        
        tf.reset_default_graph() 
        
        X = None
        decoder_op = None
        
        if self.vae:
            X = tf.placeholder("float", [None, self.n_z], name="x2_input")
            decoder_op = self.vae_decoder(X)
        elif self.rnn:
            if self.n_reccurent_input==0:
                X = tf.placeholder("float", [None, self.hidden[len(self.hidden)-1], 1], name="x2_input")
            elif self.time_theory:
                self.n_reccurent_input = 2
                X = tf.placeholder("float", [None, self.hidden[len(self.hidden)-1]+self.n_reccurent_input, 1], name="x2_input")
            else:
                X = tf.placeholder("float", [None, self.hidden[len(self.hidden)-1]+self.n_reccurent_input, 1], name="x2_input")
            
            if self.guess_one_hot>0:
                decoder_op = self.GRUencoder(X, self.guess_one_hot, "x2_input")
            else:
                decoder_op = self.GRUencoder(X, self.n_input, "x2_input")
        else:
            
            if self.n_reccurent_input==0:
                X = tf.placeholder("float", [None, self.hidden[len(self.hidden)-1]], name="x2_input")
            elif self.time_theory:
                self.n_reccurent_input = 2
                X = tf.placeholder("float", [None, self.hidden[len(self.hidden)-1]+self.n_reccurent_input], name="x2_input")
            else:
                X = tf.placeholder("float", [None, self.hidden[len(self.hidden)-1]+self.n_reccurent_input], name="x2_input")
        
            # Construct model
            decoder_op = self.decoder(X)
        
        # Initializing the variables
        init = tf.global_variables_initializer()
        
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        
        self.result = []
        
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            
            if len(restore_path)>0 and os.path.exists(restore_path+"/model.meta"):
                 # Restore model weights from previously saved model
                #saver = tf.train.import_meta_graph(restore_path+'/model.meta')
                load_path=saver.restore(sess, restore_path+"/model")
                print ("Model restored from file: %s" % load_path)  
                #TODO: restore acc_log and loss_log
            
                
            for index in range(len(feature_maps)-1):
                
                if receptive_field>0:
                    i = index+1
                
                    features = []
                    for f in range(len(feature_maps[i])):
                        features.append(feature_maps[i][f])
                        
                    lastfeatures = []
                    for f in range(len(feature_maps[i-1])):
                        lastfeatures.append(feature_maps[i-1][f])
                         
                    
                    res = sess.run(decoder_op, feed_dict={X: [lastfeatures, features]})
                    
                    index_to_cut = 0
                    if len(self.result)!=0:
                        delta = self.result[len(self.result)-2]-self.result[len(self.result)-1]
                        
                        index_to_cut = self.getNearestIndexFromStart(res[1], self.result[len(self.result)-1], delta, receptive_field)
                       
                    cut_sample = res[1][index_to_cut:]
                      
                    for c in range(compression_rate):
                       self.result.append(cut_sample[c])
                elif gaussian_mixture:
                    i = index+1
                    
                    for c in range(compression_rate):
                        features = []
                        t = c/compression_rate
                        
                        for f in range(len(feature_maps[i])):
                            features.append(self.lerp(feature_maps[i-1][f], feature_maps[i][f], t))
                            
                        res = sess.run(decoder_op, feed_dict={X: [features]})
                        self.result.append(res[0][0])
                
                else:
                    i = index
                    features = []
                    for f in range(len(feature_maps[i])):
                        features.append(feature_maps[i][f])
                    
                    if self.guess_one_hot>0:
                        if i%display_step==0:
                            clear_output()
                            plt.plot(self.result)
                            plt.ylabel("Result")
                            plt.show()
                            plt.plot(feature_maps)
                            plt.ylabel("feature_maps")
                            plt.show()
                            
                        for c in range(compression_rate):
                            _features = []
                            if self.n_reccurent_input>0:
                                if self.time_theory:
                                    t = 0
                                    if self.advanced_time_theory:
                                        small_sample = self.extract_result(len(self.result)-1-self.loader.fixed_size, -0.5)
                                        small_sample = Encoder.normalize(None, small_sample)
                                        freq = Encoder.get_frequency(small_sample, 0, 0.1, differential=True)
                                        t = (len(self.result)/self.fixed_size)*np.pi*2*freq
                                    else:
                                        t = (len(self.result)/self.fixed_size)*np.pi*2
                                    _features = [np.sin(t)/2+0.5, np.cos(t)/2+0.5]
                                else:
                                    _features = self.extract_result(len(self.result)-1-self.n_reccurent_input, len(self.result)-1)
                            
                            _features = features+_features
                                    
                            if self.rnn:
                                _features = np.reshape(_features, (len(_features), 1))
                                    
                            res = sess.run(decoder_op, feed_dict={X: [_features]})
                    
                            val = self.max_index(res[0])/self.guess_one_hot
                            self.result.append(val)
                    else:
                        _features = []
                        if self.n_reccurent_input>0:
                            if self.time_theory:
                                t = 0
                                if self.advanced_time_theory:
                                    small_sample = self.extract_result(len(self.result)-1-self.loader.fixed_size, -0.5)
                                    small_sample = Encoder.normalize(None, small_sample)
                                    freq = Encoder.get_frequency(small_sample, 0, 0.1, differential=True)
                                    t = (len(self.result)/self.fixed_size)*np.pi*2*freq
                                else:
                                    t = (len(self.result)/self.fixed_size)*np.pi*2
                                _features = [np.sin(t)/2+0.5, np.cos(t)/2+0.5]
                            else:
                                _features = self.extract_result(len(self.result)-1-self.n_reccurent_input, len(self.result)-1)
                         
                        _features = features+_features
                                
                        if self.rnn:
                            _features = np.reshape(_features, (len(_features), 1))
                                
                        res = sess.run(decoder_op, feed_dict={X: [_features]})
                    
                        for c in range(compression_rate):
                            val = res[0][c]
    
                            if (val<0.05 or val>0.95) and len(self.result)>0:
                                val = self.result[len(self.result)-1]
    
                            self.result.append(val)
                        
                if i%display_step==0:
                    print("status: "+str(i)+"/"+str(len(feature_maps)))
               
                    
            
            sess.close()
            
        return self.result
    
    def extract_result(self, start_frame, end_frame, offset=0):
         extract = []
         i = start_frame
         
         while i<end_frame:
             val = 0
             if i<0:
                 val = (0.5+offset)
             elif i>=len(self.result):
                 val = (0.5+offset)
             else:
                 val = self.result[i]+offset

             extract.append(val)
                
             i+=1
                
         return extract
                
        
        
    def lerp(self, a, b, t):
        return a+(b-a)*t

    def rangeFactor(self, t, point, _range):
        ratio = np.abs (point - t) / _range;
        if ratio < 1:
            return 1 - ratio;
        else:
            return 0;
            
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
            
    def getNearestIndexFromStart(self, array, value = 0, t_delta = 0, length = 10):
        dist = 99999999999
        index = -1
        for u in range(length-1):
            i = u+1
            delta = array[i]-array[i-1]
            delta_dist = abs(delta-t_delta) 
            c_dist = abs(array[i]-value)
            if c_dist+delta_dist<dist:
                dist = c_dist+delta_dist
                index = i
        return index