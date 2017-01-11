# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 20:53:38 2017

@author: Boris Musarais
"""
import tensorflow as tf
import os

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

class NNOperation:
    def __init__(self, _type, shape=[], bias=[0]):
        self._type = _type
        self.shape = shape
        self.bias = bias

    def getGraph(self, graph):
        obj = None
        if self._type=="relu":
            obj = tf.nn.relu(graph)
        elif self._type=="reshape":
            obj = tf.reshape(graph, shape=self.shape)
        elif self._type=="dropout":
            obj = tf.nn.dropout(graph, self.shape)
        elif self._type=="conv2d":
            obj = conv2d(graph, tf.Variable(tf.random_normal(self.shape)), tf.Variable(tf.random_normal([self.shape[3]])))
        elif self._type=="maxpool2d":
            obj = maxpool2d(graph, self.shape)
        elif self._type=="wx+b":
            obj = tf.add(tf.matmul(graph, tf.Variable(tf.random_normal(self.shape))), tf.Variable(tf.random_normal(self.bias)))
        
        print(self._type+": "+str(obj.get_shape()))
            
        return obj
        
class ConvNet:
    def __init__(self, loader, learning_rate = 0.001, training_iters = 200000, batch_size = 128, display_step = 10):
        self.loader = loader
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.batch_size = batch_size
        self.display_step = display_step
        
        # Network Parameters
        self.n_input = loader.getImageBytes() # MNIST data input (img shape: 28*28)
        self.n_classes = loader.LABELS_COUNT # MNIST total classes (0-9 digits)
        self.img_width = int(loader.getImageWidth())
        self.final_img_width = int((loader.getImageWidth()/2)/2)
        self.dropout = 0.75 # Dropout, probability to keep units
        self.report = []
    
    def getDefaultModel(self):
        layers = []
        layers.append(NNOperation("reshape", [-1, self.img_width, self.img_width, 1]))
        layers.append(NNOperation("conv2d", [5, 5, 1, 32]))
        layers.append(NNOperation("maxpool2d", 2))
        layers.append(NNOperation("conv2d", [5, 5, 32, 64]))
        layers.append(NNOperation("maxpool2d", 2))
        layers.append(NNOperation("reshape", [-1, self.final_img_width*self.final_img_width*64]))
        layers.append(NNOperation("wx+b", [self.final_img_width*self.final_img_width*64, 1024], [1024]))
        layers.append(NNOperation("relu"))
        layers.append(NNOperation("dropout", self.dropout))
        layers.append(NNOperation("wx+b", [1024, self.n_classes], [self.n_classes]))
        return layers
        
    def Run(self, layers=None, save_path="", restore_path=""):
        
        # tf Graph input
        x = tf.placeholder(tf.float32, [None, self.n_input])
        y = tf.placeholder(tf.float32, [None, self.n_classes])
        keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        
        if layers == None:
            layers = self.getDefaultModel()
        
        graph = layers[0].getGraph(x)
        
        l=1
        while l<len(layers):
            graph = layers[l].getGraph(graph)
            l+=1
        
        # Construct model
        pred = graph
        
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
                saver.restore(sess, tf.train.latest_checkpoint('./'))
                load_path = saver.restore(sess, restore_path)
                print ("Model restored from file: %s" % load_path)
            
            step = 1
            # Keep training until reach max iterations
            while step * self.batch_size < self.training_iters:
                batch = self.loader.getNextBatch(self.batch_size)
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch[0], y: batch[1], keep_prob: self.dropout})
                if step % self.display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([cost, accuracy], feed_dict={x: batch[0],
                                                                      y: batch[1],
                                                                      keep_prob: 1.})
                    print ("Iter " + str(step*self.batch_size) + ", Minibatch Loss= " \
                    "{:.6f}".format(loss) + ", Training Accuracy= " \
                    "{:.5f}".format(acc))
                    self.report.append(acc)
                step += 1
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch[0], y: batch[1],
                                                                      keep_prob: 1.})
            print ("Iter " + str(step*self.batch_size) + ", Minibatch Loss= " \
                    "{:.6f}".format(loss) + ", Training Accuracy= " \
                    "{:.5f}".format(acc))
            print ("Optimization Finished!")
            
            
            batch = self.loader.getTestBatch(256)
            
            if len(save_path)>0:
                # Save model weights to disk
                s_path = saver.save(sess, save_path)
                print ("Model saved in file: %s" % s_path)
                
            # Calculate accuracy for 256 mnist test images
            print ("Testing Accuracy:"+\
                sess.run(accuracy, feed_dict={x: batch[0],
                                              y: batch[1],
                                              keep_prob: 1.}) )

        