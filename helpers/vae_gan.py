import tensorflow as tf
from helpers.operators import *
from helpers.sound_tools import Encoder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
from IPython.display import clear_output, display

class AdversarialVariationalAutoEncoder:
	def __init__(self, 
					loader, 
					learning_rate = 1e-3, 
					batch_size = 32, 
					z_size = 10,
					X_dim = None,
					hidden = [128],
					cells = [32],
					injected_z = None,
					rnn = False,
					use_conv = False,
					conv_features = 16,
					x = None,
					unsupervised = False,
					):
		
		self.unsupervised = unsupervised
		self.cells = cells
		self.n_reccurent_input = injected_z
		self.loader = loader
		self.batch_size = batch_size
		self.z_dim = z_size
		self.x_shape = x
		
		if X_dim == None:
			self.X_dim = loader.getImageBytes()
		else:
			self.X_dim = X_dim
		
		self.Y_dim = self.loader.fixed_size
		
		if injected_z!=None:
			self.injected_z_dim = injected_z
		else:
			self.injected_z_dim = 0
			
		self.h_dim = hidden_size
		self.lr = learning_rate
		self.conv_features = conv_features
		self.first_init = True

	def init(self):
		
		self.final_conv_dims = 1
		
		if self.x_shape!=None and self.first_init==True:
			self.X_dim *= self.x_shape[2]
			self.first_init = False
		
		if self.x_shape==None:
			self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim], name="x_input")
		else:
			self.X = tf.placeholder(tf.float32, shape=self.x_shape, name="x_input")
		
		self.y = tf.placeholder(tf.float32, shape=[None, self.Y_dim], name="y_original")
		
		self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="z_latent")
		
		"""conv vars"""
		self.C_W1 = tf.Variable(self.xavier_init([self.final_conv_dims*1*self.conv_features*2, self.h_dim]))
		self.DC_W1 = tf.Variable(self.xavier_init([self.final_conv_dims*1*self.conv_features*2, self.h_dim]))
		
		
		""" Encoder Q(z|X) """
		self.Q_W1 = tf.Variable(self.xavier_init([self.X_dim, 1024]), name="Q_W1")
		self.Q_b1 = tf.Variable(tf.zeros(shape=[1024]), name="Q_b1")

		self.Q_W2 = tf.Variable(self.xavier_init([1024, self.z_dim]), name="Q_W2")
		self.Q_b2 = tf.Variable(tf.zeros(shape=[self.z_dim]), name="Q_b2")

		self.theta_Q = [self.Q_W1, self.Q_W2, self.Q_b1, self.Q_b2]

		""" generator P(X|z) """
		self.P_W1 = tf.Variable(self.xavier_init([self.z_dim, self.h_dim]), name="P_W1")
		self.P_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]), name="P_b1")

		self.P_W2 = tf.Variable(self.xavier_init([self.h_dim, self.Y_dim]), name="P_W2")
		self.P_b2 = tf.Variable(tf.zeros(shape=[self.Y_dim]), name="P_b2")
		if self.injected_z_dim>0:
			self.injected_z = tf.placeholder(tf.float32, shape=[None, self.injected_z_dim], name="z_latent")
			self.P_iW1 = tf.Variable(self.xavier_init([self.injected_z_dim, self.h_dim]), name="P_iW1")
			self.P_ib1 = tf.Variable(self.xavier_init([self.h_dim]), name="P_iW1")
			self.theta_P = [self.P_iW1, self.P_ib1, self.P_W1, self.P_W2, self.P_b1, self.P_b2]
		else:
			self.theta_P = [self.P_W1, self.P_W2, self.P_b1, self.P_b2]

		""" Discriminator D(z) """
		self.D_W1 = tf.Variable(self.xavier_init([self.Y_dim, self.h_dim]), name="D_W1")
		self.D_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]), name="D_b1")

		self.D_W2 = tf.Variable(self.xavier_init([self.h_dim, 1]), name="D_W2")
		self.D_b2 = tf.Variable(tf.zeros(shape=[1]), name="D_b2")

		self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]
	
	#Q = Encoder
	def Encoder(self, X, reuse=False):
		print("Encoder: reuse="+str(reuse))
		h=X
		if self.x_shape==None:
			h = X
		else:
			h = tf.reshape(X, [-1, self.loader.extract_length*3])
		#h = tf.reshape(h, [-1, self.X_dim, 1, 1], name="encoder_reshape0")
		#print(X.get_shape())
		#h = conv2d_simple(h, 1, self.conv_features, name="encoder_conv_1", reuse=reuse) #14*14*16
		#print(h.name+" "+str(h.get_shape()))
		#h = conv2d_simple(h, self.conv_features, self.conv_features*2, name="encoder_conv_2", reuse=reuse) #7*7*32
		#print(h.name+" "+str(h.get_shape()))
		#h = tf.reshape(h, [-1, self.final_conv_dims*7*self.conv_features*2], name="encoder_reshape1")
		#print(h.name+" "+str(h.get_shape()))
		h = tf.nn.relu(tf.matmul(h, self.Q_W1) + self.Q_b1)
		z = tf.matmul(h, self.Q_W2) + self.Q_b2
		
		return z
		
	def GRUEncoder(self, X, reuse=False):
		X = tf.reshape(X, (-1, self.X_dim, 1))
		gru = None
		if reuse:
			gru = GRUOperation(cells=self.cells, n_classes=self.z_dim, name="GRU_encoder2")
		else:
			gru = GRUOperation(cells=self.cells, n_classes=self.z_dim, name="GRU_encoder")
			
			graph = gru.getGraph(X)
		return graph
		
	#P = Discriminator
	def Generator(self, z, reuse=False, no_reshape=False):
		print("Generator:")
		
		h = tf.nn.relu(tf.matmul(z, self.P_W1) +  self.P_b1)
		h = tf.matmul(h, self.P_W2) + self.P_b2

		if self.x_shape!=None and no_reshape==False:
			h = tf.reshape(h, [-1, self.loader.extract_length, 3])
			#h = tf.nn.sigmoid(h)
		else:
			h = tf.nn.sigmoid(h)
		
		#print(h.name+" "+str(h.get_shape()))
		#h = tf.reshape(h, [-1, 7, 7, self.conv_features*2], name="generator_reshape1")
		#print(h.name+" "+str(h.get_shape()))
		#h = conv_transpose(h, self.conv_features*2, [self.batch_size, 14, 14, self.conv_features], name='generator_convt', reuse=reuse) #7x7x32
		#print(h.name+" "+str(h.get_shape()))
		#h = conv_transpose(h, self.conv_features, [self.batch_size, 28, 28, 1], name='generator_convt_2', reuse=reuse) #14x14x32
		#print(h.name+" "+str(h.get_shape()))
		#h = tf.reshape(h, [-1, self.X_dim], name="generator_reshape2")
		#print(h.name+" "+str(h.get_shape()))
		
		print(h.name+" "+str(h.get_shape()))
		
		return h, h
		
	def GRUGenerator(self, z, reuse=False):
		z = tf.reshape(z, (-1, self.z_dim, 1))
		gru = GRUOperation(cells=self.cells, n_classes=self.Y_dim, name="GRU_generator")
		graph = gru.getGraph(z)
		return graph
		
	def GeneratorInjected(self, z, Inj_z, no_reshape=False):
		print("Generator: Injection="+str(self.injected_z_dim))
		h = tf.nn.relu(tf.matmul(z, self.P_W1) + self.P_b1)
		print(h.name+" "+str(h.get_shape()))
		h2 = tf.nn.relu(tf.matmul(Inj_z, self.P_iW1) + self.P_ib1)
		h = tf.matmul(h+h2, self.P_W2) + self.P_b2
		
		if self.x_shape!=None and no_reshape==False:
			h = tf.reshape(h, [-1, self.loader.extract_length, 3])
		

		prob = tf.nn.sigmoid(h)
		#print(prob.name+" "+str(prob.get_shape()))
		return prob, prob
  
	def GRUGeneratorInjected(self, z, Inj_z, reuse=False):
		z = tf.reshape(z+Inj_z, (-1, self.z_dim, 1))
		gru = GRUOperation(cells=self.cells, n_classes=self.Y_dim, name="GRU_generator")
		graph = gru.getGraph(z)
		return graph, graph

	#D = Generator
	def Discriminator(self, X, reuse=False):
		print("Discriminator:")
		
		h = None
		if self.x_shape==None:
			h = X
		else:
			h = tf.reshape(X, [-1, self.loader.extract_length*3])
		"""
		h = tf.reshape(h, [-1, self.Y_dim, 1, 1], name="discriminator_reshape0")
		print(X.get_shape())
		h = conv2d_simple(h, 1, self.conv_features, name="discriminator_conv_1", reuse=reuse) #14*14*16
		print(h.name+" "+str(h.get_shape()))
		h = conv2d_simple(h, self.conv_features, self.conv_features*2, name="discriminator_conv_2", reuse=reuse) #7*7*32
		print(h.name+" "+str(h.get_shape()))
		h = tf.reshape(h, [-1, self.final_conv_dims*1*self.conv_features*2], name="discriminator_reshape1")
		print(h.name+" "+str(h.get_shape()))
		"""
		h = tf.matmul(h, self.D_W1) + self.D_b1
		logits = tf.matmul(h, self.D_W2) + self.D_b2
		prob = tf.nn.sigmoid(logits)
		print(prob.name+" "+str(prob.get_shape()))
		return prob
		
	def GRUDiscriminator(self, X, reuse=False):
		z = tf.reshape(z, (-1, self.X_dim, 1))
		gru = GRUOperation(cells=self.cells, n_classes=1, name="GRU_discriminator")
		graph = gru.getGraph(z)
		return graph
		
	def plot(self, samples):
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

		return fig


	def xavier_init(self, size):
		in_dim = size[0]
		xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
		return tf.random_normal(shape=size, stddev=xavier_stddev)
		
	def Train(self, save_path="", restore_path="", iterations=10000, display_step = 1000, preview=False):
	
		tf.reset_default_graph() 
		
		self.init()
		
		""" AutoEncoder part """
		z_sample = self.Encoder(self.X)
		print("latent: "+str(z_sample.get_shape()))
			
		fake_prob = None
		fake_logits = None
		
		if self.injected_z_dim>0:
			fake_prob, fake_logits = self.GeneratorInjected(z_sample, self.injected_z)
		else:
			fake_prob, fake_logits = self.Generator(z_sample)
		# E[log P(X|z)]
		#recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, targets=self.X))
		#AE_solver = tf.train.AdamOptimizer().minimize(recon_loss, var_list=self.theta_P + self.theta_Q)
		""" Minimize the encoder """
		recon_loss = tf.reduce_mean(tf.pow(self.y - fake_logits, 2))
		AE_solver = tf.train.RMSPropOptimizer(self.lr).minimize(recon_loss, var_list=self.theta_P + self.theta_Q)
		
		""" Adversarial part """
		# Adversarial loss to approx. Q(z|X)
		D_real = self.Discriminator(self.y)
		D_fake = self.Discriminator(fake_logits, reuse=True)
  
		D_loss = -tf.reduce_mean(tf.log(D_real + 1e-10) + tf.log(1. - D_fake + 1e-10))
		G_loss = -tf.reduce_mean(tf.log(D_fake + 1e-10))

		D_solver = tf.train.AdamOptimizer(self.lr).minimize(D_loss, var_list=self.theta_D)
		G_solver = tf.train.AdamOptimizer(self.lr).minimize(G_loss, var_list=self.theta_P)
  
		# Sample from random z
		Test_Generator = None
		if self.injected_z_dim>0:
			_, Test_Generator = self.GeneratorInjected(self.z, self.injected_z)
		else:
			_, Test_Generator = self.Generator(self.z, reuse=True)
			
		Test_Encoder = self.Encoder(self.X)

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
				
			for it in range(iterations):
				batch = self.loader.getNextFeatureBatch(self.batch_size, n_reccurent_input = self.n_reccurent_input)
				X_batch  = batch[0]
				Y_batch  = batch[1]
				#z_batch = np.random.randn(self.batch_size, self.z_dim)
				AE_loss = float(-1)
				D_loss_curr = float(-1)
				G_loss_curr = float(-1)
				
				if self.injected_z_dim>0:
					past_batch = batch[2]
					_, AE_loss = sess.run([AE_solver, recon_loss], feed_dict={self.X: X_batch, self.y: Y_batch, self.injected_z:past_batch})
					_, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={self.X: X_batch, self.y: Y_batch, self.injected_z:past_batch})
					_, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={self.X: X_batch, self.y: Y_batch, self.injected_z:past_batch})
				else:
					_, AE_loss = sess.run([AE_solver, recon_loss], feed_dict={self.X: X_batch, self.y: Y_batch})
					_, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={self.X: X_batch, self.y: Y_batch})
					_, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={self.X: X_batch, self.y: Y_batch})
	    
				if it % display_step == 0:
					print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}; '
						  .format(it, D_loss_curr, G_loss_curr)+" AE_loss: "+str(AE_loss))
					
					if preview == "image":
						rand_number = X_batch[0]
						test_latent = sess.run(Test_Encoder, feed_dict={self.X:[rand_number]})[0]
						
						
						samples = sess.run(Test_Generator, feed_dict={self.z: [test_latent]*1})
						
						print(str(len(samples))+" samples")
						
						fig = self.plot(samples)
						init_fig = self.plot([rand_number])
						display(init_fig)
						display(fig)
					elif preview == "graph":
						rand_number = X_batch[0]
						original = Y_batch[0]
						test_latent = sess.run(Test_Encoder, feed_dict={self.X:[rand_number]})[0]

						if self.injected_z_dim>0:
							past_batch = batch[2][0]
							samples = sess.run(Test_Generator, feed_dict={self.z: [test_latent]*1, self.injected_z:[past_batch]})
						else:
							samples = sess.run(Test_Generator, feed_dict={self.z: [test_latent]*1})
							
						A=plt.plot(samples[0], label='Generated', color="red")
						B=plt.plot(original, label='Original' , color="blue")
						plt.legend(['Generated', 'Original'])
						plt.show()
		
			if len(save_path)>0:
				# Save model weights to disk
				s_path = saver.save(sess, save_path+"/model")
				print ("Model saved in file: %s" % s_path)
				
	def BasicSeqEncode(self, restore_path, sequence):
		seq = []
		for i in range(len(sequence)):
			if i%self.Y_dim==0:
				image = sequence[i-self.Y_dim:i]
                
				if len(image)>0:
					fft_img = np.fft.fft(image)
					seq.append(fft_img[0:12])
		print("Raw Latent")
		plt.plot(seq)
		plt.show()
    
		return self.Encode(restore_path, seq)
            
	def Encode(self, restore_path, _input):
		tf.reset_default_graph() 
		
		self.init()
		
		Encoder = self.Encoder(self.X)
		
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
				
			self.result = sess.run(Encoder, feed_dict={self.X: _input})
		return self.result
	
	def EncodeSequence(self, restore_path, sequence):
		
		seq = []
		tmp = []
		for i in range(len(sequence)):
			if i%self.X_dim==0:
				if len(tmp)==self.X_dim:
					seq.append(tmp)
				tmp = []

			tmp.append(sequence[i])
			
		return self.Encode(restore_path, seq)
		
	def Generate(self, restore_path, latent, no_reshape=False):
		tf.reset_default_graph() 
		
		self.init()
		 
		latent_z = None
		if self.injected_z_dim>0:
			latent_z = self.z+self.injected_z
		else:
			latent_z = self.z
		
		Generator = self.Generator(latent_z, no_reshape=no_reshape)
		
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
				
			self.result = sess.run(Generator, feed_dict={self.z: latent})
		return self.result
		
	def GenerateSequence(self, restore_path, feature_map, no_reshape=False):
		tf.reset_default_graph() 
		
		self.init()
		 
		latent_z = None
		Generator = None
		
		if self.injected_z_dim>0:
			Generator = self.GeneratorInjected(self.z, self.injected_z, no_reshape=no_reshape)
		else:
			Generator = self.Generator(self.z, no_reshape=no_reshape)
		
		
		
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
				
			for f in feature_map:
				res = None
				if self.injected_z_dim>0:
					last_res = self.extract_result(len(self.result)-self.injected_z_dim, len(self.result))
					res = sess.run(Generator, feed_dict={self.z: [f], self.injected_z:[last_res]})[0]
				else:
					res = sess.run(Generator, feed_dict={self.z: [f]})[0]
					
				for r in res:
					for k in r:
						self.result.append(k)
				
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
	