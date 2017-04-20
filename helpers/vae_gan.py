import tensorflow as tf
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
					hidden_size = 128,
					injected_z = None,
					):
		
		self.n_reccurent_input = injected_z
		self.loader = loader
		self.batch_size = batch_size
		self.z_dim = z_size
		
		if X_dim == None:
			self.X_dim = loader.getImageBytes()
		else:
			self.X_dim = X_dim
			
		if injected_z!=None:
			self.injected_z_dim = injected_z
		else:
			self.injected_z_dim = 0
			
		self.h_dim = hidden_size
		self.lr = learning_rate

	def init(self):
		
		""" Q(z|X) """
		self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim], name="x_input")
		self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="z_latent")
		
		if self.injected_z_dim>0:
			self.injected_z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="z_latent")

		self.Q_W1 = tf.Variable(self.xavier_init([self.X_dim, self.h_dim]), name="Q_W1")
		self.Q_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]), name="Q_b1")

		self.Q_W2 = tf.Variable(self.xavier_init([self.h_dim, self.z_dim]), name="Q_W2")
		self.Q_b2 = tf.Variable(tf.zeros(shape=[self.z_dim]), name="Q_b2")

		self.theta_Q = [self.Q_W1, self.Q_W2, self.Q_b1, self.Q_b2]

		""" P(X|z) """
		self.P_W1 = tf.Variable(self.xavier_init([self.z_dim, self.h_dim]), name="P_W1")
		self.P_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]), name="P_b1")

		self.P_W2 = tf.Variable(self.xavier_init([self.h_dim, self.X_dim]), name="P_W2")
		self.P_b2 = tf.Variable(tf.zeros(shape=[self.X_dim]), name="P_b2")

		self.theta_P = [self.P_W1, self.P_W2, self.P_b1, self.P_b2]

		""" D(z) """
		self.D_W1 = tf.Variable(self.xavier_init([self.X_dim, self.h_dim]), name="D_W1")
		self.D_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]), name="D_b1")

		self.D_W2 = tf.Variable(self.xavier_init([self.h_dim, 1]), name="D_W2")
		self.D_b2 = tf.Variable(tf.zeros(shape=[1]), name="D_b2")

		self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]
		
	#Q = Encoder
	def Encoder(self, X):
		h = tf.nn.relu(tf.matmul(self.X, self.Q_W1) + self.Q_b1)
		z = tf.matmul(h, self.Q_W2) + self.Q_b2
		return tf.nn.sigmoid(z)

	#P = Discriminator
	def Generator(self, z):
		h = tf.nn.relu(tf.matmul(z, self.P_W1) + self.P_b1)
		logits = tf.matmul(h, self.P_W2) + self.P_b2
		prob = tf.nn.sigmoid(logits)
		return prob, logits

	#D = Generator
	def Discriminator(self, X):
		h = tf.nn.relu(tf.matmul(X, self.D_W1) + self.D_b1)
		logits = tf.matmul(h, self.D_W2) + self.D_b2
		prob = tf.nn.sigmoid(logits)
		return prob
		
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
		
		latent_z = None
		if self.injected_z_dim>0:
			latent_z = z_sample+self.injected_z
		else:
			latent_z = z_sample
			
		fake_prob, fake_logits = self.Generator(latent_z)

		# E[log P(X|z)]
		#recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, targets=self.X))
		#AE_solver = tf.train.AdamOptimizer().minimize(recon_loss, var_list=self.theta_P + self.theta_Q)
		
		recon_loss = tf.reduce_mean(tf.pow(self.X - fake_logits, 2))
		AE_solver = tf.train.RMSPropOptimizer(self.lr).minimize(recon_loss, var_list=self.theta_P + self.theta_Q)
		
		""" Adversarial part """
		# Adversarial loss to approx. Q(z|X)
		D_real = self.Discriminator(self.X)
		D_fake = self.Discriminator(fake_logits)

		D_loss = -tf.reduce_mean(tf.log(D_real + 1e-10) + tf.log(1. - D_fake + 1e-10))
		G_loss = -tf.reduce_mean(tf.log(D_fake + 1e-10))

		D_solver = tf.train.AdamOptimizer(self.lr).minimize(D_loss, var_list=self.theta_D)
		G_solver = tf.train.AdamOptimizer(self.lr).minimize(G_loss, var_list=self.theta_Q)
		
		# Sample from random z
		Test_Generator = None
		if self.injected_z_dim>0:
			_, Test_Generator = self.Generator(self.z+self.injected_z)
		else:
			_, Test_Generator = self.Generator(self.z)
			
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
				batch = self.loader.getNextBatch(self.batch_size, n_reccurent_input = self.n_reccurent_input)
				X_batch  = batch[0]
				#z_batch = np.random.randn(self.batch_size, self.z_dim)
				AE_loss = None
				D_loss_curr = None
				G_loss_curr = None
				
				if self.injected_z_dim>0:
					past_batch = batch[2]
					_, AE_loss = sess.run([AE_solver, recon_loss], feed_dict={self.X: X_batch, self.injected_z:past_batch})
					_, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={self.X: X_batch, self.injected_z:past_batch})
					_, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={self.X: X_batch, self.injected_z:past_batch})
				else:
					_, AE_loss = sess.run([AE_solver, recon_loss], feed_dict={self.X: X_batch})
					_, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={self.X: X_batch})
					_, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={self.X: X_batch})
	    
				if it % display_step == 0:
					print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}; AE_loss: {:.4}'
						  .format(it, D_loss_curr, G_loss_curr, AE_loss))
					
					if preview == "image":
						rand_number = X_batch[0]
						test_latent = sess.run(Test_Encoder, feed_dict={self.X:[rand_number]})[0]
						
						
						samples = sess.run(Test_Generator, feed_dict={self.z: [test_latent]*1})
						
						fig = self.plot(samples)
						init_fig = self.plot([rand_number])
						display(init_fig)
						display(fig)
					elif preview == "graph":
						rand_number = X_batch[0]
						test_latent = sess.run(Test_Encoder, feed_dict={self.X:[rand_number]})[0]
						
						if self.injected_z_dim>0:
							past_batch = batch[2][0]
							samples = sess.run(Test_Generator, feed_dict={self.z: [test_latent]*1, self.injected_z:[past_batch]})
						else:
							samples = sess.run(Test_Generator, feed_dict={self.z: [test_latent]*1})
							
						A=plt.plot(samples[0], label='Generated', color="red")
						B=plt.plot(rand_number, label='Original' , color="blue")
						plt.legend(['Generated', 'Original'])
						plt.show()
		
			if len(save_path)>0:
				# Save model weights to disk
				s_path = saver.save(sess, save_path+"/model")
				print ("Model saved in file: %s" % s_path)
				
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
		
	def Generate(self, restore_path, latent):
		tf.reset_default_graph() 
		
		self.init()
		 
		latent_z = None
		if self.injected_z_dim>0:
			latent_z = self.z+self.injected_z
		else:
			latent_z = self.z
		
		Generator = self.Generator(latent_z)
		
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
		
	def GenerateSequence(self, restore_path, feature_map):
		tf.reset_default_graph() 
		
		self.init()
		 
		latent_z = None
		if self.injected_z_dim>0:
			latent_z = self.z+self.injected_z
		else:
			latent_z = self.z
		
		Generator = self.Generator(latent_z)
		
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
				last_res = self.extract_result(len(self.result)-self.injected_z_dim, len(self.result))
				res = None
				if self.injected_z_dim>0:
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
	