# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 22:11:47 2017

@author: Boris
"""

import os
import io
from multiprocessing import Process, Lock, Manager, Value
import pydub
import requests
from random import *
import random as rand
from urllib import request
from urllib.request import urlopen
from pathlib import Path

import numpy as np

import soundfile as sf
from PIL import Image
from matplotlib.pyplot import imshow
from IPython.display import clear_output, display, HTML
import _thread as thread
import time
import pickle
from os import listdir

deezer_api_url = "http://api.deezer.com/"

class SoundCombiner:
    def __init__(self, loaders):
        self.loaders = loaders
        self.LABELS_COUNT = loaders[0].LABELS_COUNT
        self.fixed_size = loaders[0].fixed_size
        self.converter = loaders[0].converter
        self.sample_shape = loaders[0].sample_shape
        self.batch_size = loaders[0].batch_size
        self.n_steps = loaders[0].n_steps
        self.label_offset = loaders[0].label_offset
        self.uLawEncode = loaders[0].uLawEncode
        self.samplerate = loaders[0].samplerate
        self.one_hot = loaders[0].one_hot
        self.insert_global_input_state = loaders[0].insert_global_input_state
        self.extract_length = loaders[0].extract_length
        self.sample_steps = loaders[0].sample_steps
        self.entropy = loaders[0].entropy
        self.image_width = loaders[0].image_width
        self.pool = []
        
    def getTestTimeBatch(self, 
                         batch_size, 
                         shuffle_rate=-1, 
                         n_steps=32):
        return self.getNextTimeBatch(batch_size, shuffle_rate, n_steps)
            
    def getNextTimeBatch(self, batch_size, 
                         shuffle_rate=-1, 
                         n_steps=32):
        return rand.choice(self.loaders).getNextTimeBatch(batch_size, shuffle_rate, n_steps)
    
    def getImageBytes(self):
        if self.fixed_size>0:
            return self.fixed_size
        else:
            return self.image_width*self.image_width
    def getImageWidth(self):
        return self.image_width 
class SoundLoader:
    def __init__(self, 
                 sound,
                 extract_length = 32,
                 LABELS_COUNT=-1, 
                 fixed_size=-1, 
                 sample_size=1/15, 
                 insert_global_input_state = -1,
                 one_hot = -1,
                 uLaw = -1,
                 log = False,
                 samplerate = 44100,
                 random = True,
                 multiplier = 1,
                 amplitude = 1,
                 label_offset = 0,
                 sample_shape = [], #[[2, 20, 200]]*64
                 n_steps = 16,
                 batch_size = 128,
                 n_threads = 0,
                 max_pool = 200,
                 entropy = None #{"step":1, "increase_rate":0, "max_step":50, "size":64},
                 ):
        self.sound = sound
        self.log = log
        self.random = random
        self.label_offset = label_offset
        self.sample_shape = sample_shape
        self.entropy = entropy
        
        if self.entropy!=None:
            step = self.entropy["step"]
            state = 0
            for k in range(self.entropy["size"]):
                state+=step
                if step<self.entropy["max_step"]:
                    step+=self.entropy["increase_rate"]
            print ("Entropy range: "+str(state)+" frames.")
        
        if(LABELS_COUNT!=-1):
            self.LABELS_COUNT = LABELS_COUNT
        else:
            self.LABELS_COUNT = extract_length
            
        self.sample_size = sample_size
        self.fixed_size = fixed_size
        
        self.converter = SoundConverter(sound)
        self.converter.resample(samplerate)
        
        #if len(sample_shape)>0:
        #    for sample_r in sample_shape[0]:
        #        self.converter.resample_as_clone(sample_r)
        self.use_avg = False
        self.sample_avg = -1
        
        self.insert_global_input_state = insert_global_input_state
        self.one_hot = one_hot
        
        if(extract_length!=-1):
            self.extract_length = extract_length
        else:
            self.extract_length = LABELS_COUNT
        
        if one_hot!=-1:
             if str(one_hot.__class__())!="[]":
                 self.LABELS_COUNT = one_hot
                 self.extract_length = 1
             else:
                 self.LABELS_COUNT = one_hot[0]*one_hot[1]
                 self.extract_length = one_hot[1]
        
        if fixed_size!=-1 and self.extract_length!=-1:
            self.fixed_size += self.extract_length
        
        test_audio = self.converter.ExtractRandomSample(sample_size=sample_size, multiplier=multiplier)
        self.sound_length = len(self.converter.data)
        
        self.multiplier = multiplier
        self.amplitude = amplitude
        
        if fixed_size<=0:
            self.image_width = int(np.sqrt(len(test_audio)))
        else:
            self.image_width = int(np.sqrt(fixed_size))
            
        self.uLawEncode = uLaw
        
        self.extract_log = []
        self.samplerate = samplerate
        
        self.pool = []
        self.raw_pool = None
        
        self.active_threads = 0
        self.max_threads = 4
        self.running = True
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_threads = n_threads
        if n_threads!=0:
            self.max_pool = max_pool/n_threads
        else:
            self.max_pool = max_pool
            
        self.cache_file="temp/sound_pool/"
        self.file_counter = 0
        self.use_file_system = False
        
        lock = Lock()
        for k in range(n_threads):
            thread.start_new_thread(self.factory_generate_samples, (lock, self.batch_size, self.n_steps, k, self.raw_pool))
            print ("Thread started, id: "+str(k))
            
    
    def getExtract(self, startRatio, length):
        return self.converter.SoundToImage(compress=False, offset=self.multiplier/2, multiplier=self.multiplier*self.amplitude, ratio=1, sample_range=[startRatio, 1], uLawEncode = self.uLawEncode, log=False, as_tensors=True)
        
    def getTestTimeBatch(self, 
                         batch_size, 
                         shuffle_rate=-1, 
                         n_steps=32):
        return self.getNextTimeBatch(batch_size, shuffle_rate, n_steps)
            
    def getNextTimeBatch(self, 
                         batch_size, 
                         shuffle_rate=-1, 
                         n_steps=32):
        if self.n_threads>0:
            if self.use_file_system:
                while len(listdir(self.cache_file))==0:
                    wait = 0
                    
                file_name = listdir(self.cache_file)[self.file_counter]
                self.file_counter+=1
                if os.path.exists(self.cache_file+file_name):
                    data = None
                    f = open(self.cache_file+file_name, 'rb')
                    data = pickle.load(f)
                    #os.remove(self.cache_file+file_name)
                    return data
            else:
                while True:
                    if len(self.pool)>0:
                        return self.pool.pop(0)
                   
        else:
            self.thread_getNextTimeBatch(batch_size, shuffle_rate, n_steps)
            return self.pool.pop(0)
                
    def factory_generate_samples(self, lock, batch_size, steps, _id, raw_pool):
        self.thread_tmp = 0
        while self.running:
            if self.use_file_system:
                if len(listdir(self.cache_file))<self.max_pool:
                    self.thread_getNextTimeBatch(batch_size, -1, steps, lock, _id, raw_pool)
                    time.sleep(0.01)
            else:
                if len(self.pool)<self.max_pool:
                    self.thread_getNextTimeBatch(batch_size, -1, steps, lock, _id, raw_pool)
                    
        print("Thread stopped: "+str(_id))
        
    def thread_getNextTimeBatch(self, 
                         batch_size, 
                         shuffle_rate=-1, 
                         n_steps=32,
                         lock=None,
                         _id = None, 
                         raw_pool = None):
            
        images = []
        labels = []

        converter = self.converter
        
        while len(images)<batch_size:
            image = None
            
            if self.random:
                image = converter.ExtractRandomSample(fixed_size=self.fixed_size+self.label_offset, sample_size=self.sample_size, multiplier=self.multiplier*self.amplitude, offset=self.multiplier/2, uLawEncode = self.uLawEncode)
            else:
                image = converter.ExtractNextSample(fixed_size=self.fixed_size+self.label_offset, sample_size=self.sample_size, multiplier=self.multiplier*self.amplitude, offset=self.multiplier/2, uLawEncode = self.uLawEncode)
            
            last_start_index = converter.last_start_index
                
            label = image[len(image)-self.extract_length:len(image)]
            
            if self.one_hot!=-1:
                label = Encoder.OneHot(label, self.one_hot)
                
            labels.append(label)
            
            #print("init: "+str(converter.last_start_index)+" len: "+str(len(image[0:len(image)-self.extract_length-self.label_offset])))
            
            image = converter.reshapeAsSequence(image[0:len(image)-self.extract_length-self.label_offset], n_steps)
            offset = 0
            if len(self.sample_shape)!=0:
                samples = []
                sample_length = len(self.sample_shape)
                for sample_range in self.sample_shape[0]:
                    
                    sample = []
                    for frame_id in range(sample_length):
                        if self.use_avg:
                            val = 0
                            small_sample = converter.Extract(last_start_index-offset-sample_range, last_start_index-offset, multiplier=self.multiplier*self.amplitude, offset=self.multiplier/2, uLawEncode = self.uLawEncode)
                            #process small_sample
                            if self.sample_avg>0:
                                val = Encoder.avg(small_sample, self.sample_avg)
                            else:
                                if self.uLawEncode!=-1:
                                    val = np.average(small_sample)
                                else:
                                    val = Encoder.uLawEncode(np.average(small_sample), self.uLawEncode)
                            sample.append(val)
                        else:
                            small_sample = converter.Extract(last_start_index-offset-1, last_start_index-offset, multiplier=self.multiplier*self.amplitude, offset=self.multiplier/2, uLawEncode = self.uLawEncode)
                            #process small_sample
                            val = small_sample[0]
                            sample.append(val)
                        
                        offset += sample_range
                    
                    sample = converter.reshapeAsSequence(sample, n_steps)
                    
                    samples = sample + samples
                    
                image = samples + image
            elif self.entropy != None:
                size = self.entropy["size"]
                step = self.entropy["step"]
                increase_rate = self.entropy["increase_rate"]
                max_step = self.entropy["max_step"]
                
                offset = 0
                sample = []

                for k in range(size):
                    small_sample = converter.Extract(last_start_index-offset-step, last_start_index-offset, multiplier=self.multiplier*self.amplitude, offset=self.multiplier/2, uLawEncode = self.uLawEncode)
                    
                    val = 0
                    if self.sample_avg>0:
                        val = Encoder.avg(small_sample, self.sample_avg)
                    else:
                        if self.uLawEncode!=-1:
                            val = Encoder.entropy(small_sample)
                        else:
                            val = Encoder.uLawEncode(Encoder.entropy(small_sample), self.uLawEncode)
                    
                    sample.insert(0, val) 
                    
                    offset += step 
                    if step<max_step:
                        step+=increase_rate
                        
                #sample = np.flip(sample, 0).tolist()
                sample = converter.reshapeAsSequence(sample, n_steps)
                self.debug_offset = offset
                self.debug_max_step = step
                image = sample + image
            else:
                if self.insert_global_input_state!=-1:
                    if self.uLawEncode!=-1:
                        image = [[Encoder.uLawEncode(converter.last_sample_position, self.uLawEncode)]*len(image[0])]*self.insert_global_input_state+image
                    else:
                        image = [[converter.last_sample_position]*len(image[0])]*self.insert_global_input_state+image
            if lock==None:
                self.extract_log.append(converter.last_sample_position)
                     
                
            images.append(image)
        
        if lock!=None:
            lock.acquire()
            if self.log:
                print("pool: "+str(self.thread_tmp)+" extracted "+str(len(images))+" samples from "+self.sound)
            
            if self.use_file_system:
                #instance.pool.append([images, labels])
                with open(self.cache_file+"_t_"+str(_id)+"_"+str(self.thread_tmp), 'wb') as f:
                    pickle.dump([images, labels], f, pickle.HIGHEST_PROTOCOL)
                    f.close()
                self.thread_tmp+=1
            else:
                self.pool.append([images, labels])
            lock.release()
        else:
            if self.log:
                print("extracted "+str(len(images))+" samples from "+self.sound)
                
            self.pool.append([images, labels])

    def getTestBatch(self, 
                         batch_size, 
                         shuffle_rate=-1):
        return self.getNextBatch(batch_size, shuffle_rate)

    def getNextBatch(self, 
                         batch_size, 
                         shuffle_rate=-1):
        images = []
        labels = []

        converter = self.converter
        
        while len(images)<batch_size:
            image = None
            if self.random:
                image = converter.ExtractRandomSample(fixed_size=self.fixed_size, sample_size=self.sample_size, multiplier=self.multiplier*self.amplitude, offset=self.multiplier/2, uLawEncode = self.uLawEncode)
            else:
                image = converter.ExtractNextSample(fixed_size=self.fixed_size, sample_size=self.sample_size, multiplier=self.multiplier*self.amplitude, offset=self.multiplier/2, uLawEncode = self.uLawEncode)
                
            label = image[len(image)-self.extract_length+self.label_offset:len(image)+self.label_offset]
            
            if self.one_hot!=-1:
                label = Encoder.OneHot(label, self.one_hot)

            labels.append(label)
            
            image = image[0:self.fixed_size-self.extract_length]
            
            self.extract_log.append(converter.last_sample_position)
            
            if len(self.sample_shape)!=0:
                samples = []
                for clone_id in self.sample_shape[0]:
                    sample = converter.ExtractFromClone(clone_id, converter.last_start_index-len(self.sample_shape), converter.last_start_index, multiplier=self.multiplier*self.amplitude, offset=self.multiplier/2, uLawEncode = self.uLawEncode)
                    
                    samples = samples + sample
                    
                image = samples + image
            
            if self.insert_global_input_state:
                if self.uLawEncode!=-1:
                    image[0] = Encoder.uLawEncode(converter.last_sample_position, self.uLawEncode)
                else:
                    image[0] = converter.last_sample_position
                    
                
            images.append(image)
        
        if self.log:
            print("extracted "+str(len(images))+" samples from "+self.sound)
        
        return [images, labels]    

    def getImageBytes(self):
        if self.fixed_size>0:
            return self.fixed_size
        else:
            return self.image_width*self.image_width
    def getImageWidth(self):
        return self.image_width 
        
class SoundImage:
    def __init__(self, width, height, sample_rate, r=None, g=None, b=None, a=None):
        if r==None:
            self.r = []
            self.g = []
            self.b = []
            self.a = []

            for i in range(width*height):
                self.r.append(255)
                self.g.append(255)
                self.b.append(255)
                self.a.append(255)
        else:
            self.r = r
            self.g = g
            self.b = b
            self.a = a

        self.width = width
        self.height = height
        self.sample_rate = sample_rate
        self.compressed = False

    def load(self, path, sample_rate=44100):
        img = Image.open(path)
        pixels = img.load()
        self.sample_rate = sample_rate
        self.width = img.size[0]
        self.height = img.size[0]

        for i in range(self.width*self.height):
            self.r.append(255)
            self.g.append(255)
            self.b.append(255)
            self.a.append(255)

        for i in range(img.size[0]):    # for every pixel:
            for j in range(img.size[1]):
                self.setPixel(i, j, pixels[i,j][0], pixels[i,j][1], pixels[i,j][2], 255)
                
    def _import(self, pixels, sample_rate=44100, multiplier=1024):
        self.sample_rate = sample_rate
        self.width = int(np.sqrt(len(pixels)))
        self.height = int(np.sqrt(len(pixels)))

        for i in range(self.width*self.height):
            self.r.append(255)
            self.g.append(255)
            self.b.append(255)
            self.a.append(255)

        for i in range(self.width):    # for every pixel:
            for j in range(self.height):
                self.setPixel(i, j, pixels[i+j*self.width]*multiplier, pixels[i+j*self.width]*multiplier, pixels[i+j*self.width]*multiplier, 255)

    def getPixel(self, x, y):
        index = y*self.width+x
        return [max(0, int(self.r[index])), max(0, int(self.g[index])), max(0, int(self.b[index])), max(0, int(self.a[index]))]

    def getPixelAtIndex(self, index):
        return [int(self.r[index]), int(self.g[index]), int(self.b[index]), int(self.a[index])]

    def setPixel(self, x, y, r, g, b, a):
        index = y*self.width+x
        self.r[index] = r
        self.g[index] = g
        self.b[index] = b
        self.a[index] = a

    def setPixelAtIndex(self, index, r, g, b, a):
        self.r[index] = r
        self.g[index] = g
        self.b[index] = b
        self.a[index] = a

    def getTensor(self, image_format="Grayscale"):
        _image = []
        if self.format=="Grayscale":
            for index in range(self.width*self.height):
                pixel = (self.r[index]+self.g[index]+self.b[index])/3
                _image.append(pixel)
        return _image

    def show(self):
        img = Image.new( 'RGBA', (self.width,self.height), "white")
        pixels = img.load()

        for i in range(img.size[0]):    # for every pixel:
            for j in range(img.size[1]):
                pixel = self.getPixel(i, j)
                pixels[i,j] = (pixel[0], pixel[1], pixel[2], pixel[3]) # set the colour accordingly

        img.show()

    def preview(self):
        img = Image.new( 'RGBA', (self.width,self.height), "white")
        pixels = img.load()

        for i in range(img.size[0]):    # for every pixel:
            for j in range(img.size[1]):
                pixel = self.getPixel(i, j)
                pixels[i,j] = (pixel[0], pixel[1], pixel[2], pixel[3]) # set the colour accordingly

        display(img)

    def write(self, path):
        img = Image.new( 'RGB', (self.width,self.height), "white")
        pixels = img.load()

        for i in range(img.size[0]):    # for every pixel:
            for j in range(img.size[1]):
                pixel = self.getPixel(i, j)
                pixels[i,j] = (pixel[0], pixel[1], pixel[2]) # set the colour accordingly

        img.save(path)

    def getSize(self):
        return self.width*self.height
        
class SoundConverter:
     def __init__(self, path="", isUrl=False):
         self.path = path
         self.isUrl = isUrl
         self.state = 0
         self.clones = {}
         self.flat_data = None
         
         if len(path)>0:
             data, samplerate = sf.read(self.path)
             self.data = data
             self.samplerate = samplerate
         
     def ExtractSamples(self, samples_count=10, offset=127, multiplier=2048, ratio=1, as_tensors=False):
         result = []
         for i_counter in range(samples_count):
             sample_len = 1/samples_count
             rand_start = np.min(rand.uniform(0.0, 1.0), 1-sample_len)
             sample_range = [rand_start, rand_start+sample_len]
             result.append(self.SoundToImage(compress=False, offset=offset, multiplier=multiplier, ratio=ratio, sample_range=sample_range, log=False, as_tensors=as_tensors))
         return result
         
     def ExtractNextSample(self, limiter=99999999999, fixed_size=-1, sample_size=1/15, offset=127, multiplier=2048, ratio=1, as_tensors=True, uLawEncode=False):
         sample_len = sample_size
         
         if fixed_size!=-1:
             self.state+=1/len(self.data)
         else:
             self.state+=sample_size
             
         if self.state>=1:
             self.state=0
             
         self.last_sample_position = self.state
         
         sample_range = [self.state, 1]

         #print(str(sample_range[1]-sample_range[0])+"=="+str(sample_len))

         return (self.SoundToImage(fixed_size=fixed_size, compress=False, offset=offset, multiplier=multiplier, ratio=ratio, sample_range=sample_range, log=False, as_tensors=as_tensors, limiter=limiter, uLawEncode=uLawEncode))

     def ExtractRandomSample(self, limiter=99999999999, fixed_size=-1, sample_size=1/15, offset=127, multiplier=2048, ratio=1, as_tensors=True, uLawEncode=-1):
         sample_len = sample_size
         rand_start = rand.uniform(0.0, 1.0-sample_len)
         
         self.last_sample_position = rand_start
         
         sample_range = [rand_start, rand_start+sample_len]

         #print(str(sample_range[1]-sample_range[0])+"=="+str(sample_len))

         return (self.SoundToImage(fixed_size=fixed_size, compress=False, offset=offset, multiplier=multiplier, ratio=ratio, sample_range=sample_range, log=False, as_tensors=as_tensors, limiter=limiter, uLawEncode=uLawEncode))

     def GetEncodedBackground(self, start=0, length=256*256, offset=127, multiplier=2048, ratio=1):
         data = self.data
         samplerate = self.samplerate
         sample_begin = 0
         sample_end = start
         
         i = [0]*length

         index = 0
         
         #blurried part...
         while index<length:
             i[index] = self.getAvgFrame(start/length, int(index/start))
            
             index+=1
         
         #clear part
         index = sample_begin
         
         while index<sample_end:
             i.append(min(1, ((data[index][0]+data[index][1])/2*multiplier+offset)/255))
            
             index+=1
         
         return i
        
     def getAvgFrame(_range, index):
         todo=0
         
     def SoundToImage(self, fixed_size=-1, limiter=99999999999, compress=False, offset=127, multiplier=2048, ratio=1, sample_range=[0,1], log=True, as_tensors=False, uLawEncode=-1):

         data = self.data
         samplerate = self.samplerate
         
         sample_begin = int(len(data)*sample_range[0])
         self.last_start_index = sample_begin
         
         sample_end = int(len(data)*sample_range[1])
         if fixed_size>0:
             sample_end = sample_begin+fixed_size

         diff = sample_end-sample_begin
        
         i = None
         
         if as_tensors:
             i = []
         else:
             i = SoundImage(int(np.sqrt(diff)*0.7*(ratio)), int(np.sqrt(diff)*0.7*(1/ratio)), samplerate)
             i.compressed = compress

         if compress==False:
             diff=0
         if log:
             print("image initialized, begin: "+str(sample_begin)+" end: "+str(sample_end))

         counter = 0
         pixel = 0
         index = sample_begin
         while index<sample_end-diff/2:
            if counter>len(data)/20:
                if log:
                    print("status: "+str(index)+"/"+str(data[index]))
                counter=0
            counter+=1
            
            if as_tensors:
                if pixel<limiter:
                    
                    if index>=len(data):
                        val = ((data[0][0]+data[0][1])/2)*multiplier+offset
                    else:
                        val = ((data[index][0]+data[index][1])/2)*multiplier+offset
                    
                    if uLawEncode!=-1:
                        i.append(Encoder.uLawEncode(val, uLawEncode))
                    else:
                        i.append(val)
            else:
                if pixel<i.getSize():
                    if compress:
                        i.setPixelAtIndex(pixel, data[index][0]*multiplier+offset, data[index][1]*multiplier+offset, data[sample_end-index-1][0]*multiplier+offset, data[sample_end-index-1][1]*multiplier+offset)
                    else:
                        i.setPixelAtIndex(pixel, data[index][0]*multiplier+offset, data[index][1]*multiplier+offset, 0, 255)

            #f.write(data*2)

            index+=1
            pixel+=1

         if log:
             print("image generated: "+str(i.width)+"x"+str(i.height))
         return i

     def ImageToSound(self, image:SoundImage, path, multiplier=5000):
         samplerate = image.sample_rate
         data = []
         index = 0
         while index<image.getSize():
            pixel = image.getPixelAtIndex(index)
            data.append([pixel[0]*multiplier, pixel[1]*multiplier])
            index+=1

         if image.compressed:
            index = 0
            while index<image.getSize():
                pixel = image.getPixelAtIndex(image.getSize()-index-1)
                data.append([pixel[2]*multiplier, pixel[3]*multiplier])
                index+=1

         sf.write(path, data, samplerate)
         print("Sound saved as: "+path)
         
     def TensorToSound(self, tensor, path, offset=0, multiplier=5000, samplerate=44100):
         data = []
         index = 0
         while index<len(tensor):
            pixel = tensor[index]            
            data.append([pixel*multiplier+offset, pixel*multiplier+offset])
            index+=1
            
         sf.write(path, data, samplerate)
         print("Sound saved as: "+path)
         return data
         
     def reshapeAsSequence(self, data, sequences=1):
         new_len = int(len(data)/sequences)
         new_data = []
         for i in range(sequences):
                 new_data.append(data[i*new_len:(i+1)*new_len])
                 
         return new_data
         
     def Extract(self, start_frame, end_frame, offset=127, multiplier=2048, uLawEncode = -1):
         data = self.getFlatData(offset, multiplier, uLawEncode)
         extract = data[start_frame:end_frame]
         sample_len = end_frame-start_frame
         
         if len(extract)<sample_len:
             extract = [0.5]*(sample_len-len(extract))+extract
            
         return extract
         
     def getFlatData(self, offset=127, multiplier=2048, uLawEncode = -1):
         if self.flat_data==None:
             self.flat_data = []
             for i in range(len(self.data)):
                 if i<0:
                     val = (0.5)
                 elif i>=len(self.data):
                     val = (0.5)
                 else:
                     val = (self.data[i][0]*multiplier+offset)
                     
                 if uLawEncode!=-1:
                    self.flat_data.append(Encoder.uLawEncode(val, uLawEncode))
                 else:
                    self.flat_data.append(val)
             return self.flat_data
         else:
             return self.flat_data
         
     def ExtractFromClone(self, clone_id, start_frame, end_frame, offset=127, multiplier=2048, uLawEncode = -1):
         data = self.clones[clone_id]
         extract = []
         ratio = clone_id/self.samplerate
         
         length = end_frame-start_frame
         
         end_frame = int(end_frame*ratio)
         start_frame = end_frame-length
         
         #print(" samplerate: "+str(clone_id)+" start: "+str(start_frame)+" end: "+str(end_frame)+" length: "+str(end_frame-start_frame))

         i = start_frame
         
         while i<end_frame:
             if i<0:
                 val = (0.5)
             elif i>=len(data):
                 val = (0.5)
             else:
                 val = (data[i][0]*multiplier+offset)
                 
             if uLawEncode!=-1:
                extract.append(Encoder.uLawEncode(val, uLawEncode))
             else:
                extract.append(val)
             
             i+=1
        
         #print("clone: "+str(extract))
                
         return extract
         
     def resample_as_clone(self, samplerate):
         import scipy.signal as sps
         
         step = int(self.samplerate/samplerate)
         
         print("step: "+str(step)+" samplerate: "+str(samplerate))
         
         self.clones[samplerate] = None

         container = []
         for d in self.data:
             container.append(d[0])

         container = container[0:len(container):step]
         
         for d in range(len(container)):
             container[d] = [container[d], container[d]]

         self.clones[samplerate] = container
         
     def resample(self, samplerate):
         import scipy.signal as sps

         container = []
         for d in self.data:
             container.append(d[0])

         container = container[0:len(container):int(self.samplerate/samplerate)]
         
         for d in range(len(container)):
             container[d] = [container[d], container[d]]

         self.data = container
         self.samplerate = samplerate
         
class Encoder:
    def OneHot(data, labels=32, offset=0):
        output = []
        isMulti = False
        if str(labels.__class__())=="[]":
            labels = labels[0]
            isMulti = True
        
        for d in data:
            row = [0]*labels
            index = int(d*labels)
            
            if index<0:
                #raise Exception("Value: "+str(d)+" was out of bounds while encoding OneHot")
                index = 0
            elif index>labels-1:
                index = labels-1
                
            row[index] = 1
            
            if isMulti:
                output.append(row)
            else:
                output=output+row
                 
        return output
        
    #uLaw, default 8bit (256)
    def uLawEncode(data, u=1024):
        return np.floor(data/(1/u))*(1/u)
        
    def max_value(data=[]):
        _max = 0
        _max_index = -1
        for i in range(len(data)):
            if abs(data[i])>_max:
                _max_index = i
                _max = abs(data[i])
                
        return abs(data[_max_index])
        
    def avg(data=[], steps=-1):
        if steps <= 0:
            _sum = 0
            for v in data:
                _sum+=v
            return _sum/len(data)
        else:
            _sum = 0
            stride = int(len(data)/steps)
            
            if stride<=0:
                stride = 1
                
            counter = 0
            divider = 0
            for v in range(steps):
                if counter<len(data):
                    _sum+=data[counter]
                    divider+=1
                else:
                    break

                counter+=stride
        
            return _sum/divider
            
    def entropy(data=[], ground=0.5):
        _sum = 0
        for v in data:
            _sum+=abs(v-ground)
        return _sum/len(data)