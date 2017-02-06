import numpy as np
import pickle
import soundfile as sf
from PIL import Image
from matplotlib.pyplot import imshow
from IPython.display import clear_output, display, HTML
from urllib.request import urlopen
import io
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from random import *
import random as rand

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
         
     def ExtractRandomSample(self, limiter=99999999999, fixed_size=-1, sample_size=1/15, offset=127, multiplier=2048, ratio=1, as_tensors=True):
         sample_len = sample_size
         rand_start = rand.uniform(0.0, 1.0-sample_len)
         sample_range = [rand_start, rand_start+sample_len]

         #print(str(sample_range[1]-sample_range[0])+"=="+str(sample_len))

         return (self.SoundToImage(fixed_size=fixed_size, compress=False, offset=offset, multiplier=multiplier, ratio=ratio, sample_range=sample_range, log=False, as_tensors=as_tensors, limiter=limiter))

     def ExtractBlurrySample(self, start=0, length=256*256, offset=127, multiplier=2048, ratio=1):
         data = self.data
         samplerate = self.samplerate
         sample_begin = start
         sample_end = start+length
         
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
         
     def SoundToImage(self, fixed_size=-1, limiter=99999999999, compress=False, offset=127, multiplier=2048, ratio=1, sample_range=[0,1], log=True, as_tensors=False):

         data = self.data
         samplerate = self.samplerate
         
         sample_begin = int(len(data)*sample_range[0])
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
                    i.append(min(1, ((data[index][0]+data[index][1])/2*multiplier+offset)/255))
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
         
     def TensorToSound(self, tensor, path, multiplier=5000, samplerate=44100):
         data = []
         index = 0
         while index<len(tensor):
            pixel = tensor[index]            
            data.append([pixel*multiplier, pixel*multiplier])
            index+=1
            
         sf.write(path, data, samplerate)
         print("Sound saved as: "+path)
         
     def reshapeAsSequence(self, data, sequences=1):
         new_len = int(len(data)/sequences)
         new_data = []
         for i in range(sequences):
                 new_data.append(data[i*new_len:(i+1)*new_len])
                 
         return new_data

class CifarLoader:

    def __init__(self, path="inputs/datasets/cifar10/", image_format="Grayscale", IMG_SIZE=32*32, LABELS_COUNT=10, IMAGES_PER_FILE=10000, FILES_AMOUNT=5):
        self.fileID=0
        self.format = image_format
        self.IMG_SIZE = IMG_SIZE
        self.limiter = IMG_SIZE
        self.LABELS_COUNT = LABELS_COUNT
        self.IMAGES_PER_FILE = IMAGES_PER_FILE
        self.FILES_AMOUNT = FILES_AMOUNT
        self.data = []
        self.testData = {}
        self.loadAllFiles(path)

    def loadAllFiles(self, path):
        print("Loading files...")

        if self.FILES_AMOUNT<=1:
            self.loadFile(path+"train", False)
            self.loadFile(path+"test", True)
        else:
            for i in range(self.FILES_AMOUNT):
                file_name = path+"data_batch_"+str(i+1)
                self.loadFile(file_name)
    
            file_name = path+"test_batch"
            self.loadFile(path+"test_batch", True)

    def loadFile(self, file, isTest=False):
        fo = open(file, 'rb')
        u = pickle._Unpickler(fo)
        u.encoding = 'latin1'
        dict = u.load()
        fo.close()
        dict["training_state"] = 0

        print("Loaded: "+file)

        if isTest==False:
            self.data.append(dict)
        else:
            self.testData = dict

    """
        getNextBatch(int amount, int fileID=0):

        int[] image = new int[IMG_SIZE]
        int[] label = new int[LABELS_COUNT]

        return [imagesData List<image>, labels List<label>]

    """

    def getNextBatch(self, amount):

        imagesData = []
        labels = []
        flipped = []

        for i in range(amount):
            fileID = 0
            imgID = 0
            
            if self.FILES_AMOUNT<=1:
                fileID = 1
            else:
                fileID = randrange(1, self.FILES_AMOUNT)
            
            imgID = randrange(0, self.IMAGES_PER_FILE-1)
            
            imgObj = self.loadOneImage(imgID, fileID)
            imagesData.append(imgObj["image"])
            labels.append(imgObj["label"])
            flipped.append(imgObj["flipped"])

        return [imagesData, labels, flipped]

    def getTestBatch(self, amount):
        imagesData = []
        labels = []

        for i in range(amount):
            imgID = randrange(0, amount)
            
            imgObj = self.loadOneTestImage(imgID)
            imagesData.append(imgObj["image"])
            labels.append(imgObj["label"])

        return [imagesData, labels]

    def getImageData(self, index, fileID=0):
        return self.data[fileID]["data"][index]

    def loadOneTestImage(self, index):
        _image = []
        if self.format=="Grayscale":
            for byte in range(self.IMG_SIZE):
                pixel = (self.testData["data"][index][byte]+self.testData["data"][index][self.IMG_SIZE+byte]+self.testData["data"][index][self.IMG_SIZE*2+byte])/3
                _image.append(pixel/255)
        else:
             for byte in range(self.IMG_SIZE*3):
                pixel = self.testData["data"][index][byte]
                _image.append(pixel/255)

        _label = [0]*10
        _label[self.testData["labels"][index]] = 1
        
        return {"image":_image, "label":_label}

    def loadOneImage(self, index, fileID=0):
        _image = []
    
        flipped=rand.choice([0, 2]);
        staturation = rand.uniform(-10.0, 10.0)

        if self.format=="Grayscale":
            for byte in range(self.IMG_SIZE):
                if flipped:
                   pixel = (self.data[fileID]["data"][index][self.IMG_SIZE-byte-1]+self.data[fileID]["data"][index][self.IMG_SIZE+byte]+self.data[fileID]["data"][index][self.IMG_SIZE*3-byte-1])/3
                else:
                   pixel = (self.data[fileID]["data"][index][byte]+self.data[fileID]["data"][index][self.IMG_SIZE+byte]+self.data[fileID]["data"][index][self.IMG_SIZE*2+byte])/3
                _image.append(max(0, min(255, pixel+staturation))/255)
        else:
           for byte in range(self.IMG_SIZE*3):
                if flipped==1:
                   pixel = (self.data[fileID]["data"][index][self.IMG_SIZE-byte-1])
                elif flipped==2:
                   color_channel = int(byte/(32*32))*32*32
                   local = byte-color_channel
                   x = 32-local%32-1
                   y = 32-int(local/32)
                   pixel = (self.data[fileID]["data"][index][color_channel+local])
                else:
                   pixel = self.data[fileID]["data"][index][byte]
                _image.append((pixel+staturation)/255)

        _label = [0]*10
        _label[self.data[fileID]["labels"][index]] = 1
        
        return {"image":_image, "label":_label, "flipped": flipped}

    def getImageBytes(self):
        if self.format=="Grayscale":
            return self.IMG_SIZE
        else:
            return self.IMG_SIZE*3
        
    def getImageWidth(self):
        if self.format=="Grayscale":
            return int(np.sqrt(self.IMG_SIZE))
        else:
            return int(np.sqrt(self.IMG_SIZE*3))

    def getImagesCount(self):
        return self.IMAGES_PER_FILE*self.FILES_AMOUNT

    def getTrainedCount(self):
        count = 0
        for i in range(self.FILES_AMOUNT):
            count+= self.data[i]["training_state"]
        return count
        
    def rangeFactor(t, point, _range):
        ratio = np.abs (point - t) / _range;
        if ratio < 1:
            return 1 - ratio;
        else:
            return 0;

class MinstLoader:
    def __init__(self, path="inputs/datasets/MNIST_data/"):
        self.path = path
        self.LABELS_COUNT = 10
        self.loadAllImages()
        
    def loadAllImages(self):
        self.mnist = input_data.read_data_sets(self.path, one_hot=True)
        
    def loadOneImage(self, index):
        batch_x, batch_y = self.mnist.train.next_batch(1)
        return {"image":batch_x[0], "labels":batch_y[0]}
        
    def getImageBytes(self):
            return 28*28
    def getImageWidth(self):
        return 28
        
    def getNextBatch(self, batch_size):
        batch_x, batch_y = self.mnist.train.next_batch(batch_size)
        return [batch_x, batch_y]

    def getTestBatch(self, batch_size):
        return [self.mnist.test.images[:batch_size], [self.mnist.test.labels[:batch_size]]]