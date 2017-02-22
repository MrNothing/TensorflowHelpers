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
from helpers.sound_tools import SoundConverter

class CSVExtractor:
    def __init__(self, fileName, startLine=0, x=[0], y=[1], separator=';'):
        self.fileName = fileName
        self.startLine = startLine,
        self.x = x
        self.y = y
        
        text_file = open(fileName, "r")
        data = text_file.read()
        
        lines = data.split('\n')
        
        i = 0
        while i<len(lines):
            line = lines[i]
            columns = line.split(separator)
            
            _input = []
            for _x in x:
                _input.append(float(columns[_x]))
            for _y in y:
                _input.append(float(columns[_y]))
            
            i+=1
        
    def getNextBatch(self, batch_size):
        return None

class CifarLoader:

    def __init__(self, path="inputs/datasets/cifar10/", image_format="Grayscale", IMG_SIZE=32*32, LABELS_COUNT=10, IMAGES_PER_FILE=10000, FILES_AMOUNT=5, one_pixel=False):
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
        self.converter = SoundConverter("")
        self.one_pixel=one_pixel

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

    def getNextTimeBatch(self, 
                         amount,  
                         n_steps=32):
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
            imgObj["image"] = self.converter.reshapeAsSequence(imgObj["image"], n_steps)
            
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
        _label = None

        if self.one_pixel:
             if self.format=="Grayscale":
                 raise Exception("Grayscale for generation not implemented")
            
             for byte in range(self.IMG_SIZE*3-1):
                    pixel = self.testData["data"][index][byte]
                    _image.append(pixel/255)
                    
             _label = [0]*256
             _label[self.testData["labels"][index]] = 1
        else:
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
    def __init__(self, path="inputs/datasets/MNIST_data/", one_pixel=False):
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

    def getNextTimeBatch(self, batch_size, n_steps):
        batch_x, batch_y = self.mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, 28, 28))
        return [batch_x, batch_y]
    def getTestBatch(self, batch_size):
        return [self.mnist.test.images[:batch_size], [self.mnist.test.labels[:batch_size]]]

class SignalGenerator:
    def __init__(self, signal_length=128, mode="sin"):
        self.signal_length = signal_length
        self.LABELS_COUNT = 10
        
    def getImageBytes(self):
            return 28*28
    def getImageWidth(self):
        return 28
        
    def getNextBatch(self, batch_size):
        batch_x, batch_y = self.mnist.train.next_batch(batch_size)
        return [batch_x, batch_y]

    def getNextTimeBatch(self, batch_size):
        batch_x, batch_y = self.mnist.train.next_batch(batch_size)
        return [batch_x, batch_y]

    def getTestBatch(self, batch_size):
        return [self.mnist.test.images[:batch_size], [self.mnist.test.labels[:batch_size]]]

    def getTestTimeBatch(self, batch_size):
        return [self.mnist.test.images[:batch_size], [self.mnist.test.labels[:batch_size]]]

class CounterGenerator:
    def __init__(self, signal_length=128):
        self.LABELS_COUNT = 21
        self.ptr = 0
        
        NUM_EXAMPLES = 210000
        train_input = ['{0:020b}'.format(i) for i in range(2**20)]
        rand.shuffle(train_input)
        train_input = [map(int,i) for i in train_input]
        ti  = []
        for i in train_input:
            temp_list = []
            for j in i:
                    temp_list.append([j])
            ti.append(np.array(temp_list))
        train_input = ti
        
        train_output = []
        for i in train_input:
            count = 0
            for j in i:
                if j[0] == 1:
                    count+=1
            temp_list = ([0]*21)
            temp_list[count]=1
            train_output.append(temp_list)
        
        self.test_input = train_input[NUM_EXAMPLES:]
        self.test_output = train_output[NUM_EXAMPLES:]
        self.train_input = train_input[:NUM_EXAMPLES]
        self.train_output = train_output[:NUM_EXAMPLES]
        
        print ("test and training data generated")
        
    def getImageBytes(self):
        return 20
    def getImageWidth(self):
        return np.sqrt(20)
        
    def getNextTimeBatch(self, batch_size, n_steps=-1):
        batch_x, batch_y = self.train_input[self.ptr:self.ptr+batch_size], self.train_output[self.ptr:self.ptr+batch_size]
        self.ptr += batch_size        
        return [batch_x, batch_y]

    def getNextBatch(self, batch_size):
        raise Exception("This dataset only handles LSTM input")

    def getTestBatch(self, batch_size):
        return [self.test_input[0:256], self.test_output[0:256]]

    def getTestTimeBatch(self, batch_size):
        return [self.test_input[0:256], self.test_output[0:256]]