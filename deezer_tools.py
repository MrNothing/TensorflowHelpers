# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 20:05:49 2017

@author: Boris Musarais
"""
import pydub
import requests
from random import *
from urllib import request
from pathlib import Path
from helpers.extractor import SoundConverter
import numpy as np
import pickle
import os

deezer_api_url = "http://api.deezer.com/"

class DeezerLoader:
    def __init__(self, LABELS_COUNT=1000, sample_size=1/15):
        self.LABELS_COUNT = LABELS_COUNT
        self.sample_size = sample_size
        self.picker = SoundPicker()
        self.converters = {}

        track = self.picker.getRandomTrack()
        converter = SoundConverter(track["track"])
        self.converters[track["id"]] = converter
        self.image_width = int(np.sqrt(len(converter.ExtractRandomSample(sample_size=sample_size, multiplier=256))))
        
    def getNextBatch(self, batch_size):
        images = []
        labels = []
        track = self.picker.getRandomTrack()
        
        print("loaded track:"+str(track["id"])+" label: "+str(track["genre"]))
        
        converter = None
        if self.converters.__contains__(track["id"]):
            converter = self.converters[track["id"]]
        else:
            converter = SoundConverter(track["track"])
            self.converters[track["id"]] = converter
        
        for i in range(batch_size):
            
            image = converter.ExtractRandomSample(sample_size=self.sample_size, multiplier=256, limiter=self.image_width*self.image_width)
            
            if len(image)!=self.image_width*self.image_width:
                raise Exception("sample length was: "+str(len(image))+" expected: "+str(self.image_width*self.image_width))
                
            label = [0]*self.LABELS_COUNT
            label[track["genre"]] = 1

            images.append(image)
            labels.append(label)
        
        print("extracted "+str(len(images))+" samples")
        
        return [images, labels]

    def getTestBatch(self, batch_size):
        return self.getNextBatch(batch_size)
        
    def getImageBytes(self):
            return self.image_width*self.image_width
    def getImageWidth(self):
        return self.image_width 

class SoundPicker:
    def __init__(self, ffmpeg_url=r"C:\\path\\ffmpeg\\bin\\ffmpeg.exe", save_folder="inputs/sounds", cache_folder="temp", cache_size=500):
        pydub.AudioSegment.converter = ffmpeg_url
        self.tracks = {}
        self.save_folder = save_folder
        self.cache_folder = cache_folder
        self.cache_size = cache_size
        self.load_tracks()
        
    def save_tracks(self):
        with open(self.cache_folder + '/SoundPicker_tracks.pkl', 'wb') as f:
            pickle.dump(self.tracks, f, pickle.HIGHEST_PROTOCOL)

    def load_tracks(self):
        if os.path.exists(self.cache_folder + '/SoundPicker_tracks.pkl'):
            with open(self.cache_folder + '/SoundPicker_tracks.pkl', 'rb') as f:
                self.tracks = pickle.load(f)
    
    def getNextBatch(samples=100):
        for i in range(samples):
            #todo: extract an mp3 part and add it to the collection.
            tmp = 0
            
    def getRandomGenre(self):
        r = requests.get(deezer_api_url+'genre')
        genres = r.json()["data"]
        
        self.genres_count = len(genres)

        genre=genres[randrange(1, len(genres)-1)]
        
        return genre
        
    def getRandomArtist(self, genre):
        r = requests.get(deezer_api_url+'genre/'+str(genre)+'/artists')
        jsn = r.json()
        
        if jsn.__contains__("data"):
            artists = r.json()["data"]
    
            if len(artists)==0:
                return None
    
            artist=artists[randrange(1, len(artists)-1)]
            
            return artist
        else:
            return None
        
    def getRandomTrackInfos(self, artist):
        r = requests.get(deezer_api_url+'artist/'+str(artist))
        r = requests.get(r.json()["tracklist"])
        tracks = r.json()["data"]
        track=tracks[randrange(1, len(tracks)-1)]
        
        return track
        
    def getTrack(self, trackID):
        out_path = self.save_folder+"/"+str(trackID)+".wav"
        if self.tracks.__contains__(trackID):
            return self.tracks[trackID]["track"]
        else:
            r = requests.get(deezer_api_url+'track/'+str(trackID))
            track_url = r.json()["preview"]
            request.urlretrieve (track_url, self.cache_folder+"/tmp.mp3")
            song = pydub.AudioSegment.from_mp3(self.cache_folder+"/tmp.mp3")
            song.export(out_path, format="wav")
            return out_path
            
    def getRandomTrack(self, local=False):
        genre = self.getRandomGenre()
        artist = self.getRandomArtist(str(genre["id"]))
        
        while artist==None:
             artist = self.getRandomArtist(str(genre["id"]))
        
        track = self.getRandomTrackInfos(str(artist["id"]))
        trackID = track["id"]
        data = {"id":trackID, "genre":genre["id"], "genre_name":genre["name"], "artist":artist["name"], "name":track["title"], "track":self.getTrack(trackID)}
        self.tracks[trackID] = data
        self.save_tracks()
        return data
