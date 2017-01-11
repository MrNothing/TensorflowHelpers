# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 20:05:49 2017

@author: Boris Musarais
"""
import pydub
import requests
import random as rand
from random import *
from urllib import request
from pathlib import Path
from helpers.extractor import SoundConverter
import numpy as np
import pickle
import os

deezer_api_url = "http://api.deezer.com/"

class DeezerLoader:
    def __init__(self, LABELS_COUNT=1000, fixed_size=-1, sample_size=1/15, local=False, limit_genres=[], other_genres_rate=0.5):
        self.LABELS_COUNT = LABELS_COUNT
        self.sample_size = sample_size
        self.fixed_size = fixed_size
        self.picker = SoundPicker(limit_genres=limit_genres, other_genres_rate=other_genres_rate)
        self.limit_genres = limit_genres
        self.other_genres_rate = other_genres_rate
        self.shuffle_rate = 12
        
        if len(limit_genres)!=0:
            self.LABELS_COUNT = len(limit_genres)
            if other_genres_rate!=0:
                self.LABELS_COUNT += 1
        
        self.local = local

        track = self.picker.getRandomTrack(local=local)
        converter = SoundConverter(track["track"])
        if fixed_size<=0:
            self.image_width = int(np.sqrt(len(converter.ExtractRandomSample(sample_size=sample_size, multiplier=256))))
        else:
            self.image_width = int(np.sqrt(fixed_size))
            
    def getNextBatch(self, batch_size):
        images = []
        labels = []
        
        track = None
        converters = {}
        converter = None        
        
        match = 0
        
        for i in range(batch_size):
            if i%self.shuffle_rate==0 or i==0:
                track = self.picker.getRandomTrack(local=self.local)
        
                if converters.__contains__(track["id"]):
                    converter = converters[track["id"]]
                else:
                    converter = SoundConverter(track["track"])
                    converters[track["id"]] = converter     
                #print("loaded track:"+str(track["id"])+" label: "+str(track["genre"]))
        
            image = converter.ExtractRandomSample(fixed_size=self.fixed_size, sample_size=self.sample_size, multiplier=256, limiter=self.image_width*self.image_width)
            
            if len(image)!=self.image_width*self.image_width:
                raise Exception("sample length was: "+str(len(image))+" expected: "+str(self.image_width*self.image_width))
            
            label = [0]*self.LABELS_COUNT
            if len(self.limit_genres)==0:
                label[track["genre"]] = 1
            else:
                if self.limit_genres.__contains__(track["genre"]):
                    label[self.limit_genres.index(track["genre"])] = 1
                    match+=1
                else:
                    label[len(self.limit_genres)] = 1
                
            images.append(image)
            labels.append(label)
        
        print("extracted "+str(len(images))+" samples from "+str(batch_size/self.shuffle_rate)+" tracks (matches: "+str(match)+")")
        
        return [images, labels]

    def getTestBatch(self, batch_size):
        return self.getNextBatch(batch_size)
        
    def getImageBytes(self):
            return self.image_width*self.image_width
    def getImageWidth(self):
        return self.image_width 

class SoundPicker:
    def __init__(self, other_genres_rate=1, limit_genres=[], ffmpeg_url=r"C:\\path\\ffmpeg\\bin\\ffmpeg.exe", save_folder="inputs/sounds", cache_file="temp/SoundPicker_tracks.pkl", cache_size=500):
        pydub.AudioSegment.converter = ffmpeg_url
        self.tracks = {}
        self.tracksByGenre = {}
        self.save_folder = save_folder
        self.cache_file = cache_file
        self.cache_size = cache_size
        self.limit_genres = limit_genres
        self.other_genres_rate = other_genres_rate
        self.load_tracks()
        
    def save_tracks(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump([self.tracks, self.tracksByGenre], f, pickle.HIGHEST_PROTOCOL)

    def load_tracks(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)
                if len(data)==2:
                    self.tracks = data[0]
                    self.tracksByGenre = data[1]
                else:
                    self.tracks = data
    
    def getNextBatch(samples=100):
        for i in range(samples):
            #todo: extract an mp3 part and add it to the collection.
            tmp = 0
            
    def getRandomGenre(self):
        genre = randrange(1, 400)
        
        if len(self.limit_genres)==0 or randrange(0, 1000)<self.other_genres_rate*1000:
            r = requests.get(deezer_api_url+'genre')
            genres = r.json()["data"]
            
            self.genres_count = len(genres)
    
            genre=genres[randrange(1, len(genres)-1)]
            
            return genre
        else:
            if len(self.limit_genres)>1:
                genre=self.limit_genres[randrange(0, len(self.limit_genres)-1)]
            else:
                genre=self.limit_genres[0]

            r = requests.get(deezer_api_url+'genre/'+str(genre))
            genre_obj = r.json()
                
            return genre_obj
        
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
            request.urlretrieve (track_url, self.save_folder+"/tmp.mp3")
            song = pydub.AudioSegment.from_mp3(self.save_folder+"/tmp.mp3")
            song.export(out_path, format="wav")
            return out_path
    
    def SortAllTracks(self, local=False):
        for trackID in self.tracks:
            track = self.tracks[trackID]
            if self.tracksByGenre.__contains__(track["genre"])==False:
                self.tracksByGenre[track["genre"]] = []
    
            if self.tracksByGenre.__contains__(trackID)==False:
                self.tracksByGenre[track["genre"]].append(trackID)
            
    def getRandomTrack(self, local=False):
        data = {}
        if local:
            if len(self.limit_genres)==0 or randrange(0, 1000)<self.other_genres_rate*1000 or len(self.tracksByGenre)==0:
                track = rand.choice(list(self.tracks.keys()))
                return self.tracks[track]
            else:
                genre_id = rand.choice(self.limit_genres)
                track = rand.choice(self.tracksByGenre[genre_id])
                return self.tracks[track]
        else:
            genre = self.getRandomGenre()
            artist = self.getRandomArtist(str(genre["id"]))
            
            while artist==None:
                 artist = self.getRandomArtist(str(genre["id"]))
            
            track = self.getRandomTrackInfos(str(artist["id"]))
            trackID = track["id"]
            data = {"id":trackID, "genre":genre["id"], "genre_name":genre["name"], "artist":artist["name"], "name":track["title"], "track":self.getTrack(trackID)}
            self.tracks[trackID] = data

            if self.tracksByGenre.__contains__(genre["id"])==False:
                self.tracksByGenre[genre["id"]] = []

            if self.tracksByGenre.__contains__(trackID)==False:
                self.tracksByGenre[genre["id"]].append(trackID)
            
            self.save_tracks()
        return data
