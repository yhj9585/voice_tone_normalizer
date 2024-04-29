# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:01:45 2024

@author: H.J. Yoon

PICKLEd DATA load + K-means Clustering
"""

import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import subprocess

import pickle

from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture


####################################################################
## TEST DATA ###

#GMM


# 동일 사람 : 동일 폴더에 저장되어 있음
####################################################################

output_folder = "output_test/"
ult_src = "output_test/pickle/"

all_list = os.listdir(ult_src)
print(all_list)

mfcc_sound = []

numofData = 0

testData = np.empty((0,20,200))

for i, file in enumerate(all_list) :  # 폴더 전체에서 사람 찾기
    if not file.endswith("pickle") : continue
    
    pickle_dir = os.path.join(ult_src, file)
    print(pickle_dir)
    
    with open(pickle_dir, 'rb') as file:
        temp_data = pickle.load(file)
    
    print(len(temp_data))

    
    numpy_temp = np.array(temp_data)
    #print(numpy_temp)
    mfcc_sound.append(numpy_temp)
    testData = np.append(testData, mfcc_sound[numofData], axis=0)
    
    numofData += 1

print(type(mfcc_sound[0]))

print(np.shape(mfcc_sound[0]), np.shape(mfcc_sound[1]))

#testData = np.append(testData, mfcc_sound[0], axis=0)
#testData = np.append(testData, mfcc_sound[1], axis=0)
print(type(testData))
print(np.shape(testData))

#a = np.concatenate(mfcc_sound)

#a = a.reshape(-1,20*200)

#print(a.shape)

old_shape = testData.shape

testData = testData.reshape(-1,20*200)



gmm = GaussianMixture(n_components=numofData, random_state=0, covariance_type='full')
gmm.fit(testData)
clusters = gmm.predict(testData)
#clusters = clusters.reshape(old_shape[0],old_shape[1])
print(clusters)
print(len(clusters))
#print(gmm.labels_)



'''
km = KMeans(algorithm='auto',
            n_clusters=2,
            n_init='auto',
            random_state=30,
            tol=0.0001,
            verbose=0)
km.fit(a)


print(km.labels_)
print(len(km.labels_))

'''






















