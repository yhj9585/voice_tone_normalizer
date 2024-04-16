# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 23:22:00 2024

@author: yhj62
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


####################################################################
## TEST DATA ###



# 동일 사람 : 동일 폴더에 저장되어 있음
####################################################################

output_folder = "output_test/"
ult_src = "output_test/pickle/"

all_list = os.listdir(ult_src)
print(all_list)

mfcc_sound = []

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

    
print(np.shape(mfcc_sound[0]), np.shape(mfcc_sound[1]))

#a = np.concatenate(mfcc_sound)
sample_rate = 16000
n_fft = 400
hop_length = 160


y = librosa.feature.inverse.mfcc_to_audio(mfcc=mfcc_sound[0][0], 
                                          sr=sample_rate,
                                          hop_length=hop_length,
                                          n_fft=n_fft)

y2 = librosa.feature.inverse.db_to_power(y,ref=np.max)

y3 = librosa.feature.inverse.mel_to_audio(y2,
                                          sr=sample_rate,
                                          n_mels=128)


output_folder = "output_test/"
output = os.path.join(output_folder, f'test11.wav')



y2.export(output, format='wav')
    




