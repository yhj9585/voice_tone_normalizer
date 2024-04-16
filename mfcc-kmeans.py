# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:47:49 2024

@author: H.J. Yoon
"""

import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import subprocess

# ffmpeg 설치 필요
#AudioSegment.ffmpeg = "C:/Users/yhj62/scoop/apps/ffmpeg/7.0"

####################################################################
## TEST DATA ###
file = "test2.flac"
file2 = "short1.wav"

file3 = "train_data_01/003/114/114_003_0004.flac"
# 동일 사람 : 동일 폴더에 저장되어 있음
######################################################################

output_folder = "output_test/"
ult_src = "train_data_01/003/"

def loadSoundperPerson(subFolder, audioBypeople) :
    audioBypeople = AudioSegment.empty()
    print(subFolder)
    
    sublist = os.listdir(subFolder)

    for s, nowfile in enumerate(sublist) : # 사람 폴더에서 파일 불러오기
        if nowfile.endswith('txt') :
            continue
        
        if (s == 10) :return audioBypeople
    
        thisaudio = os.path.join(subFolder, nowfile)
        print(thisaudio)
        
        sound = AudioSegment.from_file(thisaudio, format='flac')
        sound = sound.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        #plt.plot(sound.get_array_of_samples())
        
        audio_chunks = split_on_silence(sound,
            min_silence_len = 500,    # 최소 무음 길이 (밀리초 단위)
            silence_thresh=-50,     # 무음으로 간주되는 dBFS 값
            keep_silence = 50        # 분리된 오디오 조각들 간의 추가적인 무음 길이 (밀리초 단위)
        )                           # 일정 이하의 소리는 자동으로 노이즈로 분류되어 제거
        
        print(len(audio_chunks))
        
        for c, chunks in enumerate(audio_chunks) :
            audioBypeople += chunks
            plt.plot(audioBypeople.get_array_of_samples())
            print(chunks)

    return audioBypeople


all_list = os.listdir(ult_src)
print(all_list)
#folder_list = [x for x in list if os.path.isdir(x)]
folder_list = ['106']

audioBypeople = AudioSegment.empty()


for i, folder in enumerate(folder_list) :  # 폴더 전체에서 사람 찾기
    nowdir = os.path.join(ult_src, folder)
    print(nowdir)
    audioBypeople = loadSoundperPerson(nowdir, audioBypeople)
    
    
    output = os.path.join(output_folder, f'test.wav')
    audioBypeople.export(output, format='wav')
    





# load audio file with Librosa
#signal, sample_rate = librosa.load(file, sr=22050)


# 노이즈 제거


#testchunk = AudioSegment.empty()

'''
for i, chunks in enumerate(audio_chunks) :     
    output_file = os.path.join(output_folder, f'chunk_{i}.wav')
    chunks.export(output_file, format='wav')
    testchunk += chunks
    
    print(chunks)




'''