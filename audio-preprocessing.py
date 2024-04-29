# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:47:49 2024

@author: H.J. Yoon

Pydub Load + Librosa Mel + MFCC >> PICKLE DATA 

git commit - merge - git pull.

"""

import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import subprocess

import pickle
import soundfile as sf

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

# 임시로 몇 개 골라서 TEST
#folder_list = ['106','131','175']

sample_rate = 16000
n_fft = 400
hop_length = 160

pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))


def audiosegment_to_ndarray(audiosegment):
    samples = audiosegment.get_array_of_samples()
    samples_float = librosa.util.buf_to_float(samples,n_bytes=2,
                                      dtype=np.float32)
    if audiosegment.channels==2:
        sample_left= np.copy(samples_float[::2])
        sample_right= np.copy(samples_float[1::2])
        sample_all = np.array([sample_left,sample_right])
    else:
        sample_all = samples_float
        
    return [sample_all,audiosegment.frame_rate]
    

def pydubTolibrosa(audioBypeople, folder, s) :
    
    '''
    sample = audioBypeople.get_array_of_samples()
    arr = np.array(sample).astype(np.float32)       # TO librosa
    y, index = librosa.effects.trim(arr)
    '''
    
    y, index = audiosegment_to_ndarray(audioBypeople)
    
    
    print(index)
    
    # Mel spectrogram
    meled = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128, hop_length=160, n_fft=400)
    meled_long = librosa.power_to_db(meled, ref=np.max)
    padded_meled = pad2d(meled_long, 200)
    
    # Output Visualization
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(meled, y_axis = 'mel', sr = 16000, hop_length = 16000, x_axis = 'time')
    plt.colorbar(format = '%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()
    
    
    # MFCC
    mfcc_sound = librosa.feature.mfcc(S=y, n_mfcc=20, n_fft=400, hop_length=160)
    padded_mfcc = pad2d(mfcc_sound, 200)
    
    '''
    y3 = librosa.feature.inverse.mfcc_to_audio(mfcc=padded_mfcc,
                                              n_mels=128,
                                              sr=sample_rate,
                                              n_fft=400
                                              )
    '''
    
    '''
    y1 = librosa.feature.inverse.mfcc_to_mel(mfcc=mfcc_sound,
                                            n_mels=128)
                                            
    y2 = librosa.db_to_power(y1, ref=1.0)'''
    
    y3 = librosa.feature.inverse.mel_to_audio(meled, sr=16000, n_fft=400, hop_length=160)

    output_folder1 = "output_test/"
    output1 = os.path.join(output_folder1, f'test1.wav')

    sf.write(output1, y3, sample_rate, 'PCM_24')
    
    
    
    # 100*200 DATASET
    #print(mfcc_sound.shape)
    #print("MFCC SHAPE : ",padded_mfcc.shape)
    #print(padded_mfcc)
    
    return padded_mfcc



def loadSoundperPerson(folder, subFolder, audioBypeople) :
    mfcc_data = []
    
    audioBypeople = AudioSegment.empty()
    print(subFolder)
    
    sublist = os.listdir(subFolder)

    for s, nowfile in enumerate(sublist) : # 사람 폴더에서 파일 불러오기
        if nowfile.endswith('txt') :
            continue
        
        if s > 3 : break
        
        thisaudio = os.path.join(subFolder, nowfile)
        print(thisaudio)
        
        sound = AudioSegment.from_file(thisaudio, format='flac')
        sound = sound.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        #plt.plot(sound.get_array_of_samples())
        
        audio_chunks = split_on_silence(sound,
            min_silence_len = 500,    # 최소 무음 길이 (밀리초 단위)
            silence_thresh=-50,       # 무음으로 간주되는 dBFS 값
            keep_silence = 50         # 분리된 오디오 조각들 간의 추가적인 무음 길이 (밀리초 단위)
        )                             # 일정 이하의 소리는 자동으로 노이즈로 분류되어 제거
        
        print(len(audio_chunks))
        
        for c, chunks in enumerate(audio_chunks) :
            audioBypeople += chunks
            #plt.plot(audioBypeople.get_array_of_samples())
            print(chunks)
            
        mfcc_data.append(pydubTolibrosa(audioBypeople, folder, s))
        audioBypeople = AudioSegment.empty()

    print(mfcc_data)
    print(len(mfcc_data))
    
    pickle_file = os.path.join(output_folder, f'pickle/{folder}_{0}.pickle')
    pickle_folder = os.path.join(output_folder, f'pickle')
    
    if not os.path.exists(pickle_folder) :
        os.makedirs(pickle_folder)
    
    with open(pickle_file, 'wb') as file :
        pickle.dump(mfcc_data, file)
    
    return 


all_list = os.listdir(ult_src)
print(all_list)
#folder_list = [x for x in all_list if os.path.isdir(x)]


audioBypeople = AudioSegment.empty()

for i, folder in enumerate(all_list) :  # 폴더 전체에서 사람 찾기

    if (i > 2) : break
    nowdir = os.path.join(ult_src, folder)
    print(nowdir)
    loadSoundperPerson(folder, nowdir, audioBypeople)
    
    output = os.path.join(output_folder, f'test.wav')
    audioBypeople.export(output, format='wav')
    

