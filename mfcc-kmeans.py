# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:47:49 2024

@author: yhj62
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


file = "test2.flac"
file2 = "short1.wav"

file3 = "train_data_01/003/114/114_003_0004.flac"
# 동일 사람 : 동일 폴더에 저장되어 있음

ult_src = "train_data_01/003/"




# load audio file with Librosa
signal, sample_rate = librosa.load(file, sr=22050)


sound = AudioSegment.from_file(file3, format='flac')
sound = sound.set_frame_rate(44100).set_channels(1).set_sample_width(2)


audio_chunks = split_on_silence(sound,
    min_silence_len=500, # 최소 무음 길이 (밀리초 단위)
    silence_thresh=-35, # 무음으로 간주되는 dBFS 값
    keep_silence=100 # 분리된 오디오 조각들 간의 추가적인 무음 길이 (밀리초 단위)
)


output_folder = "output_test/"

testchunk = AudioSegment.empty()

#testchunk;

for i, chunks in enumerate(audio_chunks) :     
    output_file = os.path.join(output_folder, f'chunk_{i}.wav')
    chunks.export(output_file, format='wav')
    testchunk += chunks
    
    print(chunks)

output = os.path.join(output_folder, f'test.wav')
testchunk.export(output, format='wav')

