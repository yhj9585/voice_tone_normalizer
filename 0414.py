'''
NOT MAIN CODE

DATA Visualization TEST

'''


import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

FIG_SIZE = (15, 10)
file = "127_003_0021.flac"

# load audio file with Librosa
signal, sample_rate = librosa.load(file, sr=22050)
print('signal shape : ', signal.shape)


# WAVEFORM
plt.figure(figsize=FIG_SIZE)
librosa.display.waveshow(signal, sr=sample_rate, color="blue")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")

# perform Fourier Transform
fft = np.fft.fft(signal)
print("fft shape : ", fft.shape)
#--> fft shape :  (661794,)

# Calculate abs values on complex numbers to get magnitude
spectrum = np.abs(fft)
print("spectrum shape : ", spectrum.shape)
#--> spectrum shape :  (661794,)

# Create Frequency Variable
f = np.linspace(0, sample_rate, len(spectrum))
print("f shape : ", f.shape)
#--> f shape :  (661794,)

# take half of the spectrum and frequency
left_spectrum = spectrum[:int(len(spectrum)/2)]
left_f = f[:int(len(spectrum)/2)]
print('left_spectrum shape : ', left_spectrum.shape)
#--> left_spectrum shape :  (330897,)

print('left_f shape : ', left_f.shape)
#--> left_f shape :  (330897,)

# plot specturm
plt.figure(figsize=FIG_SIZE)
plt.plot(left_f, left_spectrum, alpha=0.4)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")


hop_length = 512 # in num. of samples
n_fft = 2048 # window in num. of samples

# Calculate duration hop length and window in seconds 
hop_length_duration = float(hop_length)/sample_rate
n_fft_duration = float(n_fft)/sample_rate


print("STFT hop length duration is : {}s".format(hop_length_duration))
#--> STFT hop length duration is : 0.023219954648526078s

print("STFT window duration is : {}s".format(n_fft_duration))
#--> STFT window duration is : 0.09287981859410431s

# Perform STFT
stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
print("stft shape : ", stft.shape)
#--> stft shape :  (1025, 1293)

# Calculate abs values on complex numbers to get magnitude
spectrogram = np.abs(stft)
log_specto = librosa.amplitude_to_db(spectrogram)
print("spectrogram shape : ", spectrogram.shape)
#--> spectrogram shape :  (1025, 1293)

# display spectrogram
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(log_specto, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.title("Spectrogram")

S = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure().set_figwidth(12)
librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sample_rate, fmax=8000)
plt.colorbar()

print("Mel spectrogram Shape :", S.shape)
print(S)
