import matplotlib.pyplot as plt  # always import matplotlib before librosa -> librosa messes up with matplotlib loading
import librosa, librosa.display
import numpy as np

file = "blues.00000.wav"
FIG_SIZE = (15,10)

# waveform 
signal, sr = librosa.load(file, sr=22050) # sr = sr * T (duration of the sound) - > 22050 * 30
# plt.figure(figsize=(12,4))
# librosa.display.waveshow(signal, sr=sr)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()

# perform fft to get spectrum
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0,sr,len(magnitude))

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(frequency)/2)]


# plt.plot(left_frequency,left_magnitude)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()

# perform stft to get spectogram
n_fft = 2048
hop_length = 512

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectogram = np.abs(stft)

# we perceive loudness in a logarithmic way
# so we need to convert the spectogram to a log scale i.e convert them to a decibel scale
log_spectogram = librosa.amplitude_to_db(spectogram)

#plt.figure(figsize=FIG_SIZE)
#fig,ax = plt.subplots()
#img = 
#fig.colorbar(img,ax=ax)
#librosa.display.specshow(log_spectogram, sr=sr, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar() # spectogram is amplitude as a function of time and frequency, this is expressed as a color bar
# plt.show()

# Calculate MFCCs 
MFCCs = librosa.feature.mfcc(y=signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar() # spectogram is amplitude as a function of time and frequency, this is expressed as a color bar
plt.show()






