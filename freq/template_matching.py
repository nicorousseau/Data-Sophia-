import os
import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt

#### Constantes ####
sample_size = 2048
SampleRate = 44100
nb_coeffs = 32

song_directory = r"C:\Users\nicor\Documents\Mines\Data Sophia\Projet\songs"

files = os.listdir(song_directory)    
files_path = []

songs = []

for file in files:
    files_path.append(os.path.join(song_directory, file))

for i in range (len(files_path)) : 
    sr, song = wav.read(files_path[i])
    if len(song.shape) == 2 : 
        song = np.mean(song, axis = 1)
    song = signal.resample(song, int(len(song)/sr*SampleRate))
    songs.append(song)

index = [np.random.randint(0, len(songs[0])-sample_size) for _ in range(1)]

samples = [np.array(songs[0][i:i+sample_size]) for i in index]
samples = np.array(samples)

freq = np.linspace(0, SampleRate//2, sample_size//2)

sample = samples[0][:sample_size//2]
suite = samples[0][sample_size//2:]
matching = []

def ame(x, y):
    return np.mean(np.abs(x-y))

for i in range (len(sample)//10,len(sample)//2) : 
    dot = ame(sample[len(sample)-i:], sample[:i])
    matching.append(dot)

argmin = np.argmin(matching)

plt.plot(matching)
plt.show()

sample_augmented = np.concatenate((sample,np.concatenate((sample[argmin:], np.array([0 for _ in range(argmin)])))))


fig, axs = plt.subplots(2)
axs[0].plot(samples[0])
axs[0].set_title('Sample avec la suite')
axs[1].plot(sample_augmented)
axs[1].set_title('Sample augmenté')
plt.show()