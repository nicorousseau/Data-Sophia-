import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

songs_path = "songs"
directory = "time_series"

files = os.listdir(songs_path)
files_path = []

for file in files:
    files_path.append(os.path.join(songs_path, file))

data = []

def is_mono(song):
    if (len(song[1].shape) == 2):
        return False
    else:
        return True

for file_path in files_path:
    sample_rate, time_series = wav.read(file_path)
    if (is_mono(time_series)):
        data.append([sample_rate, time_series.tolist()])
    else:
        data.append([sample_rate, np.mean(np.array(time_series), axis=1).tolist()])

with open('time_series/data.pkl', 'wb') as fichier:
    pickle.dump(data, fichier)
    

