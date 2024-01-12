import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle


directory = "dataset"

files = os.listdir(directory)
files_path = []

for file in files:
    files_path.append(os.path.join(directory, file))

data = []

def is_mono(data):
    if (len(data[1].shape) == 2):
        return True
    else:
        return False

for i in range (len(files_path)):
    sample_rate, time_series = wav.read(files_path[i])
    if (is_mono(time_series)):
        data.append([sample_rate, time_series])
    else:
        data.append([sample_rate, np.mean(np.array(time_series), axis=1).tolist()])

with open('time_series\data.pkl', 'wb') as fichier:
    pickle.dump(data, fichier)
    

