import numpy as np
import pickle 
from Autoreg import Autoreg
import matplotlib.pyplot as plt

from scipy.io import wavfile


with open('time_series\data.pkl', 'rb') as fichier:
    data = pickle.load(fichier)

sample_rate = data[0][0]
context = 0.05
predict = 0.005
context_size = int(context * sample_rate)
predict_size = int(predict * sample_rate)

song_1 = data[0][1][2000:4000]

AR = Autoreg(song_1, 400)
AR.fit()
AR.plot(100)
plt.plot(song_1)
plt.show()


