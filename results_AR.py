import numpy as np
import matplotlib.pyplot as plt
import pickle 
from statsmodels.tsa.ar_model import AutoReg
from scipy.io import wavfile


with open('time_series\data.pkl', 'rb') as fichier:
    data = pickle.load(fichier)

sample_rate = data[0][0]
context_length = 0.1                                 #### Durée en seconde de la fenêtre de contexte
predict_length = 0.025                               #### Durée en seconde de la fenêtre de prédiction
context_size = int(context_length * sample_rate)     #### Taille en nombre de points de la fenêtre de contexte
predict_size = int(predict_length * sample_rate)     #### Taille en nombre de points de la fenêtre de prédiction
crossfade_size = predict_size // 4            #### Taille en nombre de points de la fenêtre de crossfade


song_1 = data[0][1][2000:200000]               #### On prend un morceau de la série temporelle
song_predicted = song_1.copy()
song_1_empty = song_1.copy()
song_predicted_crossfaded = song_1.copy()

pos_loss_packets = np.random.randint(context_size, len(song_1) - predict_size, 20)  #### On choisit 10 positions de paquets perdus au hasard
pos_loss_packets = np.sort(pos_loss_packets)

crossfade = np.sqrt(np.linspace(1,0,crossfade_size))   #### Fenêtre de crossfade

###### On remplace les paquets perdus ######

for pos in pos_loss_packets :

    print('remplissage de la position : ', pos)

    # split dataset
    context, lost_packet, crossfade_window = song_1[pos - context_size: pos], song_1[pos: pos + predict_size], song_1[pos + predict_size: pos + predict_size + crossfade_size]

    # train autoregression
    model = AutoReg(context, lags=300)
    model_fit = model.fit()

    # make predictions
    predictions = model_fit.predict(start=context_size, end=context_size + predict_size + crossfade_size - 1, dynamic=False)

    # complete song        
    song_1_empty[pos: pos + predict_size] = np.zeros(predict_size)
    song_predicted[pos: pos + predict_size] = predictions[:predict_size]

    song_predicted_crossfaded[pos: pos + predict_size] = predictions[:predict_size]
    song_predicted_crossfaded[pos + predict_size: pos + predict_size + crossfade_size] = (crossfade * predictions[predict_size:] + (1-crossfade) * crossfade_window)

song_1 = np.array(song_1)

song_1_empty = np.array(song_1_empty)
song_1_empty = song_1_empty.astype(np.int16)

song_predicted = np.array(song_predicted)
song_predicted = song_predicted.astype(np.int16)

song_predicted_crossfaded = np.array(song_predicted_crossfaded)
song_predicted_crossfaded = song_predicted_crossfaded.astype(np.int16)

fig, axs = plt.subplots(4)

axs[0].plot(song_1)
axs[0].set_title('song_1')

axs[1].plot(song_1_empty)
axs[1].set_title('song_1_empty')

axs[2].plot(song_predicted)
axs[2].set_title('song_predicted')

axs[3].plot(song_predicted_crossfaded)
axs[3].set_title('song_predicted_crossfaded')

plt.tight_layout()

plt.show()

# Enregistrer la liste en tant que fichier audio WAV
wavfile.write('predictions_AR\song_1.wav', sample_rate, song_1)
wavfile.write('predictions_AR\song_1_empty.wav', sample_rate, song_1_empty)
wavfile.write('predictions_AR\song_predicted.wav', sample_rate, song_predicted)
wavfile.write('predictions_AR\song_1_predicted_crossfaded.wav', sample_rate, song_predicted_crossfaded)