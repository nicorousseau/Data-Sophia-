import numpy as np
import matplotlib.pyplot as plt 
#from scipy.io import wavfile
#from scipy.signal import spectrogram
#from scipy.signal import resample
#from librosa.display import specshow
import statsmodels.tsa.ar_model
import os
import benchmark
import predict
#from AR_freq import AR_freq

new_sample_rate = 32000
directory = r"C:\Users\hippo\OneDrive\Bureau\MINES\2A\T2_Data_Sophia\PROJET\Bach10_v1.1"
instr_wanted = "saxphone.wav"
temps_paquet = 0.02#0 #20ms
taille_paquet = int(temps_paquet *new_sample_rate) #nb_echantillons par paquet
list_audio_sax, list_pos_gap, conc_audio, pos_gap = utils.load_data(directory, taille_paquet, instr = instr_wanted, resample_rate = new_sample_rate, normalise=True)
#audio_1 = list_audio_sax[0]
print(pos_gap, len(pos_gap), len(conc_audio))
conc_audio = np.array(conc_audio)
#BENCHMARK PREP
nmses = []
nmses_m = []
mels_cs = []
labels = []

#Silence et Persistance
loss, persistance = predict.filling_silence_persistance(conc_audio, pos_gap, taille_paquet)
nmse_silence_m, nmse_silence = benchmark.nmse_mean(conc_audio, loss, pos_gap, taille_paquet)
nmse_persistance_m, nmse_persistance = benchmark.nmse_mean(conc_audio, persistance, pos_gap, taille_paquet)
mel_cs_silence = benchmark.mel_cs(conc_audio, loss, new_sample_rate)
mel_cs_persistance = benchmark.mel_cs(conc_audio, persistance, new_sample_rate)
nmses.append(nmse_silence)
nmses_m.append(nmse_silence_m)
mels_cs.append(mel_cs_silence)
labels.append("0 Filling ")
nmses.append(nmse_persistance)
nmses_m.append(nmse_persistance_m)
mels_cs.append(mel_cs_persistance)
labels.append("Persistance")

#PERSISTANCE FREQUENTIELLE
freq_persistance = predict.freq_persistance(conc_audio, pos_gap, taille_paquet, sample_rate=new_sample_rate)
nmse_freq_pers_m, nmse_freq_pers = benchmark.nmse_mean(conc_audio, freq_persistance, pos_gap, taille_paquet)
mel_freq_pers = benchmark.mel_cs(conc_audio, freq_persistance, new_sample_rate)
nmses.append(nmse_freq_pers)
nmses_m.append(nmse_freq_pers_m)
mels_cs.append(mel_freq_pers)
labels.append("Persistance Freq")

#AR 32
ar_32 = predict.audio_predictions(conc_audio, pos_gap, taille_paquet, order=32, adapt = False)
nmse_AR_32_m, nmse_AR_32 = benchmark.nmse_mean(conc_audio, ar_32, pos_gap, taille_paquet)
mel_cs_AR_32 = benchmark.mel_cs(conc_audio, ar_32, new_sample_rate)
nmses.append(nmse_AR_32)
nmses_m.append(nmse_AR_32_m)
mels_cs.append(mel_cs_AR_32)
labels.append("AR 32")

#AR 64
ar_64 = predict.audio_predictions(conc_audio, pos_gap, taille_paquet, order=64, adapt = False)
nmse_AR_64_m, nmse_AR_64 = benchmark.nmse_mean(conc_audio, ar_64, pos_gap, taille_paquet)
mel_cs_AR_64 = benchmark.mel_cs(conc_audio, ar_64, new_sample_rate)
nmses.append(nmse_AR_64)
nmses_m.append(nmse_AR_64_m)
mels_cs.append(mel_cs_AR_64)
labels.append("AR 64")

#AR 128
ar_128 = predict.audio_predictions(conc_audio, pos_gap, taille_paquet, order=128, adapt = False)
nmse_AR_128_m, nmse_AR_128 = benchmark.nmse_mean(conc_audio, ar_128, pos_gap, taille_paquet)
mel_cs_AR_128 = benchmark.mel_cs(conc_audio, ar_128, new_sample_rate)
nmses.append(nmse_AR_128)
nmses_m.append(nmse_AR_128_m)
mels_cs.append(mel_cs_AR_128)
labels.append("AR 128")

#AR 256
ar_256 = predict.audio_predictions(conc_audio, pos_gap, taille_paquet, order=256, adapt = False)
nmse_AR_256_m, nmse_AR_256 = benchmark.nmse_mean(conc_audio, ar_256, pos_gap, taille_paquet)
mel_cs_AR_256 = benchmark.mel_cs(conc_audio, ar_256, new_sample_rate)
nmses.append(nmse_AR_256)
nmses_m.append(nmse_AR_256_m)
mels_cs.append(mel_cs_AR_256)
labels.append("AR 256")

#AR ADAPTATIF
adapt_ar_128 = predict.audio_predictions(conc_audio, pos_gap, taille_paquet, order=256, adapt = True)
nmse_our_AR_m, nmse_our_AR = benchmark.nmse_mean(conc_audio, adapt_ar_128, pos_gap, taille_paquet)
mel_cs_our_AR = benchmark.mel_cs(conc_audio, adapt_ar_128, new_sample_rate)
nmses.append(nmse_our_AR)
nmses_m.append(nmse_our_AR_m)
mels_cs.append(mel_cs_our_AR)
labels.append("Adaptative AR")

#AR ADAPTATIF AVEC ENVELOPPE PROCESSING 
env_audio = predict.env_predictions(conc_audio, pos_gap, taille_paquet, order=256, adapt = True)
nmse_env_m, nmse_env_AR = benchmark.nmse_mean(conc_audio, env_audio, pos_gap, taille_paquet)
mel_cs_env = benchmark.mel_cs(conc_audio, env_audio, new_sample_rate)
nmses.append(nmse_env_AR)
nmses_m.append(nmse_env_m)
mels_cs.append(mel_cs_env)
labels.append("Env processing + AR and interpolate")

#BENCHMARK 
for i in range(len(nmses_m)) :
    print(f"Pour la méthode de {labels[i]} : NMSES = {nmses_m[i]} et MEL_CS = {mels_cs[i]}")
# Visual 
fig, axes = plt.subplots(nrows = 1, ncols = 2,  figsize = (9,4))
axes[0,0].boxplot(nmses, labels= labels)
axes[0,0].set_title("NSME")

axes[0,1].plot(labels,np.log10(mels_cs))
axes[0,1].set_title("Mel_CS")
plt.show()

