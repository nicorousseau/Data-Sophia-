import numpy as np
import matplotlib.pyplot as plt 
#from scipy.io import wavfile
#from scipy.signal import spectrogram
#from scipy.signal import resample
#from librosa.display import specshow
import statsmodels.tsa.ar_model #sinon fonctionne pas en utilisant predict.py
import os
import benchmark
import predict
#from AR_freq import AR_freq

new_sample_rate = 32000
directory = r"C:\Users\hippo\OneDrive\Bureau\MINES\2A\T2_Data_Sophia\PROJET\Bach10_v1.1"
instr_wanted = "saxphone.wav"
temps_paquet = 0.02#0 #20ms
taille_paquet = int(temps_paquet *new_sample_rate) #nb_echantillons par paquet
list_audio_sax, list_pos_gap, conc_audio, pos_gap = predict.load_data(directory, taille_paquet, n_loss_per_audio= 110, instr = instr_wanted, resample_rate = new_sample_rate, normalise=True)
#audio_1 = list_audio_sax[0]
print(f"{len(pos_gap)}, {len(conc_audio)}, {len(conc_audio)/new_sample_rate}")
conc_audio = np.array(conc_audio)
print(np.max(conc_audio), np.min(conc_audio))
print(conc_audio)
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
print("-------- 0 Filling and Persistance : Done ")

#PERSISTANCE FREQUENTIELLE
freq_persistance = predict.freq_persistance(conc_audio, pos_gap, taille_paquet, sample_rate=new_sample_rate)
nmse_freq_pers_m, nmse_freq_pers = benchmark.nmse_mean(conc_audio, freq_persistance, pos_gap, taille_paquet)
mel_freq_pers = benchmark.mel_cs(conc_audio, freq_persistance, new_sample_rate)
nmses.append(nmse_freq_pers)
nmses_m.append(nmse_freq_pers_m)
mels_cs.append(mel_freq_pers)
labels.append("Persistance Freq")
print("-------- Persistance Fréquentielle : Done ")

def ar_process(order, adapt = False) : 
    ar_32 = predict.audio_predictions(conc_audio, pos_gap, taille_paquet, order=order, adapt = adapt)
    nmse_AR_32_m, nmse_AR_32 = benchmark.nmse_mean(conc_audio, ar_32, pos_gap, taille_paquet)
    mel_cs_AR_32 = benchmark.mel_cs(conc_audio, ar_32, new_sample_rate)
    nmses.append(nmse_AR_32)
    nmses_m.append(nmse_AR_32_m)
    mels_cs.append(mel_cs_AR_32)
    labels.append(f"AR {order} {adapt}")
    print(f"-------- AR {order} adapt : {adapt} : Done ")
    return ar_32

def process_env(audio_norm, env_max, env_min, order) : 
    env_audio, env_max_int, env_min_int = predict.env_predictions(audio_norm, env_max, env_min, pos_gap, taille_paquet, order=order, train_size = 3* taille_paquet, adapt = True, ar_on_env = False)
    nmse_env_m, nmse_env_AR = benchmark.nmse_mean(conc_audio, env_audio, pos_gap, taille_paquet)
    mel_cs_env = benchmark.mel_cs(conc_audio, env_audio, new_sample_rate)
    nmses.append(nmse_env_AR)
    nmses_m.append(nmse_env_m)
    mels_cs.append(mel_cs_env)
    labels.append(f"Env {order}")
    print(f"-------- Enveloppe Processing  {order}: Done ")
    return env_audio
    
env_max, env_min = predict._compute_env(conc_audio)
audio_norm = predict._compute_without_env_audio(conc_audio, env_max, env_min)

## 16
#ar16 = ar_process(16, adapt = False)
##ar_process(16, adapt = True)
#env16 = process_env(audio_norm, env_max, env_min, 16)

# 32 
ar32 = ar_process(32, adapt = False)
#ar_process(32, adapt = True)
env32 = process_env(audio_norm, env_max, env_min, 32)

#64
ar64 = ar_process(64, adapt = False)
ar_adapt64 = ar_process(64, adapt = True)
env64 = process_env(audio_norm, env_max, env_min, 64)

#128
ar128 = ar_process(128, adapt = False)
ar_adapt128 = ar_process(128, adapt = True)
env128 = process_env(audio_norm, env_max, env_min, 128)
#
#256
ar256 = ar_process(256, adapt = False)
ar_adapt256 = ar_process(256, adapt = True)
env256 = process_env(audio_norm, env_max, env_min, 256)

##AR 64
#ar_64 = predict.audio_predictions(conc_audio, pos_gap, taille_paquet, order=64, adapt = False)
#nmse_AR_64_m, nmse_AR_64 = benchmark.nmse_mean(conc_audio, ar_64, pos_gap, taille_paquet)
#mel_cs_AR_64 = benchmark.mel_cs(conc_audio, ar_64, new_sample_rate)
#nmses.append(nmse_AR_64)
#nmses_m.append(nmse_AR_64_m)
#mels_cs.append(mel_cs_AR_64)
#labels.append("AR 64")
#print("-------- AR 64 : Done ")

##AR 128
#ar_128 = predict.audio_predictions(conc_audio, pos_gap, taille_paquet, order=128, adapt = False)
#nmse_AR_128_m, nmse_AR_128 = benchmark.nmse_mean(conc_audio, ar_128, pos_gap, taille_paquet)
#mel_cs_AR_128 = benchmark.mel_cs(conc_audio, ar_128, new_sample_rate)
#nmses.append(nmse_AR_128)
#nmses_m.append(nmse_AR_128_m)
#mels_cs.append(mel_cs_AR_128)
#labels.append("AR 128")
#print("-------- AR 128 : Done ")

##AR 256
#ar_256 = predict.audio_predictions(conc_audio, pos_gap, taille_paquet, order=256, adapt = False)
#nmse_AR_256_m, nmse_AR_256 = benchmark.nmse_mean(conc_audio, ar_256, pos_gap, taille_paquet)
#mel_cs_AR_256 = benchmark.mel_cs(conc_audio, ar_256, new_sample_rate)
#nmses.append(nmse_AR_256)
#nmses_m.append(nmse_AR_256_m)
#mels_cs.append(mel_cs_AR_256)
#labels.append("AR 256")
#print("-------- AR 256 : Done ")

#AR ADAPTATIF
#adapt_ar_128 = predict.audio_predictions(conc_audio, pos_gap, taille_paquet, order=256, adapt = True)
#nmse_our_AR_m, nmse_our_AR = benchmark.nmse_mean(conc_audio, adapt_ar_128, pos_gap, taille_paquet)
#mel_cs_our_AR = benchmark.mel_cs(conc_audio, adapt_ar_128, new_sample_rate)
#nmses.append(nmse_our_AR)
#nmses_m.append(nmse_our_AR_m)
#mels_cs.append(mel_cs_our_AR)
#labels.append("Adaptative AR")
#print("-------- AR Adaptatif 256 : Done ")

#AR ADAPTATIF AVEC ENVELOPPE PROCESSING


#BENCHMARK 
print(f"Pour les méthodes NMSES puis MEL_CS:")
for i in range(len(nmses_m)) :
    print(f"{nmses_m[i]} \n  {mels_cs[i]}")
# Visual 
fig, axes = plt.subplots(nrows = 1, ncols = 2,  figsize = (9,4))
axes[0].boxplot(nmses, labels= labels)
axes[0].set_title("NSME")

axes[1].plot(labels,np.log10(mels_cs))
axes[1].set_title("Mel_CS")
plt.show()

def conv_to_write(audio) : 
    new_audio = audio * np.iinfo(np.int16).max//2
    new_audio = audio.astype(np.int16)
    return new_audio

directory = r"C:\Users\hippo\OneDrive\Bureau\MINES\2A\T2_Data_Sophia\PROJET\Data-Sophia-\produced_audio"
peaq_sr  = 48000
directory = r"C:\Users\hippo\OneDrive\Bureau\MINES\2A\T2_Data_Sophia\PROJET\Data-Sophia-\peaq_test"
instr = "sax"
predict.write_wav(loss, new_sample_rate, f"Silence_filling {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(persistance, new_sample_rate, f"Persistance {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(conc_audio, new_sample_rate, f"original {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(freq_persistance, new_sample_rate, f"Persistance Freq {instr} 5m", directory=directory, new_samplerate=peaq_sr)

#predict.write_wav(ar16, new_sample_rate, f"ar16 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(ar32, new_sample_rate, f"ar32 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(ar64, new_sample_rate, f"ar64 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(ar128, new_sample_rate, f"ar128 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(ar256, new_sample_rate, f"ar256 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(ar_adapt64, new_sample_rate, f"ar64 adapt {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(ar_adapt128, new_sample_rate, f"ar128 adapt {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(ar_adapt256, new_sample_rate, f"ar256 adapt {instr} 5m", directory=directory, new_samplerate=peaq_sr)
#predict.write_wav(env16, new_sample_rate, f"Enveloppe16 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(env32, new_sample_rate, f"Enveloppe32 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(env64, new_sample_rate, f"Enveloppe64 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(env128, new_sample_rate, f"Enveloppe128 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(env256, new_sample_rate, f"Enveloppe256 {instr} 5m", directory=directory, new_samplerate=peaq_sr)

directory = r"C:\Users\hippo\OneDrive\Bureau\MINES\2A\T2_Data_Sophia\PROJET\Data-Sophia-\plcmos_test"
peaq_sr = 16000
predict.write_wav(loss, new_sample_rate, f"Silence_filling {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(persistance, new_sample_rate, f"Persistance {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(conc_audio, new_sample_rate, f"original {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(freq_persistance, new_sample_rate, f"Persistance Freq {instr} 5m", directory=directory, new_samplerate=peaq_sr)

#predict.write_wav(ar16, new_sample_rate, f"ar16 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(ar32, new_sample_rate, f"ar32 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(ar64, new_sample_rate, f"ar64 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(ar128, new_sample_rate, f"ar128 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(ar256, new_sample_rate, f"ar256 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(ar_adapt64, new_sample_rate, f"ar64 adapt {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(ar_adapt128, new_sample_rate, f"ar128 adapt {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(ar_adapt256, new_sample_rate, f"ar256 adapt {instr} 5m", directory=directory, new_samplerate=peaq_sr)
#predict.write_wav(env16, new_sample_rate, f"Enveloppe16 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(env32, new_sample_rate, f"Enveloppe32 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(env64, new_sample_rate, f"Enveloppe64 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(env128, new_sample_rate, f"Enveloppe128 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
predict.write_wav(env256, new_sample_rate, f"Enveloppe256 {instr} 5m", directory=directory, new_samplerate=peaq_sr)

