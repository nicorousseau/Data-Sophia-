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
import time
#from AR_freq import AR_freq
from nico_work import AR_hybride_2 as nico1
from nico_work import persistance_freq as nico2

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

audios_processed = []



def ar_process(order, adapt = False) : 
    start = time.time()
    ar_32 = predict.audio_predictions(conc_audio, pos_gap, taille_paquet, order=order, adapt = adapt)
    end = time.time()
    process_time = format(end-start)
    nmse_AR_32_m, nmse_AR_32 = benchmark.nmse_mean(conc_audio, ar_32, pos_gap, taille_paquet)
    mel_cs_AR_32 = benchmark.mel_cs(conc_audio, ar_32, new_sample_rate)
    nmses.append(nmse_AR_32)
    nmses_m.append(nmse_AR_32_m)
    mels_cs.append(mel_cs_AR_32)
    #labels.append(f"AR {order} {adapt}")
    print(f"-------- AR {order} adapt : {adapt} : Done ")
    print(f"Executed in : {process_time}")
    return ar_32

def process_env(audio_norm, env_max, env_min, order) : 
    start = time.time()
    env_audio, env_max_int, env_min_int = predict.env_predictions(audio_norm, env_max, env_min, pos_gap, taille_paquet, order=order, train_size = 3* taille_paquet, adapt = True, ar_on_env = False)
    end = time.time()
    process_time = format(end-start)
    nmse_env_m, nmse_env_AR = benchmark.nmse_mean(conc_audio, env_audio, pos_gap, taille_paquet)
    mel_cs_env = benchmark.mel_cs(conc_audio, env_audio, new_sample_rate)
    nmses.append(nmse_env_AR)
    nmses_m.append(nmse_env_m)
    mels_cs.append(mel_cs_env)
    #labels.append(f"Env {order}")
    print(f"-------- Enveloppe Processing  {order}: Done ")
    print(f"Executed in : {process_time}")
    return env_audio
    

#Silence et Persistance
start = time.time()
loss, persistance = predict.filling_silence_persistance(conc_audio, pos_gap, taille_paquet)
end = time.time()
time_loss_pers = format(end - start)

nmse_silence_m, nmse_silence = benchmark.nmse_mean(conc_audio, loss, pos_gap, taille_paquet)
nmse_persistance_m, nmse_persistance = benchmark.nmse_mean(conc_audio, persistance, pos_gap, taille_paquet)
mel_cs_silence = benchmark.mel_cs(conc_audio, loss, new_sample_rate)
mel_cs_persistance = benchmark.mel_cs(conc_audio, persistance, new_sample_rate)
audios_processed.append(loss)
audios_processed.append(persistance)
nmses.append(nmse_silence)
nmses_m.append(nmse_silence_m)
mels_cs.append(mel_cs_silence)
labels.append("0 Filling ")
nmses.append(nmse_persistance)
nmses_m.append(nmse_persistance_m)
mels_cs.append(mel_cs_persistance)
labels.append("Persistance")
print("-------- 0 Filling and Persistance : Done ")
print(time_loss_pers)


#PERSISTANCE FREQUENTIELLE
start = time.time()
freq_persistance = predict.freq_persistance(conc_audio, pos_gap, taille_paquet, sample_rate=new_sample_rate)
end = time.time()
time_pers_pers = format(end - start)
nmse_freq_pers_m, nmse_freq_pers = benchmark.nmse_mean(conc_audio, freq_persistance, pos_gap, taille_paquet)
mel_freq_pers = benchmark.mel_cs(conc_audio, freq_persistance, new_sample_rate)
audios_processed.append(freq_persistance)
nmses.append(nmse_freq_pers)
nmses_m.append(nmse_freq_pers_m)
mels_cs.append(mel_freq_pers)
labels.append("Persistance Freq")
print("-------- Persistance Fréquentielle : Done ")
print(time_pers_pers)



# NICO PERSISTANCE FREQUENTIELLE 
nb_freq_kept = 50
nb_lags = 256
train_size = int(1.5*taille_paquet)
predict_size = taille_paquet

Reg_freq = nico2.Persistance_freq(conc_audio, new_sample_rate, new_sample_rate, nb_freq_kept, train_size)
#AR_hybrid.initialise()
regression_freq = conc_audio.copy()
for pos in pos_gap :
    Reg_freq.fit(pos=pos) #chgt la méthode de reg
    Reg_freq.predict(predict_size)
    regression_freq[pos: pos+taille_paquet] = Reg_freq.pred

nmse_reg_freq_m, nmses_reg_freq = benchmark.nmse_mean(conc_audio, regression_freq, pos_gap, taille_paquet)
mel_cs_hy_env = benchmark.mel_cs(conc_audio, regression_freq, new_sample_rate)
nmses.append(nmses_reg_freq)
nmses_m.append(nmse_reg_freq_m)
mels_cs.append(mel_cs_hy_env)
labels.append("Reg Freq")
#64
ar64 = ar_process(64, adapt = False)
audios_processed.append(ar64)
labels.append("ar64")
#ar_adapt64 = ar_process(64, adapt = True)
#env64 = process_env(audio_norm, env_max, env_min, 64)
#audios_processed.append(env64)
#labels.append("env_ar64")


#128
ar128 = ar_process(128, adapt = False)
audios_processed.append(ar128)
labels.append("ar128")
#ar_adapt128 = ar_process(128, adapt = True)
#env128 = process_env(audio_norm, env_max, env_min, 128)
#audios_processed.append(env128)
#labels.append("env128")



#256
ar256 = ar_process(256, adapt = False)
audios_processed.append(ar256)
labels.append("ar256")

aar256 = ar_process(256, adapt = True)
audios_processed.append(aar256)
labels.append("aar256")


env_max, env_min = predict._compute_env(conc_audio)
audio_norm = predict._compute_without_env_audio(conc_audio, env_max, env_min)
env256 = process_env(audio_norm, env_max, env_min, 256)
audios_processed.append(env256)
labels.append("env256")



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


directory = r"C:\Users\hippo\OneDrive\Bureau\MINES\2A\T2_Data_Sophia\PROJET\Data-Sophia-\produced_audio"
peaq_sr  = 48000
directory = r"C:\Users\hippo\OneDrive\Bureau\MINES\2A\T2_Data_Sophia\PROJET\Data-Sophia-\peaq_test"
instr = "sax"
for i in range(len(audios_processed)) :
    audio = audios_processed[i]
    label = labels[i]
    predict.write_wav(audio, new_sample_rate, label, directory = directory, new_samplerate = peaq_sr)
#predict.write_wav(loss, new_sample_rate, f"Silence_filling {instr} 5m", directory=directory, new_samplerate=peaq_sr)
#predict.write_wav(persistance, new_sample_rate, f"Persistance {instr} 5m", directory=directory, new_samplerate=peaq_sr)
#predict.write_wav(conc_audio, new_sample_rate, f"original {instr} 5m", directory=directory, new_samplerate=peaq_sr)
#predict.write_wav(freq_persistance, new_sample_rate, f"Persistance Freq {instr} 5m", directory=directory, new_samplerate=peaq_sr)
#
##predict.write_wav(ar16, new_sample_rate, f"ar16 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
#predict.write_wav(ar64, new_sample_rate, f"ar64 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
#predict.write_wav(ar128, new_sample_rate, f"ar128 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
#predict.write_wav(ar256, new_sample_rate, f"ar256 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
#predict.write_wav(ar_adapt64, new_sample_rate, f"ar64 adapt {instr} 5m", directory=directory, new_samplerate=peaq_sr)
#predict.write_wav(ar_adapt128, new_sample_rate, f"ar128 adapt {instr} 5m", directory=directory, new_samplerate=peaq_sr)
#predict.write_wav(ar_adapt256, new_sample_rate, f"ar256 adapt {instr} 5m", directory=directory, new_samplerate=peaq_sr)
##predict.write_wav(env16, new_sample_rate, f"Enveloppe16 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
#predict.write_wav(env32, new_sample_rate, f"Enveloppe32 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
#predict.write_wav(env64, new_sample_rate, f"Enveloppe64 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
#predict.write_wav(env128, new_sample_rate, f"Enveloppe128 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
#predict.write_wav(env256, new_sample_rate, f"Enveloppe256 {instr} 5m", directory=directory, new_samplerate=peaq_sr)

directory = r"C:\Users\hippo\OneDrive\Bureau\MINES\2A\T2_Data_Sophia\PROJET\Data-Sophia-\plcmos_test"
plcmos_sr = 16000
for i in range(len(audios_processed)) :
    audio = audios_processed[i]
    label = labels[i]
    predict.write_wav(audio, new_sample_rate, label, directory = directory, new_samplerate = plcmos_sr)
    
    
    
##predict.write_wav(loss, new_sample_rate, f"Silence_filling {instr} 5m", directory=directory, new_samplerate=peaq_sr)
##predict.write_wav(persistance, new_sample_rate, f"Persistance {instr} 5m", directory=directory, new_samplerate=peaq_sr)
##predict.write_wav(conc_audio, new_sample_rate, f"original {instr} 5m", directory=directory, new_samplerate=peaq_sr)
##predict.write_wav(freq_persistance, new_sample_rate, f"Persistance Freq {instr} 5m", directory=directory, new_samplerate=peaq_sr)
##
###predict.write_wav(ar16, new_sample_rate, f"ar16 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
##predict.write_wav(ar32, new_sample_rate, f"ar32 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
##predict.write_wav(ar64, new_sample_rate, f"ar64 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
##predict.write_wav(ar128, new_sample_rate, f"ar128 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
##predict.write_wav(ar256, new_sample_rate, f"ar256 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
##predict.write_wav(ar_adapt64, new_sample_rate, f"ar64 adapt {instr} 5m", directory=directory, new_samplerate=peaq_sr)
##predict.write_wav(ar_adapt128, new_sample_rate, f"ar128 adapt {instr} 5m", directory=directory, new_samplerate=peaq_sr)
##predict.write_wav(ar_adapt256, new_sample_rate, f"ar256 adapt {instr} 5m", directory=directory, new_samplerate=peaq_sr)
###predict.write_wav(env16, new_sample_rate, f"Enveloppe16 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
##predict.write_wav(env32, new_sample_rate, f"Enveloppe32 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
##predict.write_wav(env64, new_sample_rate, f"Enveloppe64 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
##predict.write_wav(env128, new_sample_rate, f"Enveloppe128 {instr} 5m", directory=directory, new_samplerate=peaq_sr)
##predict.write_wav(env256, new_sample_rate, f"Enveloppe256 {instr} 5m", directory=directory, new_samplerate=peaq_sr)

