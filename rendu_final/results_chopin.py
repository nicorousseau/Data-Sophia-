import numpy as np
import time 
import os 
import pathlib
from scipy.io import wavfile
#from scipy.signal import resample

import utils
import benchmark
import methods
from scipy.signal import resample
from pandas import DataFrame

new_sample_rate = 32000
temps_paquet = 0.02 #20ms
taille_paquet = int(temps_paquet *new_sample_rate) #nb_echantillons par paquet
n_loss_per_audio = 120

def ar_process(order, adapt = False) : 
    start = time.time()
    ar = utils.audio_predictions(conc_audio, pos_gap, taille_paquet, order=order, crossfade_size=0.1, adapt = adapt)
    end = time.time()
    ar = utils.clip_audio(ar)
    process_time = format(end-start)
    nmse_AR_m, nmse_AR = benchmark.nmse_mean(conc_audio, ar, pos_gap, taille_paquet)
    mel_cs_AR = benchmark.mel_cs(conc_audio, ar, new_sample_rate)
    nmses.append(nmse_AR)
    nmses_m.append(nmse_AR_m)
    mels_cs.append(mel_cs_AR)
    #labels.append(f"AR {order} {adapt}")
    print(f"-------- AR {order} adapt : {adapt} : Done ")
    print(f"Executed in : {process_time}")
    return utils.clip_audio(ar), process_time

def process_env(audio_norm, env_max, env_min, order) : 
    start = time.time()
    env_audio, env_max_int, env_min_int, audio_norm_pred = methods.env_predictions(audio_norm, env_max, env_min, pos_gap, taille_paquet, order=order, train_size = 3* taille_paquet, crossfade_size = 0.1, adapt = True)
    end = time.time()
    env_audio = utils.clip_audio(env_audio)
    process_time = format(end-start)
    nmse_env_m, nmse_env_AR = benchmark.nmse_mean(conc_audio, env_audio, pos_gap, taille_paquet)
    mel_cs_env = benchmark.mel_cs(conc_audio, env_audio, new_sample_rate)
    nmses.append(nmse_env_AR)
    nmses_m.append(nmse_env_m)
    mels_cs.append(mel_cs_env)
    #labels.append(f"Env {order}")
    print(f"-------- Enveloppe Processing  {order}: Done ")
    print(f"Executed in : {process_time}")
    return env_audio, process_time


#audio_path = r"C:\Users\hippo\OneDrive\Bureau\MINES\2A\T2_Data_Sophia\PROJET\Chopin_2.wav"
audio_path = os.path.join(pathlib.Path().resolve(), "Chopin_2.wav")
sample_rate, conc_audio = wavfile.read(audio_path)
conc_audio = utils.data_prep(conc_audio, sample_rate, new_sample_rate = new_sample_rate, to_mono = True) 
pos_gap = utils.los_generation(len(conc_audio), taille_paquet, n_loss = 1000)

#pos_gap = pos_gap[:5]
#conc_audio = conc_audio[:pos_gap[-1]+ 2*taille_paquet]
#BENCHMARK PREP
nmses = []
nmses_m = []
mels_cs = []
labels = []
times = []
audios_processed = []
#Silence et Persistance
start = time.time()
loss, persistance = methods.filling_silence_persistance(conc_audio, pos_gap, taille_paquet)
end = time.time()
time_loss_pers = format(end - start)
times.append(time_loss_pers)
times.append(time_loss_pers)
nmse_silence_m, nmse_silence = benchmark.nmse_mean(conc_audio, loss, pos_gap, taille_paquet)
nmse_persistance_m, nmse_persistance = benchmark.nmse_mean(conc_audio, persistance, pos_gap, taille_paquet)
mel_cs_silence = benchmark.mel_cs(conc_audio, loss, new_sample_rate)
mel_cs_persistance = benchmark.mel_cs(conc_audio, persistance, new_sample_rate)
loss = utils.clip_audio(loss)
presistance = utils.clip_audio(persistance)
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
freq_persistance = methods.freq_persistance(conc_audio, pos_gap, taille_paquet, sample_rate=new_sample_rate, n_harm = 10)
end = time.time()
time_pers_pers = format(end - start)
times.append(time_pers_pers)
nmse_freq_pers_m, nmse_freq_pers = benchmark.nmse_mean(conc_audio, freq_persistance, pos_gap, taille_paquet)
mel_freq_pers = benchmark.mel_cs(conc_audio, freq_persistance, new_sample_rate)
audios_processed.append(freq_persistance)
nmses.append(nmse_freq_pers)
nmses_m.append(nmse_freq_pers_m)
mels_cs.append(mel_freq_pers)
labels.append("Persistance Freq")
print("-------- Persistance Fréquentielle : Done ")
print(time_pers_pers)


# PERSITANCE FREQUENTIELLE par regression linéaire 
nb_freq_kept = 10
nb_lags = 0
train_size = 1*taille_paquet
predict_size = taille_paquet
crossfade_size = int(0.1*predict_size)
crossfade = np.linspace(-0.5,0.5, crossfade_size)
cross_window = 1/(1+np.exp(crossfade))
FREQ_reg = methods.AR_freq(conc_audio, new_sample_rate, new_sample_rate, nb_lags, nb_freq_kept, train_size)
#AR_hybrid.initialise()
freq_reg = conc_audio.copy()
start = time.time()
for pos in pos_gap :
    FREQ_reg.fit(pos=pos)
    FREQ_reg.predict(predict_size + crossfade_size)
    freq_reg[pos: pos+taille_paquet] = FREQ_reg.pred[:predict_size]
    freq_reg[pos+taille_paquet: pos+taille_paquet+crossfade_size] = FREQ_reg.pred[predict_size:]*cross_window + (1-cross_window)*freq_reg[pos+taille_paquet: pos+taille_paquet+crossfade_size]
    
end = time.time()
time_freq_reg = format(end - start)
times.append(time_freq_reg)
freq_reg = utils.clip_audio(freq_reg)
audios_processed.append(freq_reg)
nmse_hy_m, nmses_hy = benchmark.nmse_mean(conc_audio, freq_reg, pos_gap, taille_paquet)
mel_cs_hy = benchmark.mel_cs(conc_audio, freq_reg, new_sample_rate)
nmses.append(nmses_hy)
nmses_m.append(nmse_hy_m)
mels_cs.append(mel_cs_hy)
labels.append("PERS FREQ par REG")
print("-------- PERS FREQ par REG : Done ")
print(time_freq_reg)

###
#entrainement de l'AR sur le résidus
nb_freq_kept = 20
nb_lags = 0
train_size = taille_paquet
predict_size = taille_paquet
audio_data = conc_audio.copy()
REG_freq = methods.AR_freq(audio_data, new_sample_rate, new_sample_rate, nb_lags, nb_freq_kept, train_size)
altered = conc_audio.copy()
predictions = []
start = time.time()
for pos in pos_gap :
    REG_freq.fit(pos=pos)
    REG_freq.predict(predict_size+crossfade_size)
    predictions.append(REG_freq.pred)
    new_train = REG_freq.remove_freq()
    altered[pos-train_size:pos] = new_train
    
    
adapt_ar_128_1 = utils.audio_predictions(altered, pos_gap, taille_paquet+crossfade_size, order=256, train_size = 3*taille_paquet,adapt = True)
strange_hybrid = conc_audio.copy()

for i in range(len(pos_gap)) : 
    pos = pos_gap[i]
    strange_hybrid[pos:pos+predict_size] = predictions[i][:predict_size] + adapt_ar_128_1[pos:pos+predict_size]
    strange_hybrid[pos+predict_size:pos+predict_size+crossfade_size] = cross_window*predictions[i][predict_size:] + (1-cross_window)*adapt_ar_128_1[pos+predict_size:pos+predict_size+crossfade_size]
end = time.time()
time_ar_residus = format(end - start)
times.append(time_ar_residus)
strange_hybrid = utils.clip_audio(strange_hybrid)
audios_processed.append(strange_hybrid)
nmse_strhy_m, nmses_strhy = benchmark.nmse_mean(conc_audio, strange_hybrid, pos_gap, taille_paquet)
mel_cs_strhy = benchmark.mel_cs(conc_audio, strange_hybrid, new_sample_rate)
nmses.append(nmses_strhy)
nmses_m.append(nmse_strhy_m)
mels_cs.append(mel_cs_strhy)
labels.append("AR sur RésidusFreq")
print("-------- AR sur résidus : Done ")
print(time_ar_residus)


#64
ar64, time_64 = ar_process(64, adapt = False)
times.append(time_64)
audios_processed.append(ar64)
labels.append("ar64")



#128
ar128, time_128 = ar_process(128, adapt = False)
times.append(time_128)
audios_processed.append(ar128)
labels.append("ar128")




##256
ar256, time_256= ar_process(256, adapt = False)
times.append(time_256)
audios_processed.append(ar256)
labels.append("ar256")

aar256, times_a256= ar_process(256, adapt = True)
times.append(times_a256)
aar256 = utils.clip_audio(aar256)
audios_processed.append(aar256)
labels.append("aar256")


env_max, env_min = utils.compute_env(conc_audio)
audio_norm = utils.compute_squared_audio(conc_audio, env_max, env_min)
env256, time_e256 = process_env(audio_norm, env_max, env_min, 256)
times.append(time_e256)
env256 = utils.clip_audio(env256)
audios_processed.append(env256)
labels.append("env256")

#METHODE HYBRIDE AVEC REGRESSION 
nb_freq_kept = 20
nb_lags = 256
train_size = 3*taille_paquet
predict_size = taille_paquet

AR_hybrid = methods.AR_freq(conc_audio, new_sample_rate, new_sample_rate, nb_lags, nb_freq_kept, train_size)
#AR_hybrid.initialise()
ar_hybrid = conc_audio.copy()
start = time.time()
for pos in pos_gap :
    AR_hybrid.fit(pos=pos)
    AR_hybrid.predict(predict_size+crossfade_size)
    #ajout du crossfade
    ar_hybrid[pos: pos+taille_paquet] = AR_hybrid.pred[:taille_paquet]
    ar_hybrid[pos+taille_paquet: pos+taille_paquet+crossfade_size] = cross_window*AR_hybrid.pred[taille_paquet:] + (1-cross_window)*ar_hybrid[pos+taille_paquet:pos+taille_paquet+crossfade_size]
end = time.time()
time_hybrid_reg = format(end - start)
times.append(time_hybrid_reg)
ar_hybrid = utils.clip_audio(ar_hybrid)
audios_processed.append(ar_hybrid)
nmse_hy_m, nmses_hy = benchmark.nmse_mean(conc_audio, ar_hybrid, pos_gap, taille_paquet)
mel_cs_hy = benchmark.mel_cs(conc_audio, ar_hybrid, new_sample_rate)
nmses.append(nmses_hy)
nmses_m.append(nmse_hy_m)
mels_cs.append(mel_cs_hy)
labels.append("Hybrid 256")
print("-------- Régression hybride 256 : Done ")
print(time_hybrid_reg)


#BENCHMARK 
print(f"Pour les méthodes NMSES puis MEL_CS:")
for i in range(len(nmses_m)) :
    print(f"{labels[i]}")
    print(f"{nmses_m[i]} \n  {mels_cs[i]}")



#directory = r"C:\Users\hippo\OneDrive\Bureau\MINES\2A\T2_Data_Sophia\PROJET\Data-Sophia-\Results\produced_audio"
directory = pathlib.Path().resolve()
audio_directory = os.path.join(directory, "produced_audio")

if not os.path.isdir(audio_directory) : 
    os.mkdir(audio_directory)
for i in range(len(audios_processed)) :
    audio = audios_processed[i]
    label = labels[i]
    utils.write_wav(audio, samplerate = new_sample_rate, name= label, directory = audio_directory)

peaq_sr  = 48000

#directory = r"C:\Users\hippo\OneDrive\Bureau\MINES\2A\T2_Data_Sophia\PROJET\Data-Sophia-\Results\peaq_test"
peaq_directory = os.path.join(directory,"peaq_test")
if not os.path.isdir(peaq_directory) : 
    os.mkdir(peaq_directory)
for i in range(len(audios_processed)) :
    audio = audios_processed[i]
    label = labels[i]
    utils.write_wav(audio, new_sample_rate, label, directory = peaq_directory, new_samplerate = peaq_sr)

#directory = r"C:\Users\hippo\OneDrive\Bureau\MINES\2A\T2_Data_Sophia\PROJET\Data-Sophia-\Results\metrics"
metrics_directory = os.path.join(directory, "metrics")
if not os.path.isdir(metrics_directory) :
    os.mkdir(metrics_directory)
plcmos_sr = 16000
plcmos_values = []
print(len(labels), labels)
print(len(nmses_m), nmses_m)
print(len(mels_cs), mels_cs)
#print(len(plcmos_values), plcmos_values)
print(len(times), times)
for i in range(len(audios_processed)) :
    audio = audios_processed[i]
    label = labels[i]

    #predict.write_wav(audio, new_sample_rate, label, directory = directory, new_samplerate = plcmos_sr)
    #audio_path = os.path.join(directory, label)
    plc_value = benchmark.plcmos_process(audio, plcmos_sr, new_sample_rate)
    plcmos_values.append(plc_value)

nb_loss_tot = [len(pos_gap)]*len(times)

benchmark_dict = {'méthode' : labels, 
        'NMSE' : nmses_m,
        'MEL_CS' : mels_cs,
        'PLCMOS' : plcmos_values,
        'Execution Time ' : times,
        'Nombre total de pertes' : nb_loss_tot
        }

df = DataFrame(benchmark_dict)
print(df) 
df_path = os.path.join(metrics_directory, "benchmark_results.csv")
df.to_csv(df_path)