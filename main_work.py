import numpy as np
import matplotlib.pyplot as plt 
#from scipy.io import wavfile
#from scipy.signal import spectrogram
from scipy.signal import resample
#from librosa.display import specshow
import statsmodels.tsa.ar_model #sinon fonctionne pas en utilisant predict.py
import os
import benchmark
import Benchmark.plc_mos as plc_mos
import predict
import time
import pandas as pd
#from AR_freq import AR_freq
from nico_work import AR_hybride_2 as nico1
from nico_work import persistance_freq as nico2

new_sample_rate = 32000
directory = r"C:\Users\hippo\OneDrive\Bureau\MINES\2A\T2_Data_Sophia\PROJET\Bach10_v1.1"
instr_wanted = "saxphone.wav"
list_instr = ['all', 'saxphone.wav', 'bassoon.wav', 'clarinet.wav', 'violin.wav']
temps_paquet = 0.02 #20ms
taille_paquet = int(temps_paquet *new_sample_rate) #nb_echantillons par paquet
n_loss_per_audio = 120


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
    return ar_32, process_time

def process_env(audio_norm, env_max, env_min, order) : 
    start = time.time()
    env_audio, env_max_int, env_min_int, audio_norm_pred = predict.env_predictions(audio_norm, env_max, env_min, pos_gap, taille_paquet, order=order, train_size = 3* taille_paquet, adapt = True, ar_on_env = False)
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
    return env_audio, process_time
    

for instr in list_instr :
    print(instr)
    directory = r"C:\Users\hippo\OneDrive\Bureau\MINES\2A\T2_Data_Sophia\PROJET\Bach10_v1.1"

    if instr == 'all' :
        directory = os.path.join(directory, "All_instr")
    list_audio, list_pos_gap, conc_audio, pos_gap = predict.load_data(directory, taille_paquet, n_loss_per_audio= n_loss_per_audio, instr = instr, resample_rate = new_sample_rate, normalise=True)
    print(f"{len(pos_gap)}, {len(conc_audio)}, {len(conc_audio)/new_sample_rate}")
    #pos_gap = pos_gap[:2]
    #conc_audio = conc_audio[:pos_gap[1]+ 2*taille_paquet]
    conc_audio = np.array(conc_audio)
    #print(np.max(conc_audio), np.min(conc_audio))
    #print(conc_audio)
    #BENCHMARK PREP
    nmses = []
    nmses_m = []
    mels_cs = []
    labels = []
    times = []
    audios_processed = []
    #Silence et Persistance
    start = time.time()
    loss, persistance = predict.filling_silence_persistance(conc_audio, pos_gap, taille_paquet)
    end = time.time()
    time_loss_pers = format(end - start)
    times.append(time_loss_pers)
    times.append(time_loss_pers)
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



    ## NICO PERSISTANCE FREQUENTIELLE 
    ##nb_freq_kept = 50
    ##nb_lags = 256
    ##train_size = int(1.5*taille_paquet)
    ##predict_size = taille_paquet
#
    #Reg_freq = nico2.Persistance_freq(conc_audio, new_sample_rate, new_sample_rate, nb_freq_kept, train_size)
    ##AR_hybrid.initialise()
    ##regression_freq = conc_audio.copy()
    ##start = time.time()
    ##for pos in pos_gap :
    ##    Reg_freq.fit(pos=pos) #chgt la méthode de reg
    ##    Reg_freq.predict(predict_size)
    ##    regression_freq[pos: pos+taille_paquet] = Reg_freq.pred
    ##end = time.time()
    ##ref_freq_time = format(end-start)
    ##times.append(ref_freq_time)
    ##audios_processed.append(regression_freq)
    ##nmse_reg_freq_m, nmses_reg_freq = benchmark.nmse_mean(conc_audio, regression_freq, pos_gap, taille_paquet)
    ##mel_cs_hy_env = benchmark.mel_cs(conc_audio, regression_freq, new_sample_rate)
    ##nmses.append(nmses_reg_freq)
    ##nmses_m.append(nmse_reg_freq_m)
    ##mels_cs.append(mel_cs_hy_env)
    ##labels.append("Reg Freq")
    
    #64
    #ar64, time_64 = ar_process(64, adapt = False)
    #times.append(time_64)
    #audios_processed.append(ar64)
    #labels.append("ar64")
    ###ar_adapt64 = ar_process(64, adapt = True)
    ###env64 = process_env(audio_norm, env_max, env_min, 64)
    ###audios_processed.append(env64)
    ###labels.append("env_ar64")


    #128
    #ar128, time_128 = ar_process(128, adapt = False)
    #times.append(time_128)
    #audios_processed.append(ar128)
    #labels.append("ar128")
    ###ar_adapt128 = ar_process(128, adapt = True)
    ###env128 = process_env(audio_norm, env_max, env_min, 128)
    ###audios_processed.append(env128)
    ###labels.append("env128")



    ##256
    #ar256, time_256= ar_process(256, adapt = False)
    #times.append(time_256)
    #audios_processed.append(ar256)
    #labels.append("ar256")

    #aar256, times_a256= ar_process(256, adapt = True)
    #times.append(times_a256)
    #audios_processed.append(aar256)
    #labels.append("aar256")


    #env_max, env_min = predict._compute_env(conc_audio)
    #audio_norm = predict._compute_without_env_audio(conc_audio, env_max, env_min)
    #env256, time_e256 = process_env(audio_norm, env_max, env_min, 256)
    #times.append(time_e256)
    #audios_processed.append(env256)
    #labels.append("env256")

    #METHODE HYBRIDE AVEC REGRESSION 
    nb_freq_kept = 20
    nb_lags = 256
    train_size = 2*taille_paquet
    predict_size = taille_paquet

    AR_hybrid = nico1.AR_freq(conc_audio, new_sample_rate, new_sample_rate, nb_lags, nb_freq_kept, train_size, taille_paquet)
    #AR_hybrid.initialise()
    ar_hybrid = conc_audio.copy()
    start = time.time()
    for pos in pos_gap :
        AR_hybrid.fit(pos=pos)
        AR_hybrid.predict(predict_size)
        ar_hybrid[pos: pos+taille_paquet] = AR_hybrid.pred
    end = time.time()
    time_hybrid_reg = format(end - start)
    times.append(time_hybrid_reg)
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
 
    ## Visual 
    #fig, axes = plt.subplots(nrows = 1, ncols = 2,  figsize = (9,4))
    #axes[0].boxplot(nmses, labels= labels)
    #axes[0].set_title("NSME")
#
    #axes[1].plot(labels,np.log10(mels_cs))
    #axes[1].set_title("Mel_CS")
    #plt.show()

    
    directory = r"C:\Users\hippo\OneDrive\Bureau\MINES\2A\T2_Data_Sophia\PROJET\Data-Sophia-\Results\produced_audio"
    directory = os.path.join(directory, instr[:-4])
    if instr == 'all':
        directory = os.path.join(directory, 'all')
    if not os.path.isdir(directory) : 
        os.mkdir(directory)
    for i in range(len(audios_processed)) :
        audio = audios_processed[i]
        label = labels[i]
        audio[audio>1] = 1.
        audio[audio<-1] = -1.
        predict.write_wav(audio, samplerate = new_sample_rate, name= label, directory = directory)

    peaq_sr  = 48000
    
    directory = r"C:\Users\hippo\OneDrive\Bureau\MINES\2A\T2_Data_Sophia\PROJET\Data-Sophia-\Results\peaq_test"
    directory = os.path.join(directory, instr[:-4])
    if not os.path.isdir(directory) : 
        os.mkdir(directory)
    for i in range(len(audios_processed)) :
        audio = audios_processed[i]
        label = labels[i]
        audio[audio>1] = 1.
        audio[audio<-1] = -1.
        predict.write_wav(audio, new_sample_rate, label, directory = directory, new_samplerate = peaq_sr)

    directory = r"C:\Users\hippo\OneDrive\Bureau\MINES\2A\T2_Data_Sophia\PROJET\Data-Sophia-\Results\metrics"
    if instr == 'all':
        directory = os.path.join(directory, 'all')
    directory = os.path.join(directory, instr[:-4])
    if not os.path.isdir(directory) : 
        os.mkdir(directory)
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
        audio[audio>1] = 1.
        audio[audio<-1] = -1.
        #predict.write_wav(audio, new_sample_rate, label, directory = directory, new_samplerate = plcmos_sr)
        #audio_path = os.path.join(directory, label)
        plcmos_audio = resample(audio, int(len(audio)*plcmos_sr/new_sample_rate),window= "hamming", domain = "time")
        plcmos = plc_mos.PLCMOSEstimator()
        plc_value = plcmos.run(plcmos_audio, plcmos_sr)
        plcmos_values.append(plc_value)
    
    nb_loss_tot = [len(pos_gap)]*len(times)
    
    benchmark_dict = {'méthode' : labels, 
            'NMSE' : nmses_m,
            'MEL_CS' : mels_cs,
            'PLCMOS' : plcmos_values,
            'Execution Time ' : times,
            'Nombre total de pertes' : nb_loss_tot
            }
    df = pd.DataFrame(benchmark_dict)
    print(df) 
    df_path = os.path.join(directory, "benchmark_results.csv")
    df.to_csv(df_path)