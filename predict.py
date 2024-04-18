import numpy as np
import matplotlib.pyplot as plt
import statsmodels
from scipy.io import wavfile
from scipy.signal import resample
import numpy.fft as fft
import os
import pathlib
from numpy.matlib import repmat
import scipy.interpolate
from sklearn.linear_model import LinearRegression
import time
#from statsmodels.tsa.stattools import adfuller



def load_data(directory, taille_paquet, n_loss_per_audio = 100, instr = 'saxphone.wav', resample_rate = None, normalise = False):
    audio_filespath =[]
    audio_files_tuple = []
    if instr == 'all' :
        for root, dirs, files in os.walk(directory):
            for filename in files:
                #print(filename[len(filename)-len(instr):len(filename)])
                pathfile = os.path.join(root, filename)
                audio_filespath.append(pathfile)
                audio_files_tuple.append(wavfile.read(pathfile))
    else :
        for root, dirs, files in os.walk(directory):
            for filename in files:
                #print(filename[len(filename)-len(instr):len(filename)])
                if filename[len(filename)-len(instr):len(filename)] == instr :
                    pathfile = os.path.join(root, filename)
                    audio_filespath.append(pathfile)
                    audio_files_tuple.append(wavfile.read(pathfile))
    audio_files = []
    list_audio = []
    list_pos_gap = []
    conc_audio = []
    pos_gap = np.array([], dtype = np.int32)
    for i in range(len(audio_files_tuple)):
        audio = audio_files_tuple[i][1].astype(dtype = np.float64)
        if resample_rate is not None : 
            audio = resample(audio, int(len(audio)*resample_rate/audio_files_tuple[i][0]), window= "hamming", domain = "time")
        if normalise : 
            max = np.iinfo(audio_files_tuple[i][1][0].dtype).max#//2
            audio = audio/max
        audio_files.append([audio_files_tuple[i][0], audio])
        list_audio.append(audio)
        #CREATIONS LOSS ALEATOIRES : 
        loss_audio = los_generation(len(audio), taille_paquet, n_loss = n_loss_per_audio)
        list_pos_gap.append(loss_audio)
        pos_gap = np.concatenate((pos_gap, loss_audio + len(conc_audio)))
        conc_audio = np.concatenate((conc_audio, audio))
    return list_audio, list_pos_gap, np.array(conc_audio), np.array(pos_gap)

def spectrogram(signal, samplerate = 22050, n_fft = 512, window = "haming", windows_length = None, hop_length = None) : 
    if windows_length is None : 
        windows_length = n_fft
    if hop_length is None : 
        hop_length = windows_length//4
        
    n_window = len(signal)//windows_length
    r = len(signal) % windows_length
    padded_signal = np.pad(signal, (0,r))
    signal_fenetree = []
    for i in range(n_window) : 
        signal_fenetree.append(np.pad(padded_signal[i*windows_length:(i+1)*windows_length], pad_width = ((windows_length-n_fft)//2, (windows_length-n_fft)//2), mode = 'constant'))
    spec = []
    for j in range(n_window):
        spec.append(np.flip(np.fft.fft(signal_fenetree[j], n_fft))[0:n_fft//2])
    img = np.array(spec).T
    t = np.linspace(0, len(signal)/samplerate, n_window)
    f = np.linspace(0,windows_length//2, windows_length//2)
    #plt.imshow(img)
    return t, f, np.log(np.abs(img))

#TODO : apodiser le signal avec la fenetre ? 
#TODO : compute la fft sur chaque fenetre e

def specshow(spec) :
    pass
def los_generation(taille_audio, taille_paquet, n_loss):
    """Fonction générant les positions des gaps dans l'audio

    Args:
        taille_audio (int): nb de sample dans l'audio
        taille_paquet (int): taille des gaps que l'on va créer
        n_loss (int): nombre de trous à partir duquel on va générer

    Returns:
        np.array: position des gaps dans l'audio dont on a fourni la taille
    """
    pos_gap = np.random.randint(2*taille_paquet, taille_audio- taille_paquet, n_loss)
    pos_gap = np.sort(pos_gap)
    pos_loss_decale = pos_gap[:n_loss-1]
    diff = pos_gap[1:n_loss]- pos_loss_decale
    mask = diff > 4*taille_paquet
    pos_gap = pos_gap[1:n_loss][mask]
    
    return pos_gap


#fonction pour silence et persistance
def filling_silence_persistance(audio, pos_gap, taille_paquet) :
    audio_persistance = audio.copy()
    audio_gap_filled = audio.copy()

    for x in pos_gap : 
        persistant = np.copy(audio[x-taille_paquet:x])

        audio_persistance[x:x+taille_paquet] = persistant
        audio_gap_filled[x:x+taille_paquet] = np.zeros(taille_paquet)
        
    return audio_gap_filled, audio_persistance

def forecast(train_data, n_pred, lags = 128, n_thresh = 1, clip_value = 1.3, crossfade_size = None):

    model = statsmodels.tsa.ar_model.AutoReg(train_data, lags)
    model_fit = model.fit()
    n_train = len(train_data)
    clip_value = clip_value*np.max(np.abs(train_data))#np.iinfo(format).max
    if crossfade_size is None: 
        end = n_train+n_pred-1
    else : 
        end = n_train+ n_pred*(1 + 1//int(1/crossfade_size)) -1
    
    pred = model_fit.predict(start = n_train, end = end, dynamic = True)
    if (np.sum(np.abs(pred) > clip_value) > n_thresh) and (lags >= 32): 
        #adf, pval, usedlag, nobs, crit_vals, icbest =  adfuller(train_data[-n_pred:])
        #print('ADF test statistic:', adf)
        #print('ADF p-values:', pval)
        #print('ADF number of lags used:', usedlag)
        #print('ADF number of observations:', nobs)
        #print('ADF critical values:', crit_vals)
        #print('ADF best information criterion:', icbest)
        new_pred = forecast(train_data, n_pred, lags//2, n_thresh = n_thresh, crossfade_size = crossfade_size)
        if np.sum(np.abs(pred) > clip_value) > np.sum(np.abs(new_pred) > clip_value) : 
            return new_pred
        else :
            return pred
    return pred 

def audio_predictions(audio, pos_gap, taille_paquet, order = 128, train_size = None, adapt = False, clip_value = 1.3, crossfade_size = None, p_value = False):
    """_summary_

    Args:
        audio (_type_): _description_
        pos_gap (_type_): _description_
        taille_paquet (_type_): _description_
        order (int, optional): _description_. Defaults to 128.
        adapt (bool, optional): _description_. Defaults to False.
        crossfade_size (_type_, optional): _description_. Defaults to None.
        p_value (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    AR_filled = audio.copy()
    if train_size is None : 
        taille_train = taille_paquet
    else : 
        taille_train = train_size
    for x in pos_gap : 
        train_data = np.copy(AR_filled[x-taille_train:x])
        if adapt : 
            pred = forecast(train_data, taille_paquet, lags = order, n_thresh = 1, clip_value=1.3, crossfade_size=crossfade_size)

        else : 
            model = statsmodels.tsa.ar_model.AutoReg(train_data, order)
            model = model.fit()
            pred = model.predict(start = len(train_data), end = len(train_data)+taille_paquet-1, dynamic=True)
            if p_value :
                print(model.pvalues)
        
        if crossfade_size is not None : 
            n_cross = len(pred)-taille_paquet
            crossfade = np.linspace(-0.5, 0.5, n_cross)
            window = 1/(1+np.exp(crossfade))
            crossfaded = pred[taille_paquet:]*window + (1-window)*AR_filled[x+taille_paquet:x+taille_paquet+n_cross]
            AR_filled[x+taille_paquet:x+taille_paquet+n_cross] = crossfaded
        AR_filled[x:x+taille_paquet] = pred[:taille_paquet]
    return AR_filled


def _compute_env(audio, window_size = 200) : 
    window_size = 200
    sample_pad = np.pad(audio, (window_size//2, window_size//2), mode='edge')

    sample_max = [np.max(sample_pad[i-window_size//2:i+window_size//2]) for i in range(window_size//2, len(sample_pad)-window_size//2)]
    env_max = np.convolve(sample_max, np.ones((window_size,))/window_size, mode='same')

    sample_min = [np.min(sample_pad[i-window_size//2:i+window_size//2]) for i in range(window_size//2, len(sample_pad)-window_size//2)]
    env_min = np.convolve(sample_min, np.ones((window_size,))/window_size, mode='same')
    return env_max, env_min

def _compute_without_env_audio(audio, env_max, env_min) :

    audio_pos = audio.copy()
    audio_neg = audio.copy()
    audio_pos[audio < 0] = 0
    audio_neg[audio>0] = 0

    audio_pos_norm = audio_pos / env_max
    audio_neg_norm = audio_neg / np.abs(env_min)

    audio_norm = audio_pos_norm.copy()
    audio_norm[audio <0] = audio_neg_norm[audio<0]
    return audio_norm

def env_predictions(audio_norm, env_max, env_min, pos_gap, taille_paquet, order = 128, train_size = None, adapt = False, clip_value = 1.3, crossfade_size = None, ar_on_env = False):

    norm_to_fill = audio_norm.copy()
    
    norm = audio_predictions(norm_to_fill, pos_gap, taille_paquet, order=order, train_size = train_size, adapt = adapt)
    if ar_on_env : 
        env_max_int = env_ar(env_max, pos_gap, taille_paquet=taille_paquet, train_size = 2*taille_paquet, order = order)#env_interpo(env_max, pos_gap, taille_paquet, train_size = taille_paquet//4)
        env_min_int = env_ar(np.abs(env_min), pos_gap, taille_paquet=taille_paquet, train_size=2*taille_paquet, order = order)#env_interpo(np.abs(env_min), pos_gap, taille_paquet, train_size = taille_paquet//4)
    else : 
        env_max_int = env_interpo(env_max, pos_gap, taille_paquet, train_size = taille_paquet//4)
        env_min_int = env_interpo(np.abs(env_min), pos_gap, taille_paquet, train_size = taille_paquet//4)


    norm_pos_1 = norm.copy()
    norm_neg_1 = norm.copy()
    mask_pos = norm > 0
    mask_neg = norm < 0
    norm_pos_1[mask_neg] = 0
    norm_neg_1[mask_pos] = 0
    audio_norm_pred = norm.copy()
    audio_norm_pred[mask_neg] = norm_neg_1[mask_neg]
    audio_norm_pred[mask_pos] = norm_pos_1[mask_pos]
    audio_pos = norm_pos_1 * env_max_int
    audio_neg = norm_neg_1 * env_min_int
    double_env_int = audio_pos.copy()
    double_env_int[mask_neg] = audio_neg[mask_neg]
    
    
    return double_env_int, env_max_int, env_min_int, audio_norm_pred

def env_interpo(env, pos_gap, taille_paquet, train_size, order = 1):
    env_pred = env.copy()
    for x in pos_gap :
        train_data = np.copy(env_pred[x-train_size:x])
        coord = np.arange(x-train_size,x).reshape(-1,1)
        reg = LinearRegression().fit(coord, train_data)
        def f(cord): 
            a = reg.coef_
            b = train_data[-1] - a*(x-1)
            return a*cord + b
        fv = np.vectorize(f)
        #print(f"coef : {reg.coef_}")
        #cs = scipy.interpolate.make_interp_spline(np.arange(x-train_size,x), train_data, k = order)#np.interp(np.arange(x, x+taille_paquet), np.arange(x-train_size,x), train_data)
        data = np.arange(x, x+taille_paquet)#.reshape(-1,1)
        pred = fv(data)
        mask_neg = pred < 0
        pred[mask_neg] = 0
        env_pred[x:x+taille_paquet] = pred
    return env_pred

def env_ar(env, pos_gap, taille_paquet, train_size, order = 256):
    env_pred = env.copy()
    for x in pos_gap : 
        clip_value = np.max(env[:x])
        train_data = np.copy(env_pred[x-train_size:x])
        pred = env_forecast(train_data, n_pred = taille_paquet, lags = order, clip_value = clip_value, n_thresh = 1)
        env_pred[x:x+taille_paquet] = pred
    return env_pred
        
def env_forecast(train_data, n_pred, lags = 128, clip_value = None, n_thresh = 1,  crossfade_size = None):

    model = statsmodels.tsa.ar_model.AutoReg(train_data, lags, trend = 'c')
    model_fit = model.fit()
    n_train = len(train_data)
    #np.iinfo(format).max
    if crossfade_size is None: 
        end = n_train+n_pred-1
    else : 
        end = n_train+ n_pred*(1 + 1//int(1/crossfade_size)) -1
    
    pred = model_fit.predict(start = n_train, end = end, dynamic = True)
    if ((np.sum(pred < 0) > n_thresh) or np.sum(np.abs(pred) > clip_value) > n_thresh) and (lags >= 32): 
        #print(lags//2)
        #print(n_train//2)
        new_pred = env_forecast(train_data, n_pred, lags//2, clip_value = clip_value, n_thresh = n_thresh,  crossfade_size = crossfade_size)
        if (np.sum(np.abs(pred) > clip_value) > np.sum(np.abs(new_pred) > clip_value)) and (np.sum(new_pred < 0) > np.sum(pred < 0)): 
            pred = new_pred
        pred[pred<0] = 0
        pred[pred > clip_value] = clip_value
    return pred 

def freq_persistance(audio, pos_gap, taille_paquet, sample_rate, n_harm = 15, crossfade_size = None):
    audio_corr = audio.copy()
    temps_paquet = taille_paquet/sample_rate
    taille_context = 1#.5
    for pos in pos_gap :
        if pos>taille_context*taille_paquet :
            train_size = int(taille_context*taille_paquet)
        else : 
            train_size = taille_paquet
        train = audio_corr[pos-train_size: pos].copy()
        m = np.mean(train)
        #Finding the period of the signal with autocorrelation
        #computing fft and dephasing it 
        train = train*np.hamming(len(train))
        fft_signal = fft.fft(train, 10*train_size)
        freq = np.fft.fftfreq(len(fft_signal), d=1/sample_rate)
        fft_abs = np.abs(fft_signal)
        kept_ind, max_kept = keeping_important_freq(fft_abs, 2*n_harm)
        freq_kept = freq[kept_ind]
        fft_kept = fft_signal[kept_ind]
        amp_kept = fft_abs[kept_ind]
        if crossfade_size is not None : 
            n_cross = crossfade_size
            cross_window = np.linspace(-0.5, 0.5, n_cross)
            cross_window = 1/(1+np.exp(cross_window))
            time = np.arange(train_size, train_size+taille_paquet+n_cross)/sample_rate
            pred = []
            for i in range (len(time)) : 
                pred.append(2*np.sum([fft_kept[:len(fft_kept)//2]*np.exp(2j*np.pi*freq_kept[:len(fft_kept)//2]*time[i])])/len(time))#norm)
            pred = np.real(pred) +m
            audio_corr[pos:pos+taille_paquet] = pred[:taille_paquet]
            audio_corr[pos+taille_paquet: pos+taille_paquet+ n_cross] = cross_window*pred[taille_paquet:]+ (1-cross_window)*audio_corr[pos+taille_paquet: pos+taille_paquet+ n_cross]
        else : 
            time = np.arange(train_size, train_size+taille_paquet)/sample_rate
            pred = []
            for i in range (len(time)) : 
                pred.append(2*np.sum([fft_kept[:len(fft_kept)//2]*np.exp(2j*np.pi*freq_kept[:len(fft_kept)//2]*time[i])])/len(time))#norm)
            pred = np.real(pred) + m
            #plt.plot(np.arange(0, train_size+taille_paquet)/sample_rate, audio_corr[pos-train_size:pos+taille_paquet])
            #plt.plot(time,pred, 'g--')
            #plt.show()
            audio_corr[pos:pos+taille_paquet] = pred
            
        #METHODE JUSTE AVEC LE CALCUL DU TEMPS
        #ws = audio_corr[pos-taille_paquet:pos]
        #wt = np.arange(0, taille_paquet) * 1/sample_rate
        #wN_ = taille_paquet * 2
        #wt_ = np.arange(0, wN_) * 1/sample_rate
        #TFws = np.fft.fft(ws, taille_paquet)
        #f = np.linspace(-0.5, 0.5 - 1/taille_paquet, taille_paquet) * sample_rate
        #TFws_p = TFws[:taille_paquet//2]
        #f_p = f[np.arange(taille_paquet//2 + 1, taille_paquet )]
        #aTFws_p = np.abs(TFws_p)
        #Lf = 150
        #idx_maxloc = np.where(aTFws_p > np.maximum(np.roll(aTFws_p, 1), np.roll(aTFws_p, -1)))[0]
        #sort_maxloc = np.sort(aTFws_p[idx_maxloc])[::-1]
        #idx_sort_maxloc = np.argsort(aTFws_p[idx_maxloc])[::-1]
        #sel_f_p = f_p[idx_maxloc[idx_sort_maxloc[:Lf]]]
        #sel_TFws_p = TFws_p[idx_maxloc[idx_sort_maxloc[:Lf]]]
        #temp = np.exp(2j * np.pi * np.outer(sel_f_p, wt_ )+ np.outer(np.angle(sel_TFws_p), np.ones(len(wt_))))
        #ws_rec_ = 2 * np.real(np.sum(repmat(np.abs(sel_TFws_p), wN_, 1).T * temp, axis=0)) / taille_paquet
        #fig, axs = plt.subplots(2)
        #axs[0].plot(audio_corr[pos-taille_paquet:pos+taille_paquet], 'b')
        #axs[1].plot(ws_rec_, 'r--')
        #plt.show()
    return audio_corr


def freq_persistance2(audio, pos_gap, taille_paquet, sample_rate, n_harm):
    audio_corr = audio.copy()
    temps_paquet = taille_paquet/sample_rate
    taille_context = 2#.5
    taille_train = 2*taille_paquet
    time = np.linspace(taille_train//2, (taille_train+taille_paquet-1)/sample_rate, taille_train//2+taille_paquet)
    for pos in pos_gap :
        train = audio_corr[pos-taille_train: pos].copy()
        train = np.pad(train*np.hanning(taille_train), (10000,10000), 'constant')
        #computing fft and dephasing it 
        fft_signal = fft.fft(train)
        freq = fft.fftfreq(len(fft_signal), d=1/sample_rate)[:len(fft_signal)//2]
        fft_signal = fft_signal[:len(fft_signal)//2]
        fft_abs = np.abs(fft_signal)
        phase = np.angle(fft_signal)
        ind_kept, abs_kept = keeping_important_freq(fft_abs, nb_max = n_harm)
        
        pred = [2*np.sum(np.real(fft_abs[ind_kept]/(taille_train)*np.exp(1j*2*np.pi*freq[ind_kept]*time[i] + phase[ind_kept]))) for i in range(len(time))]
        audio_corr[pos:pos+taille_paquet] = pred[taille_train//2:]
    return audio_corr

#def _finding_local_max(audio):
#    r = np.concatenate((audio[1:], [0]))
#    l = np.concatenate(([0],audio[:-1]))
#    mask_r = audio > r 
#    mask_l = audio > l 
#    mask = mask_r*mask_l
#    list_max = np.zeros(len(audio))
#    list_max[mask] = audio[mask]
#    ind = np.argsort(list_max)[::-1]
#    return ind

def _finding_local_max(array):
    mask = [False]
    n = len(array)
    for i in range(1, n-1):
        if (array[i-1] < array[i]) and (array[i+1] < array[i]) :
            mask.append(True)
        else : 
            mask.append(False)
    mask.append(False)
    list_max = np.zeros(n)
    list_max[mask] = array[mask].copy()
    ind = np.argsort(list_max)[::-1]
    return ind

def keeping_important_freq(array, nb_max):
    abs_array = np.abs(array)
    r = abs_array[1:]
    l = abs_array[:-1]
    mask_l = r > l
    mask_r = l > r
    mask = mask_l[:-1] * mask_r[1:]
    mask = np.concatenate(([False], mask, [False]))
    list_max = np.zeros(len(array))
    list_max[mask] = abs_array[mask]
    ind_kept = np.argsort(list_max)[::-1][:nb_max]
    
    max_kept = np.zeros(len(array), dtype = np.complex128)
    max_kept[ind_kept] = array[ind_kept]

    return ind_kept, max_kept

def clip_audio(audio) : 
    if np.max(np.abs(audio)) < 1 : 
        return audio
    else : 
        audio[audio>1.] = 1.
        audio[audio<-1.] = -1.
    return audio

def write_wav(audio, samplerate, name, new_samplerate = None,  directory = None, return_type = True) :
    """fonction convertissant les audios dans le bon format avant de les enregistrer en .wav

    Args:
        audio (np.array): array représentant l'audio
        samplerate (int): samplerate de l'audio
        name (str): nom du fichier que l'on veut écrire
        new_samplerate (int, optional): Si donné, resample l'audio à cette valeur avant de l'enregistrer. Defaults to None.
        directory (str, optional): si donné, enregistre le fichier audio dans ce dossier. Defaults to None.
    """
    path = name + ".wav"
    current_path = pathlib.Path().resolve()
    if new_samplerate is not None : 
        audio = resample(audio, int(len(audio)*new_samplerate/samplerate), window= "hamming", domain = "time")
        samplerate = new_samplerate
    if directory is not None : 
        if not os.path.exists(directory) : 
            os.mkdir(directory)
        path = os.path.join(directory, path)
    path = os.path.join(current_path, path)
    new_audio = audio * np.iinfo(np.int16).max
    new_audio = new_audio.astype(np.int16)
    #if not return_type : 
    #    new_audio = new_audio.astype(np.float32)
    wavfile.write(path, samplerate, new_audio)

