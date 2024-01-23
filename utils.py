import numpy as np
import matplotlib.pyplot as plt
import statsmodels
from scipy.io import wavfile
from scipy.signal import resample
import os
import pathlib

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

def forecast(train, lags = 128, n_thresh = 1, clip_value = 1.3, crossfade_size = None):

    model = statsmodels.tsa.ar_model.AutoReg(train, lags)
    model_fit = model.fit()
    n_train = len(train)
    start = n_train
    clip_value = clip_value*np.max(np.abs(train))#np.iinfo(format).max
    if crossfade_size is None: 
        end = 2*n_train-1
    else : 
        end = 2*n_train+ n_train//int(1/crossfade_size) -1
    
    pred = model_fit.predict(start = start, end = end, dynamic = True)
    if (np.sum(np.abs(pred[:2*n_train-1]) > clip_value) > n_thresh) and (lags > 32): 
        new_pred = forecast(train, lags//2, n_thresh = n_thresh, crossfade_size = crossfade_size)
        if np.sum(np.abs(pred[:2*n_train-1]) > clip_value) > np.sum(np.abs(new_pred[:2*n_train-1]) > clip_value) : 
            return new_pred
        else :
            return pred
    return pred 

def audio_predictions(audio, pos_gap, taille_paquet, order = 128, adapt = False, clip_value = 1.3, crossfade_size = None, p_value = False):
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
    for x in pos_gap : 
        paquet = np.copy(audio[x-taille_paquet:x])
        n = taille_paquet
        if adapt : 
            pred = forecast(paquet, order, n_thresh = 1, clip_value=1.3, crossfade_size=crossfade_size)

        else : 
            model = statsmodels.tsa.ar_model.AutoReg(paquet, order)
            model = model.fit()
            pred = model.predict(start = n, end = 2*n-1, dynamic=True)
            if p_value :
                print(model.pvalues)
        
        if crossfade_size is not None : 
            n_cross = len(pred)-taille_paquet
            window = np.linspace(0, 1, n_cross)**2
            crossfaded = pred[taille_paquet:]*(1-window) + window*AR_filled[x+taille_paquet:x+taille_paquet+n_cross]
            AR_filled[x:x+taille_paquet+n_cross] = np.concatenate((pred[:taille_paquet],crossfaded))
        else : 
            AR_filled[x:x+n] = pred
    return AR_filled


def write_wav(audio, samplerate, name, new_samplerate = None,  directory = None) :
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
    wavfile.write(path, samplerate, audio.astype(np.int16))
