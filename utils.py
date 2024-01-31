import numpy as np
import matplotlib.pyplot as plt
import statsmodels
from scipy.io import wavfile
from scipy.signal import resample
import scipy.fft as fft
import os
import pathlib


def load_data(directory, instr = 'saxphone', resample_rate = None):
    audio_filespath =[]
    audio_files_tuple = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            #print(os.path.join(root, filename))
            if filename[len(filename)-len(instr):len(filename)] == instr :
                pathfile = os.path.join(root, filename)
                audio_filespath.append(pathfile)
                audio_files_tuple.append(wavfile.read(pathfile))
    print(len(audio_files_tuple))
    audio_files = []
    list_audio = []
    if resample_rate is not None : 
        for i in range(len(audio_files_tuple)):
            audio_files.append([audio_files_tuple[i][0], audio_files_tuple[i][1].astype(dtype = np.float64)])
            list_audio.append(resample(audio_files[i][1], int(len(audio_files[i][1])*resample_rate/audio_files[i][0]), window= "hamming", domain = "time"))
    else : 
        for i in range(len(audio_files_tuple)):
            audio_files.append([audio_files_tuple[i][0], audio_files_tuple[i][1].astype(dtype = np.float64)])
            list_audio.append(audio_files[i][1])
    return list_audio

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
    mask = diff > taille_paquet
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

def forecast(train, n_pred, lags = 128, n_thresh = 1, clip_value = 1.3, crossfade_size = None):

    model = statsmodels.tsa.ar_model.AutoReg(train, lags)
    model_fit = model.fit()
    n_train = len(train)
    clip_value = clip_value*np.max(np.abs(train))#np.iinfo(format).max
    if crossfade_size is None: 
        end = n_train+n_pred-1
    else : 
        end = n_train+ n_pred*(1 + 1//int(1/crossfade_size)) -1
    
    pred = model_fit.predict(start = n_train, end = end, dynamic = True)
    if (np.sum(np.abs(pred[:2*n_train-1]) > clip_value) > n_thresh) and (lags >= 32): 
        new_pred = forecast(train, n_pred, lags//2, n_thresh = n_thresh, crossfade_size = crossfade_size)
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
    taille_train = 3*taille_paquet
    for x in pos_gap : 
        train = np.copy(audio[x-taille_train:x])
        if adapt : 
            pred = forecast(train, taille_paquet, lags = order, n_thresh = 1, clip_value=1.3, crossfade_size=crossfade_size)

        else : 
            model = statsmodels.tsa.ar_model.AutoReg(train, order)
            model = model.fit()
            pred = model.predict(start = len(train), end = len(train)+taille_paquet-1, dynamic=True)
            if p_value :
                print(model.pvalues)
        
        if crossfade_size is not None : 
            n_cross = len(pred)-taille_paquet
            window = np.linspace(0, 1, n_cross)**2
            crossfaded = pred[taille_paquet:]*(1-window) + window*AR_filled[x+taille_paquet:x+taille_paquet+n_cross]
            AR_filled[x:x+taille_paquet+n_cross] = np.concatenate((pred[:taille_paquet],crossfaded))
        else : 
            AR_filled[x:x+taille_paquet] = pred
    return AR_filled

def freq_persistance(audio, pos_gap, taille_paquet, sample_rate):
    audio_corr = audio.copy()
    temps_paquet = taille_paquet/sample_rate
    taille_context = 1.5
    for pos in pos_gap :
        if pos>taille_context*taille_paquet :
            taille_train = int(taille_context*taille_paquet)
        else : 
            taille_train = taille_paquet
        train = audio_corr[pos-taille_train: pos]
        #Finding the period of the signal with autocorrelation
        autocor = np.correlate(train, train, mode = 'same')
        ind = _finding_local_max(autocor)
        period = np.abs(ind[0]- ind[1])/sample_rate
        #Fixing parameters needed
        x = taille_context*temps_paquet / period
        t_phased =  taille_context*temps_paquet- np.floor(x)*period + period #we add a period to stay in the right place
        #computing fft and dephasing it 
        fft_signal = fft.fft(train)
        freq = fft.fftfreq(len(train), d=1/sample_rate)
        dephasage = np.exp(1j*2*np.pi*t_phased*freq)
        fft_signal = fft_signal * dephasage
        
        pred = np.real(fft.ifft(fft_signal))[:taille_paquet]
        audio_corr[pos:pos+taille_paquet] = pred
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
