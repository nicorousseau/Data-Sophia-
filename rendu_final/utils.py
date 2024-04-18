import numpy as np
import statsmodels
from scipy.io import wavfile
from scipy.signal import resample
import numpy.fft as fft
import os
import pathlib
import methods

def clip_audio(audio) : 
    """clip the audio

    Args:
        audio (np.array): audio

    Returns:
        np.array: audio_clipped
    """
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
def load_bach_data(directory, taille_paquet, n_loss_per_audio = 100, instr = 'saxphone.wav', resample_rate = None, normalise = False):
    """fonction permettant de charger et de process les audios du dataset de bach

    Args:
        directory (path): path du dossier contenant les audios
        taille_paquet (int): taille des paquet perdus
        n_loss_per_audio (int, optional): nombre de paquets perdus dans l'audio. Defaults to 100.
        instr (str, optional): l'instrument étudié. Defaults to 'saxphone.wav'.
        resample_rate (_type_, optional): nouveau samplerate désiré. Defaults to None.
        normalise (bool, optional): condition de normalisation. Defaults to False.

    Returns:
        list, list, np.array, np.array: liste des audios, liste des positions des gaps, audios concaténés, positions des gaps concaténées
    """
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
def data_prep(audio, sample_rate, new_sample_rate = None, to_mono = False):
    """fonction permettant de préparer un fichier audio donné pour le traitement (utilisé sur chopin)

    Args:
        audio (np.array): audio à traiter
        sample_rate (int): samplerate initial
        new_sample_rate (int, optional): nouveau samplerate désiré. Defaults to None.
        to_mono (bool, optional): condition de transformation en mono. Defaults to False.

    Returns:
        np.array: audio processed
    """
    max = np.iinfo(audio.dtype).max
    audio = audio.astype(np.float64)
    audio = audio/max
    if to_mono :
        audio = np.mean(audio, axis=1)
    if new_sample_rate is not None : 
        audio = resample(audio, int(len(audio)*new_sample_rate/sample_rate), window= "hamming", domain = "time")
    return audio
    
def keeping_important_freq(array, nb_max):
    """fonction permettant de réduire une fft à ses pics les plus importants (en valeur absolu)

    Args:
        array (np.array): fft du signal audio dont on veut extraire les fréquences
        nb_max (int): nombre de fréquences à garder

    Returns:
        np.array, np.array: indices des fréquences gardées, fréquences gardées
    """
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

def freq_max_sorted(audio, pos, train_size, sr, nb_freq_kept):
    """fonction extrayant les fréquences les plus importantes dans un signal audio
    en prenant en compte la position du gap dans l'audio

    Args:
        audio (np.array): audio dont on veut extraire les fréquences
        pos (int): position du gap dans l'audio
        train_size (int): taille du contexte avant le gap
        sr (int): samplerate de l'audio
        nb_freq_kept (int): nombre de fréquences à garder

    Returns:
        liste, float: liste des fréquences conservés, moyenne de l'audio
    """
    sample = audio[pos-train_size:pos]
    sample = np.pad(sample * np.hamming(train_size), (2000, 2000), 'constant')
    
    fft_abs = np.abs(np.fft.fft(sample)[:len(sample)//2])
    moy = 2 * fft_abs[0]/train_size
    freq = np.linspace(0,sr/2 - sr/len(fft_abs), len(fft_abs))
    
    r = fft_abs[1:]
    l = fft_abs[:-1]
    mask_l = r > l
    mask_r = l > r
    mask = mask_l[:-1] * mask_r[1:]
    mask = np.concatenate((np.array([False]), mask, np.array([False])))
    list_max = np.zeros(len(fft_abs))
    list_max[mask] = fft_abs[mask]
    ind = np.argsort(list_max)[-nb_freq_kept:]
    return freq[ind], moy

def compute_env(audio, window_size = 200) : #fonction erronnée : utilise le futur a changer dans l'avenir 
    """Fonction calculant l'enveloppe d'un signal audio, 
    en prenant le maximum et le minimum sur une fenêtre de taille window_size.
    Le problème avec cette fonction est qu'elle utilise le futur, invalidant toute la suite du traitement.
    Le reste du code peut-être conservé mais au vu d'utilisation future il faut changer cette fonction.

    Args:
        audio (np.array): audio dont on veut process l'enveloppe
        window_size (int, optional): taille de l'enveloppe utilisée pour calculer l'enveloppe. Defaults to 200.

    Returns:
        np.array, np.array: enveloppe positive, enveloppe négative
    """
    window_size = 200
    sample_pad = np.pad(audio, (window_size//2, window_size//2), mode='edge')

    sample_max = [np.max(sample_pad[i-window_size//2:i+window_size//2]) for i in range(window_size//2, len(sample_pad)-window_size//2)]
    env_max = np.convolve(sample_max, np.ones((window_size,))/window_size, mode='same')

    sample_min = [np.min(sample_pad[i-window_size//2:i+window_size//2]) for i in range(window_size//2, len(sample_pad)-window_size//2)]
    env_min = np.convolve(sample_min, np.ones((window_size,))/window_size, mode='same')
    return env_max, env_min

def compute_squared_audio(audio, env_max, env_min) :

    audio_pos = audio.copy()
    audio_neg = audio.copy()
    audio_pos[audio < 0] = 0
    audio_neg[audio>0] = 0

    audio_pos_norm = audio_pos / env_max
    audio_neg_norm = audio_neg / np.abs(env_min)

    audio_norm = audio_pos_norm.copy()
    audio_norm[audio <0] = audio_neg_norm[audio<0]
    return audio_norm

def audio_predictions(audio, pos_gap, taille_paquet, order = 128, train_size = None, adapt = False, crossfade_size = None):
    """fonction mettant en place le processus ar sur l'audio, avec ou sans adaptation et crossfade

    Args:
        audio (np.array): audio avec les gaps
        pos_gap (list): liste des positions des gaps dans l'audio
        taille_paquet (int): taille des paquets de données perdues
        order (int, optional): nombre de paramètres de l'ar. Defaults to 128.
        adapt (bool, optional): condition d'adaptation de l'ar. Defaults to False.
        crossfade_size (int, optional): nombre de sample du crossfade. Defaults to None.
        p_value (bool, optional): _description_. Defaults to False.

    Returns:
        np.array: audio complété par AR
    """
    AR_filled = audio.copy()
    if train_size is None : 
        taille_train = taille_paquet
    else : 
        taille_train = train_size
    for x in pos_gap : 
        train_data = np.copy(AR_filled[x-taille_train:x])
        if adapt : 
            pred = methods.forecast_adapt(train_data, taille_paquet, lags = order, n_thresh = 1, clip_value=1.3, crossfade_size=crossfade_size)

        else : 
            model = statsmodels.tsa.ar_model.AutoReg(train_data, order)
            model = model.fit()
            pred = model.predict(start = len(train_data), end = len(train_data)+taille_paquet-1, dynamic=True)
        
        if crossfade_size is not None : 
            n_cross = len(pred)-taille_paquet
            crossfade = np.linspace(-0.5, 0.5, n_cross)
            window = 1/(1+np.exp(crossfade))
            crossfaded = pred[taille_paquet:]*window + (1-window)*AR_filled[x+taille_paquet:x+taille_paquet+n_cross]
            AR_filled[x+taille_paquet:x+taille_paquet+n_cross] = crossfaded
        AR_filled[x:x+taille_paquet] = pred[:taille_paquet]
    return AR_filled

