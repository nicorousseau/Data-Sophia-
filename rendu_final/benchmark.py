import numpy as np 
from scipy.signal import resample
from librosa.feature import melspectrogram
from librosa.display import specshow
import PLCMOS.plc_mos as plc_mos

def nmse(y, y_predicted) :
    """compute nmse between y and y_predicted

    Args:
        y (np.array): signal original
        y_predicted (np.array): signal prédit

    Returns:
        float : valeur de l'erreur nmse
    """
    diff = y - y_predicted
    norm_diff = np.square(diff).sum()
    norm_y = np.square(y).sum()
    try : 
        res = 10*np.log10(norm_diff/norm_y)
    except : 
        print(f"norm_diff/norm_y = {norm_diff/norm_y}")
    return res

def nmse_mean(audio, audio_filled, pos_gap, taille_paquet):
    """compute chaque nmse et la moyenne de la nmse entre les paquets de taille taille_paquet de audio et audio_filled

    Args:
        audio (np.array): audio original, avec des gap de taille taille_paquet
        audio_filled (np.array): audio corrigé par une méthode quelconque 
        pos_gap (list): liste des positions des gaps dans l'audio
        taille_paquet (int): taille des paquets de données perdues

    Returns:
        float, list: moyenne des nmse, liste des nmse
    """
    list_nmse = []
    for k in range(len(pos_gap)) : 
        original = audio[pos_gap[k]: pos_gap[k]+taille_paquet]
        predicted = audio_filled[pos_gap[k]: pos_gap[k]+taille_paquet]
        list_nmse.append(nmse(original, predicted))
    
    return np.mean(list_nmse), list_nmse
    

def mel_cs(y, y_predicted, sr) : 
    """calcule la distance (MAE) entre les spectrogrammes mels de y et y_predicted 
    (un spectrogramme mel est une représentation du signal audio en fréquence et en temps, 
    avec une échelle de fréquence logarithmique censé ccorrespondre à la perception humaine de la fréquence)

    Args:
        y (np.array): audio original
        y_predicted (np.array): audio avec perte de paquet
        sr (int): samplerate

    Returns:
        float: valeur de la distance (MAE) entre les spectrogrammes mels
    """
    y = np.array(y)
    y_predicted = np.array(y_predicted)
    sample_rate = sr
    win_length = 512
    n_fft = 1024
    hop_length = 256
    y_spec = melspectrogram(y=y, sr=sample_rate, S= None, n_fft=n_fft, hop_length= hop_length, win_length=win_length, window='hann', center=True, pad_mode='constant', power=2.0, n_mels = 64)
    y_pred_spec = melspectrogram(y=y_predicted, sr=sample_rate, S= None, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', center=True, pad_mode='constant', power=2.0, n_mels = 64)
    diff = np.linalg.norm(y_spec-y_pred_spec, ord='fro')

    return diff/np.linalg.norm(y_spec, ord = 'fro')


def plcmos_process(audio, plcmos_sr, new_sample_rate): 
    """calcule le score (MOS) à l'aide de plcmos sur des fenêtres de 10 secondes de l'audio puis moyenne

    Args:
        audio (np.array): audio original
        plcmos_sr (int): Samplerate pour le calcul du score plcmos
        new_sample_rate (_type_): sample_rate actuel de l'audio

    Returns:
        float: score MOS de l'audio
    """
    plcmos_audio = resample(audio, int(len(audio)*plcmos_sr/new_sample_rate),window= "hamming", domain = "time")
    plcmos = plc_mos.PLCMOSEstimator()
    size_window = 10*plcmos_sr
    plcmos_windowed = []
    for i in range(len(plcmos_audio)//size_window-1):
        plcmos_windowed.append(plcmos.run(plcmos_audio[i*size_window:(i+1)*size_window], plcmos_sr))
    plcmos_windowed.append(plcmos.run(plcmos_audio[(len(plcmos_audio)//size_window)*size_window:-1], plcmos_sr))
    return np.mean(plcmos_windowed)