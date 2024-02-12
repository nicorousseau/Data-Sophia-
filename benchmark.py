import numpy as np 
from librosa.feature import melspectrogram
from librosa.display import specshow
import matplotlib.pyplot as plt
def nmse(y, y_predicted) :
    diff = y - y_predicted
    norm_diff = np.square(diff).sum()
    norm_y = np.square(y).sum()
    try : 
        res = 10*np.log10(norm_diff/norm_y)
    except : 
        print(f"norm_diff/norm_y = {norm_diff/norm_y}")
    return res

def nmse_mean(audio, audio_filled, pos_gap, taille_paquet):
    list_nmse = []
    for k in range(len(pos_gap)) : 
        original = audio[pos_gap[k]: pos_gap[k]+taille_paquet]
        predicted = audio_filled[pos_gap[k]: pos_gap[k]+taille_paquet]
        list_nmse.append(nmse(original, predicted))
    
    return np.mean(list_nmse), list_nmse
    

def mel_cs(y, y_predicted, sr) : 
    y = np.array(y)
    y_predicted = np.array(y_predicted)
    #y = y.astype(np.float64)
    #y = y/np.iinfo(np.int16).max
    #y_predicted =y_predicted.astype(np.float64)
    #y_predicted = y_predicted/np.iinfo(np.int16).max
    sample_rate = sr
    win_length = 512
    n_fft = 1024#int(np.floor(np.log2(len(y)))) = 9 dans notre cas#
    hop_length = 256
    y_spec = melspectrogram(y=y, sr=sample_rate, S= None, n_fft=n_fft, hop_length= hop_length, win_length=win_length, window='hann', center=True, pad_mode='constant', power=2.0, n_mels = 64)
    y_pred_spec = melspectrogram(y=y_predicted, sr=sample_rate, S= None, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', center=True, pad_mode='constant', power=2.0, n_mels = 64)
    diff = np.linalg.norm(y_spec-y_pred_spec, ord='fro')
    #plt.figure(figsize=(9, 4), dpi=80)
    #specshow(y_spec-y_pred_spec, hop_length = hop_length, x_axis='time', y_axis='log')
    #fig, [ax1, ax2] = plt.subplots(nrows = 1, ncols=2)
    #img_1 = specshow(y_spec, hop_length = hop_length, x_axis='time', y_axis='log', ax=ax1)
    #fig.colorbar(img_1, ax=ax1, format="%+2.f dB")
    #img_2 = specshow(y_pred_spec, hop_length = hop_length, x_axis='time', y_axis='log', ax=ax2)
    #fig.colorbar(img_2, ax=ax2, format="%+2.f dB")
    #plt.show()

    return diff/np.linalg.norm(y_spec, ord = 'fro')
    #https://librosa.org/doc/0.10.1/generated/librosa.filters.mel.html#librosa.filters.mel
    
#ondelettes ? 
    