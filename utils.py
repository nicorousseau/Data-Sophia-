import numpy as np
import matplotlib.pyplot as plt
import statsmodels

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

def forecast(train, lags = 128, thresh = 20, crossfade = False, format = np.int16):

    model = statsmodels.tsa.ar_model.AutoReg(train, lags)
    model_fit = model.fit()
    n_train = len(train)
    start = n_train
    clip_value = 1.3*np.max(train)#np.iinfo(format).max
    if not crossfade : 
        end = 2*n_train-1
    elif crossfade : 
        end = 2*n_train+ n_train//2 -1
    
    pred = model_fit.predict(start = start, end = end, dynamic = True)
    print(f"nb depassement : {np.sum(pred > clip_value)}")
    if (np.sum(np.abs(pred) > clip_value) > thresh) and (lags > 16): 
        print(f"try to adapt to {lags//2}")
        new_pred = forecast(train, lags//2, thresh, crossfade, format)
        if np.sum(np.abs(pred) > clip_value) > np.sum(np.abs(new_pred) > clip_value) : 
            return new_pred
        else :
            return pred
    return pred

