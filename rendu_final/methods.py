import numpy as np
from sklearn.linear_model import LinearRegression
#import sklearn.linear_model as lm
from scipy.signal import resample
from statsmodels.tsa.ar_model import AutoReg

import numpy.fft as fft
import utils





#----------------------------------Baselines
def filling_silence_persistance(audio, pos_gap, taille_paquet) :
    audio_persistance = audio.copy()
    audio_gap_filled = audio.copy()

    for x in pos_gap : 
        persistant = np.copy(audio[x-taille_paquet:x])

        audio_persistance[x:x+taille_paquet] = persistant
        audio_gap_filled[x:x+taille_paquet] = np.zeros(taille_paquet)
        
    return audio_gap_filled, audio_persistance
#----------------------------------AR adaptatif
def forecast_adapt(train_data, n_pred, lags = 128, n_thresh = 1, clip_value = 1.3, crossfade_size = None):

    model = AutoReg(train_data, lags)
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
        new_pred = forecast_adapt(train_data, n_pred, lags//2, n_thresh = n_thresh, crossfade_size = crossfade_size)
        if np.sum(np.abs(pred) > clip_value) > np.sum(np.abs(new_pred) > clip_value) : 
            return new_pred
        else :
            return pred
    return pred 

#----------------------------------Presistance fréquentielle
def freq_persistance(audio, pos_gap, taille_paquet, sample_rate, n_harm = 15, crossfade_size = None):
    audio_corr = audio.copy()
    temps_paquet = taille_paquet/sample_rate
    taille_context = 1
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
        kept_ind, max_kept = utils.keeping_important_freq(fft_abs, 2*n_harm)
        freq_kept = freq[kept_ind]
        fft_kept = fft_signal[kept_ind]
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
            audio_corr[pos:pos+taille_paquet] = pred

    return audio_corr
#----------------------------------Process sur l'enveloppe


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

def env_predictions(audio_norm, env_max, env_min, pos_gap, taille_paquet, order = 128, train_size = None, adapt = False, clip_value = 1.3, crossfade_size = None):

    norm_to_fill = audio_norm.copy()
    
    norm = utils.audio_predictions(norm_to_fill, pos_gap, taille_paquet, order=order, train_size = train_size, adapt = adapt)
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


#----------------------------------AR Hybride 
class AR_freq():

    def __init__(self, audio, old_sr, new_sr, nb_lags, nb_freq_kept, train_size):

        self.audio = audio
        self.sr = new_sr

        self.nb_lags = nb_lags

        self.freq_kept = None
        self.nb_freq_kept = nb_freq_kept

        self.coef = []

        self.sample_trunc = None

        self.train_size = train_size
        self.predict_size = None

        self.pred = None

        self.pos = None

        self.diverge = None
    
    def _train_test(self):

        time = np.linspace(0,(self.train_size-1)/self.sr,self.train_size)
        self.sample = self.audio[self.pos-self.nb_lags-self.train_size:self.pos]
        self.freq_kept, self.moy = utils.freq_max_sorted(self.audio, self.pos, self.train_size, self.sr, self.nb_freq_kept) #on utilise ensuite dans predict

        input = []
        output = []

        for i in range(self.train_size):
            input.append(self.sample[i:i+self.nb_lags].tolist() + np.cos(2*np.pi*self.freq_kept*time[i]).tolist() + np.sin(2*np.pi*self.freq_kept*time[i]).tolist())
            output.append([self.sample[i+self.nb_lags]])
        input = np.array(input)
        label = np.array(output)

        return input, label 

    def fit(self, pos):
        self.pos = pos
        input, label = self._train_test()
        self.lineareg = LinearRegression()
        self.lineareg.fit(input, label)
        self.coef = self.lineareg.coef_[0]

    def predict(self, predict_size):
        self.predict_size = predict_size
        time = np.linspace(self.train_size/self.sr,(self.train_size+predict_size-1)/self.sr,predict_size)
        self.pred = []
        self.sample_trunc = self.sample.tolist()
        for i in range (predict_size):
            if self.nb_lags == 0 : 
                vect = np.cos(2*np.pi*self.freq_kept*time[i]).tolist() + np.sin(2*np.pi*self.freq_kept*time[i]).tolist()
            else : 
                vect = self.sample_trunc[-self.nb_lags:] + np.cos(2*np.pi*self.freq_kept*time[i]).tolist() + np.sin(2*np.pi*self.freq_kept*time[i]).tolist()
            value = np.dot(self.coef, vect)
            self.sample_trunc.append(value)
            self.pred.append(value)
        
        self.pred = np.array(self.pred) - self.moy
        if np.max(np.abs(self.pred)) > 1.5 * np.max(np.abs(self.sample)):
            self.diverge = True 
        else :
            self.diverge = False
        
    def remove_freq(self,): 
        train = self.audio[self.pos-self.train_size:self.pos]
        only_freq = []
        time = np.linspace(0,(self.train_size-1)/self.sr,self.train_size)
        for i in range(self.train_size):
            vect = np.cos(2*np.pi*self.freq_kept*time[i]).tolist() + np.sin(2*np.pi*self.freq_kept*time[i]).tolist()
            value = np.dot(self.coef, vect)
            only_freq.append(value)
        return train - only_freq

