import matplotlib.pyplot as plt
import numpy as np
import scipy 
import scipy.io.wavfile as wav
import sklearn.linear_model as lm

class AR_freq:

    def __init__(self, audio, old_sr, new_sr, nb_lags, window_size, nb_freq_kept, train_size):

        self.audio = audio
        self.sr = new_sr

        self.nb_lags = nb_lags

        self.window_size = window_size
        self.freqkept = None
        self.nb_freq_kept = nb_freq_kept

        self.coef = []

        self.sample_trunc = None

        self.train_size = train_size
        self.predict_size = None

        self.pred = None

        self.pos = None

        #### On resample l'audio à 16kHz ####
        if old_sr != new_sr:
            self.audio = scipy.signal.resample(audio, num = int(new_sr/old_sr * len(audio)))

        #### On met l'audio en mono ####
        if len(self.audio.shape) == 2:
            self.audio = np.mean(self.audio, axis=1)

    def _freq_max_sorted(self, fft):
        fft_abs = np.abs(fft)
        r = fft_abs[1:]
        l = fft_abs[:-1]
        mask_l = r > l
        mask_r = l > r
        mask = mask_l[:-1] * mask_r[1:]
        mask = np.concatenate((np.array([True]), mask, np.array([False])))
        list_max = np.zeros(len(fft_abs))
        list_max[mask] = fft_abs[mask]
        ind = np.argsort(list_max)
        fft[ind[:-self.nb_freq_kept]] = 0
        return fft, ind[-self.nb_freq_kept:]
    
    def _freq_kept(self):
        f, t, stft = scipy.signal.stft(self.audio, fs=self.sr, nperseg = self.window_size, window=np.hamming(self.window_size), noverlap=self.window_size//2, nfft = self.window_size)
        
        self.f = f
        self.t = t
        self.stft = stft

        fft_trunc = [self._freq_max_sorted(stft)[0] for stft in stft.T]
        indexes = [self._freq_max_sorted(stft)[1] for stft in stft.T]

        self.stft_trunc = np.array(fft_trunc).T

        freq_kept = [f[ind] for ind in indexes]
        freq_kept = np.array(freq_kept) 
        freq_kept = freq_kept.flatten()
        freq_kept = freq_kept.astype(int)

        freq_kept= np.unique(freq_kept, return_counts=True)
        self.freq_kept = freq_kept[0][freq_kept[1] > 10]
    
    def _train_test(self, pos):
        self.pos = pos
        self._freq_kept()

        time = np.linspace(0,(self.train_size)*self.sr,self.train_size)

        self.sample = self.audio[pos-self.nb_lags:pos+self.train_size]
        input = []
        output = []

        for i in range(self.train_size):
            input.append(self.sample[i:i+self.nb_lags].tolist() + np.cos(2*np.pi*self.freq_kept*time[i]).tolist() + np.sin(2*np.pi*self.freq_kept*time[i]).tolist())
            output.append(self.sample[i+self.nb_lags])
        input = np.array(input)
        label = np.array(output)
        return input, label 

    def fit(self, pos):
        input, label = self._train_test(pos)
        self.ElasticNet = lm.ElasticNet(alpha=0.1, l1_ratio=0.9)
        self.ElasticNet.fit(input, label)
        self.coef = self.ElasticNet.coef_

    def predict(self, predict_size):
        self.predict_size = predict_size
        time = np.linspace(self.train_size,(self.train_size+predict_size)*self.sr,predict_size)
        self.pred = []
        self.sample_trunc = self.sample.tolist()
        for i in range (predict_size):
            vect = self.sample_trunc[-self.nb_lags:] + np.cos(2*np.pi*self.freq_kept*time[i]).tolist() + np.sin(2*np.pi*self.freq_kept*time[i]).tolist()
            value = np.dot(self.coef, vect)
            self.sample_trunc.append(value)
            self.pred.append(value)

    def plot_pred(self):
        plt.plot(self.pred, label = 'pred')
        plt.plot(self.audio[self.pos+self.train_size:self.pos+self.train_size+self.predict_size], label = 'true')
        plt.legend()
        plt.show()

    def plot_scatter(self):
        plt.pcolormesh(self.t, self.f, np.log10(np.abs(self.stft_trunc)+1))
        plt.show()

    def plot_coef(self):
        plt.plot(self.coef)
        plt.show()


sample_rate, audio_data = wav.read('songs/audio_original.wav')

new_sample_rate = 16000

window_length = 0.1
window_size = int(window_length * new_sample_rate)

nb_lags = 100

nb_freq_kept = 10

train_size = 1000

'''AR = AR_freq(audio_data, sample_rate, new_sample_rate, nb_lags, window_size, nb_freq_kept, train_size)
AR.fit(10000)
AR.predict(100)
AR.plot_pred()
AR.plot_scatter()
AR.plot_coef()'''
