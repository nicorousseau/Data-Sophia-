import matplotlib.pyplot as plt
import numpy as np
import scipy 
import scipy.io.wavfile as wav
import sklearn.linear_model as lm

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

        #### On resample l'audio à 16kHz ####
        if old_sr != new_sr:
            self.audio = scipy.signal.resample(audio, num = int(new_sr/old_sr * len(audio)))

        #### On met l'audio en mono ####
        if len(self.audio.shape) == 2:
            self.audio = np.mean(self.audio, axis=1)

        #self.audio = self.audio/np.max(np.abs(self.audio))

    def _freq_max_sorted(self):
        sample = self.audio[self.pos-self.train_size:self.pos]
        sample = np.pad(sample * np.hamming(self.train_size), (2000, 2000), 'constant')
        fft_abs = np.abs(np.fft.fft(sample)[:len(sample)//2])
        freq = np.linspace(0,self.sr/2 - self.sr/len(fft_abs),len(fft_abs))
        r = fft_abs[1:]
        l = fft_abs[:-1]
        mask_l = r > l
        mask_r = l > r
        mask = mask_l[:-1] * mask_r[1:]
        mask = np.concatenate((np.array([False]), mask, np.array([False])))
        list_max = np.zeros(len(fft_abs))
        list_max[mask] = fft_abs[mask]
        ind = np.argsort(list_max)[-self.nb_freq_kept:]
        self.freq_kept = freq[ind]
        
    def _train_test(self):

        time = np.linspace(0,(self.train_size-1)/self.sr,self.train_size)
        self.sample = self.audio[self.pos-self.nb_lags-self.train_size:self.pos]

        self._freq_max_sorted()

        input = []
        output = []

        for i in range(self.train_size):
            input.append(self.sample[i:i+self.nb_lags].tolist() + np.cos(2*np.pi*self.freq_kept*time[i]).tolist() + np.sin(2*np.pi*self.freq_kept*time[i]).tolist())
            output.append([self.sample[i+self.nb_lags]])

        print(len(input[0]))
        input = np.array(input)
        label = np.array(output)

        return input, label 

    def fit(self, pos):
        self.pos = pos
        input, label = self._train_test()
        self.ElasticNet = lm.LinearRegression()
        self.ElasticNet.fit(input, label)
        self.coef = self.ElasticNet.coef_[0]

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
        if np.max(np.abs(self.pred)) > 1.5 * np.max(np.abs(self.sample)):
            self.diverge = True 
        else :
            self.diverge = False

    def plot_pred(self):
        plt.plot(self.pred, label = 'pred')
        plt.plot(self.audio[self.pos:self.pos+self.predict_size], label = 'true')
        plt.legend()
        plt.show()

    def plot_coef(self):
        _, axs = plt.subplots(2)
        axs[0].plot(self.coef[:self.nb_lags], label = 'AR')
        axs[1].plot(self.coef[self.nb_lags:], label = 'freq')
        plt.show()

sample_rate, audio_data = wav.read('songs/audio_original.wav')

new_sample_rate = 32000

#audio_data = np.mean(audio_data, axis=1)[:1000000]

nb_lags = 100

nb_freq_kept = 16

train_size = 960
predict_size = 640

positions = np.random.randint(2000, len(audio_data)-predict_size, 5)
AR_hybrid = AR_freq(audio_data, sample_rate, new_sample_rate, nb_lags, nb_freq_kept, train_size)

for pos in positions :
    AR_hybrid.fit(pos=pos)
    AR_hybrid.predict(predict_size)
    plt.plot(np.linspace(0, (train_size+predict_size-1)/new_sample_rate, train_size+predict_size),AR_hybrid.sample_trunc[nb_lags:], linestyle = '--')
    plt.plot(np.linspace(0, (train_size+predict_size-1)/new_sample_rate, train_size+predict_size),AR_hybrid.audio[pos-train_size:pos+predict_size])
    plt.show()
