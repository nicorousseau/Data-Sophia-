import matplotlib.pyplot as plt
import numpy as np
import scipy 
import scipy.io.wavfile as wav
import sklearn.linear_model as linear_model

class Persistance_freq():

    def __init__(self, audio, old_sr, new_sr, nb_freq_kept, train_size):
        
        self.audio = audio
        self.sr = new_sr

        self.freq_kept = None
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

    def _freq_max_sorted(self):
        sample = self.audio[self.pos-self.train_size:self.pos]
        fft_abs = np.abs(np.fft.fft(sample)[:len(sample)//2])
        freq = np.linspace(0, self.sr//2 - 1, len(fft_abs))
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

        time = np.linspace(0,(self.train_size)/self.sr,self.train_size)
        self.sample = self.audio[self.pos-self.train_size:self.pos]

        input = []
        output = []

        for i in range(self.train_size):
            input.append(np.cos(2*np.pi*self.freq_kept*time[i]).tolist() + np.sin(2*np.pi*self.freq_kept*time[i]).tolist())
            output.append([self.sample[i]])
        input = np.array(input)
        label = np.array(output)

        return input, label 

    def fit(self, pos):
        self.pos = pos
        self._freq_max_sorted()
        input, label = self._train_test()
        self.Linear = linear_model.LinearRegression()
        self.Linear.fit(input, label)
        self.coef = self.Linear.coef_
        time = np.linspace(0,(self.train_size)/self.sr,self.train_size)
        self.pred = []
        for i in range (self.train_size):
            vect = np.cos(2*np.pi*self.freq_kept*time[i]).tolist() + np.sin(2*np.pi*self.freq_kept*time[i]).tolist()
            value = np.dot(self.coef, vect)
            self.pred.append(value)

    def predict(self, predict_size):
        self.predict_size = predict_size
        time = np.linspace(self.train_size/self.sr,(self.train_size+self.predict_size)/self.sr,self.predict_size)
        self.pred = []
        self.sample_trunc = self.sample.tolist()
        for i in range (self.predict_size):
            vect = np.cos(2*np.pi*self.freq_kept*time[i]).tolist() + np.sin(2*np.pi*self.freq_kept*time[i]).tolist()
            value = np.dot(self.coef, vect)
            self.sample_trunc.append(value)
            self.pred.append(*value)

    def plot_pred(self):
        plt.plot(self.pred, label = 'pred')
        plt.plot(self.audio[self.pos:self.pos+self.predict_size], label = 'true')
        plt.legend()
        plt.show()

    def plot_coef(self):
        plt.plot(self.coef)
        plt.show()

'''sample_rate, audio_data = wav.read('songs/audio_original.wav')

new_sample_rate = 44100

#audio_data = np.mean(audio_data, axis=1)[:1000000]

nb_freq_kept = 20

train_size = 1280
predict_size = 640

positions = np.random.randint(2000, len(audio_data)-predict_size, 5)
AR_hybrid = Persistance_freq(audio_data, sample_rate, new_sample_rate, nb_freq_kept, train_size)

for pos in positions :
    AR_hybrid.fit(pos=pos)
    AR_hybrid.predict(predict_size)
    print(AR_hybrid.pred)
    print(AR_hybrid.sample)
    plt.plot(np.concatenate((AR_hybrid.sample, AR_hybrid.pred)), label = 'pred', linestyle = '--')
    plt.plot(AR_hybrid.audio[pos-train_size:pos+predict_size], label = 'true')
    plt.show()'''