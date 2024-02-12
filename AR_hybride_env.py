import matplotlib.pyplot as plt
import numpy as np
import scipy 
import scipy.io.wavfile as wav
import sklearn.linear_model as lm

class AR_freq_env():

    def __init__(self, audio, old_sr, new_sr, window_size, nb_freq_kept, train_size):

        self.audio = audio
        self.sr = new_sr

        self.window_size = window_size
        self.freq_kept = None
        self.nb_freq_kept = nb_freq_kept

        self.env = None

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

        self._env()

    def _freq_max_sorted(self):
        fft_abs = np.abs(np.fft.fft(self.sample)[:len(self.sample)//2])
        freq = np.linspace(0, self.sr//2 - 1, len(fft_abs))
        r = fft_abs[1:]
        l = fft_abs[:-1]
        mask_l = r > l
        mask_r = l > r
        mask = mask_l[:-1] * mask_r[1:]
        mask = np.concatenate((np.array([True]), mask, np.array([False])))
        list_max = np.zeros(len(fft_abs))
        list_max[mask] = fft_abs[mask]
        ind = np.argsort(list_max)[-self.nb_freq_kept:]
        self.freq_kept = freq[ind]
        
    def _train_test(self):
        time = np.linspace(0,(self.train_size-1)/self.sr,self.train_size)

        input_freq = []
        label_freq = []

        for i in range(self.train_size):
            input_freq.append(np.cos(2*np.pi*self.freq_kept*time[i]).tolist() + np.sin(2*np.pi*self.freq_kept*time[i]).tolist())
            label_freq.append([self.sample[i]])
        
        input_env = time[self.train_size-50:self.train_size].tolist()
        label_env_pos = self.env[1][self.pos-50:self.pos].tolist()
        label_env_neg = self.env[0][self.pos-50:self.pos].tolist()

        input_freq = np.array(input_freq)
        label_freq = np.array(label_freq)
        input_env_pos = np.array(input_env).reshape(-1,1)
        label_env_pos = np.array(label_env_pos).reshape(-1,1)
        input_env_neg = np.array(input_env).reshape(-1,1)
        label_env_neg = np.array(label_env_neg).reshape(-1,1)

        return input_freq, label_freq, input_env_pos, label_env_pos, input_env_neg, label_env_neg
    
    def _env(self):
        window_size = 300
        audio_pad = np.pad(self.audio, (window_size//2, window_size//2), mode='edge')

        env_max = [np.max(audio_pad[i-window_size//2:i+window_size//2]) for i in range(window_size//2, len(audio_pad)-window_size//2)]
        env_max = np.convolve(env_max, np.ones((window_size))/window_size, mode='same')
        
        env_min = [np.min(audio_pad[i-window_size//2:i+window_size//2]) for i in range(window_size//2, len(audio_pad)-window_size//2)]
        env_min = np.convolve(env_min, np.ones((window_size))/window_size, mode='same')
        env_min = np.abs(env_min)

        self.env = [env_min, env_max]


    def _normalize(self):
        positive = self.sample > 0
        negative = self.sample < 0

        self.sample_copy = np.zeros(len(self.sample))

        self.sample_copy[positive] = self.sample[positive]/np.array(self.env[1][self.pos-self.train_size: self.pos])[positive]
        self.sample_copy[negative] = self.sample[negative]/np.array(self.env[0][self.pos-self.train_size: self.pos])[negative]
        self.sample = np.copy(self.sample_copy)


    def fit(self, pos, alpha, l1_ratio):
        self.pos = pos
        self.sample = self.audio[self.pos-self.train_size: self.pos]
        self._normalize()
        self._freq_max_sorted()

        input_freq, label_freq, input_env_pos, label_env_pos, input_env_neg, label_env_neg = self._train_test()

        self.ElasticNet_freq = lm.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        self.ElasticNet_freq.fit(input_freq, label_freq)

        self.coef_freq = self.ElasticNet_freq.coef_

        self.ElasticNet_env_pos = lm.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        self.ElasticNet_env_pos.fit(input_env_pos, label_env_pos)

        self.coef_env_pos = self.ElasticNet_env_pos.coef_

        self.ElasticNet_env_neg = lm.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        self.ElasticNet_env_neg.fit(input_env_neg, label_env_neg)

        self.coef_env_neg = self.ElasticNet_env_pos.coef_

    def predict(self, predict_size):
        self.predict_size = predict_size
        time = np.linspace((self.train_size-10)/self.sr,(self.train_size+predict_size-1-10)/self.sr,predict_size)
        self.pred = []
        self.sample = self.sample.tolist()
        for i in range (predict_size):
            freq_vect = np.cos(2*np.pi*self.freq_kept*time[i]).tolist() + np.sin(2*np.pi*self.freq_kept*time[i]).tolist()
            freq_value = np.dot(self.coef_freq, freq_vect)
            self.pred.append(freq_value)

        self.pred = np.array(self.pred)

        self.env_pos_predict = self.env[1][self.pos-1] + (self.coef_env_pos) * np.linspace(0, predict_size/self.sr, predict_size)
        mask_pos = self.env_pos_predict < 0
        self.env_pos_predict[mask_pos] = 0

        self.env_neg_predict = self.env[0][self.pos-1] + (self.coef_env_neg) * np.linspace(0, predict_size/self.sr, predict_size)
        mask_neg = self.env_neg_predict < 0
        self.env_pos_predict[mask_neg] = 0

        mask_pred_pos = np.array(self.pred) > 0
        mask_pred_neg = np.array(self.pred) < 0

        self.pred[mask_pred_pos] = self.env_pos_predict[mask_pred_pos] * np.array(self.pred[mask_pred_pos])
        self.pred[mask_pred_neg] = self.env_neg_predict[mask_pred_neg] * np.array(self.pred[mask_pred_neg])

    def plot_pred(self):
        plt.plot(self.pred, label = 'pred')
        plt.plot(self.audio[self.pos:self.pos+self.predict_size], label = 'true')
        plt.plot(self.env[-self.predict_size:])
        plt.legend()
        plt.show()

    def plot_coef(self):
        _, axs = plt.subplots[2]
        axs[0].plot(self.coef_env)
        axs[1].plot(self.coef_freq)
        plt.show()

'''sample_rate, audio_data = wav.read('songs/audio_original.wav')

new_sample_rate = 32000

window_length = 0.1
window_size = int(window_length * new_sample_rate)

nb_freq_kept = 200

train_size = 1280
predict_size = 640

AR = AR_freq_env(audio_data, sample_rate, new_sample_rate, window_size, nb_freq_kept, train_size)

positions = np.random.randint(train_size, len(audio_data)-predict_size, 2)

for pos in positions : 

    AR.fit(pos=pos, alpha = 0, l1_ratio = 0.7)
    AR.predict(predict_size)

    #plt.plot(AR.env[pos-train_size:pos+predict_size])
    #plt.plot(np.concatenate((AR.env[1][pos-train_size:pos],AR.env_pos_predict)), linestyle = '--', color = 'r')
    plt.plot(np.concatenate((AR.audio[pos-train_size:pos],AR.pred)), label = 'pred', linestyle = '--', color = 'r')
    plt.plot(AR.audio[pos-train_size:pos+predict_size], label = 'true', color = 'g')
    plt.legend()
    plt.show()'''
