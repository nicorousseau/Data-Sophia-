import numpy as np 
import matplotlib.pyplot as plt

class Autoreg() : 

    def __init__(self, data, lags) : 
        self.data = data
        self.len_data = len(data)
        self.coeffs = None
        self.predictions = []
        self.lags = lags

    def fit(self) :
        X = [self.data[self.lags+1-i:self.len_data-i] for i in range (self.lags+1)]
        X = 1/(self.len_data - self.lags) * np.array(X)
        mat_cov = np.dot(X,np.transpose(X))
        Gamma = mat_cov[0,1:]
        mat_cov = mat_cov[:self.lags,:self.lags]
        self.coeffs = np.dot(np.linalg.inv(mat_cov), Gamma)
        return self.coeffs

    def predict(self, n) :
        for i in range (n) : 
            self.predictions.append(np.dot(np.flip(self.coeffs), self.data[-len(self.coeffs):]))
            self.data = np.append(self.data, self.predictions[-1])
        return self.predictions
    
    def plot(self, n) : 
        fig, axs = plt.subplots(3)
        axs[0].plot(self.predict(n))
        axs[0].set_title('predictions')
        axs[1].plot(self.data[:self.len_data])
        axs[1].set_title('data')
        axs[2].plot(self.data)
        axs[2].set_title('data + predictions')
        plt.tight_layout()
        plt.show()


from scipy.io import wavfile
import pickle 

with open('time_series\data.pkl', 'rb') as fichier:
    data = pickle.load(fichier)

sample_rate = data[0][0]
context = 0.05
predict = 0.005
context_size = int(context * sample_rate)
predict_size = int(predict * sample_rate)

#X = data[0][1][2000:4000]


X = np.linspace(0, 110, 2000)
X = np.sin(X/100) + np.sin(2*X/3) + np.sin(X/2)**2 + np.random.normal(0, 0.3, 2000)

model = Autoreg(X, 400)
model.fit()
model.plot(2000)


