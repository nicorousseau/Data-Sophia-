import numpy as np 
import matplotlib.pyplot as plt

class Autoreg() : 

    def __init__(self, data, lags) : 
        self.data = data
        self.len_data = len(data)
        self.coeffs = None
        self.lags = lags

    def fit(self) :
        X = [self.data[self.lags+1-i:self.len_data-i] for i in range (self.lags+1)]
        X = 1/(self.len_data - self.lags) * np.array(X)
        mat_cov = np.dot(X,np.transpose(X))
        Gamma = mat_cov[0,1:]
        mat_cov = mat_cov[:self.lags,:self.lags]
        pinv = np.linalg.pinv(mat_cov, rcond=1e-4)
        self.coeffs = np.dot(pinv, Gamma)
        return self.coeffs

    def predict(self, n) :
        predictions = []
        for i in range (n) : 
            predictions.append(np.dot(np.flip(self.coeffs), self.data[-len(self.coeffs):]))
            self.data = np.append(self.data, self.predictions[-1])
        return predictions
    
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


