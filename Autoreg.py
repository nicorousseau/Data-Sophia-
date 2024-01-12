import numpy as np 
import matplotlib.pyplot as plt 

class Autoreg() : 
    """This class allows you to make a simple linear autoregression"""

    def __init__(self, data) : 
        self.data = data
        self.len_data = len(data)
        self.coeffs = None
        self.data_and_predictions = None 
        self.predictions = None

    def fit(self, train_size, lags) :
        """
        This function fits the model to the data_train
        Input : train_size (length of the sequence we want to train our model with)
                lags (order of the autoregression)
        
        This function defines the coefficients of the autoregression.
        """
        self.coeffs = None
        if self.len_data < train_size:  
            raise ValueError('len_data must be bigger than train_size')
        elif lags >= train_size : 
            raise ValueError('lags must be stricly smaller than train_size')
        else :
            data_train = self.data[-train_size:]
            X = [data_train[lags+1-i:train_size-i] for i in range (lags+1)]
            X = 1/(train_size - lags) * np.array(X)
            mat_cov = np.dot(X,np.transpose(X))
            Gamma = mat_cov[0,1:]
            mat_cov = mat_cov[:lags,:lags]
            pinv = np.linalg.pinv(mat_cov, rcond=1e-4)
            self.coeffs = np.dot(pinv, Gamma)

    def predict(self, predict_size) :
        """
        Once the model is fitted to the data_train, 
        this function predicts the following values of the time series. 
        Input : predict_size (length of the prediction you want)

        This function doesn't return anything but affects the self.predictions.
        """
        self.predictions = []
        self.data_and_predictions = np.copy(self.data)
        for i in range (predict_size) : 
            self.predictions = np.append(self.predictions, (np.dot(np.flip(self.coeffs), self.data_and_predictions[-len(self.coeffs):])))
            self.data_and_predictions = np.append(self.data_and_predictions, self.predictions[-1])
    
    def plot(self, predict_size) : 
        """This function plots the results of the autoregression. 
        Input : predict_size (length of the prediction you want to plot)
        Output : nothing, but it plots the data we used to fit the model, 
        the prediction, and the concatenation of both of them.

        This function pedicts itself the values of the prediction, 
        it's not necessary to apply the predict function.
        """
        self.predict(predict_size)
        fig, axs = plt.subplots(3)
        axs[0].plot(self.data)
        axs[0].set_title('data')
        axs[1].plot(self.predictions)
        axs[1].set_title('predictions')
        axs[2].plot(self.data_and_predictions)
        axs[2].set_title('data + predictions')
        plt.tight_layout()
        plt.show()


def RMSE(list1, list2) : 
    """Return the RMSE of two lists"""
    return np.sqrt(np.mean(np.square(list1 - list2)))

def AME(list1, list2) :
    """Return the AME of two lists"""
    return np.mean(np.abs(list1 - list2))

def p_value(list1, list2) : 
    """Return the p-value of two lists"""
    return np.mean(np.abs(list1 - list2)) / np.mean(list1)