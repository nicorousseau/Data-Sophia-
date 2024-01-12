import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

from scipy.io import wavfile



X = np.linspace(0, 10, 100)
X = np.sin(X) + np.sin(2*X) + np.sin(3*X) + np.random.normal(0, 0.1, 100)

model = AutoReg(X, lags=10)
model_fit = model.fit()
print(model_fit.summary())