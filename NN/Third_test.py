import torchaudio 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as sc
from statsmodels.tsa.ar_model import AutoReg


#Definition of the constants
SAMPLE_RATE = 44100
epochs = 20
batch_size = 10
train_len = 0.1
test_len = 0.02
train_size = int(train_len * SAMPLE_RATE//2)
test_size = int(test_len * SAMPLE_RATE//2)
nb_lags = 500

class CustomDataset(Dataset):
    def __init__(self, file_paths, transform1=None, transform2 = None, transform3 = None, transform4 = None):
        self.file_paths = file_paths
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.transform4 = transform4

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # Load input audio file
        input_waveform, input_sr = torchaudio.load(self.file_paths[index]['input_audio'])

        # Load label audio file
        label_waveform, label_sr = torchaudio.load(self.file_paths[index]['label_audio'])
        
        # Apply any necessary transformations
        if self.transform1:
            input_waveform = self.transform1(input_waveform)
            label_waveform = self.transform1(label_waveform)
   
        if self.transform2:
            input_waveform = self.transform2(input_waveform)
            label_waveform = self.transform2(label_waveform)

        return input_waveform, label_waveform

directory = "dataset"
file_list = [
    {'input_audio': os.path.join(directory, f'input_audio_{i}.wav'), 'label_audio': os.path.join(directory, f'label_audio_{i}.wav')} for i in range(900)
]

def convert_to_mono(audio_waveform):
    if audio_waveform.size()[0] > 1:
        audio_waveform = torch.mean(audio_waveform, dim=0, keepdim=True)
    return audio_waveform[0]

transform1 = torchaudio.transforms.Resample(orig_freq=SAMPLE_RATE, new_freq=SAMPLE_RATE//2)
transform2 = convert_to_mono

# Creation of the dataset
custom_dataset = CustomDataset(file_list, transform1, transform2)

# Split the dataset into train and test
train_ratio = 0.8
nb_train_data = int(train_ratio * len(custom_dataset))
nb_test_data = len(custom_dataset) - nb_train_data

train_dataset, test_dataset = random_split(custom_dataset, [nb_train_data, nb_test_data])

# Creation of the DataLoader for train and test datasets
shuffle = True

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

# Define the network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.sequential = nn.Sequential(
        nn.Linear(train_size, train_size//2), 
        nn.ReLU(),
        nn.Linear(train_size//2, train_size//4),
        nn.ReLU(),
        nn.Linear(train_size//4, train_size//8),
        nn.Sigmoid(),
        nn.Linear(train_size//8, nb_lags))

    def forward(self, x):
        logits = self.sequential(x)
        return logits

'''def AR(params, train): 
    pred = []
    params = params.detach().numpy()
    div = torch.tensor([100000 for i in range(test_size)])
    for i in range (len(params)):
        lags = params[i][0]
        train_size = params[i][1]
        if lags > 0 and train_size > 0 and lags < train_size:
            model = AutoReg(train[i][-train_size:], lags=lags)
            model_fit = model.fit()
            pred.append(torch.tensor(model_fit.predict(start = train_size, end = train_size + test_size - 1), requires_grad=False))
        else : 
            pred.append(div)
    pred = torch.stack(pred)
    return pred'''

# Define a Loss function and optimizer
net = Net()
loss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.002)

def predict(params, train):
    pred = torch.clone(train).requires_grad_()
    params = params.detach()
    for _ in range (test_size):
        pred = torch.cat((pred, torch.inner(params,pred[:, -nb_lags:]).diagonal(0).unsqueeze(1)), dim = 1)
    return pred[:, -test_size:]

# Train the network
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()

        params = net(inputs)
        
        pred = predict(params, inputs)

        # Calcul de la loss
        batch_loss = loss(pred, labels)  # Use a different variable name for the loss value
        batch_loss.backward()
        optimizer.step()

        running_loss += batch_loss.item()

        if i % 50 == 49:    # print every 100 mini-batches
            print(params)
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')


# Create empty lists to store the predicted and actual labels
predicted_labels = []
actual_labels = []
actuals_train = []

# Iterate over the test dataset
for i, data in enumerate(test_loader):
    inputs, labels = data

    # Forward pass
    outputs = net(inputs)

    predictions = predict(outputs, inputs)

    inputs_np = inputs.detach()
    labels_np = labels.detach()
    predictions_np = predictions.detach()

    # Append the predicted and actual labels to their respective lists
    predicted_labels.append(predictions_np)
    actual_labels.append(labels_np)
    actuals_train.append(inputs_np)

for i in range (3):
    fig, axs = plt.subplots(2)

    axs[0].plot(actuals_train[i][0])
    axs[0].set_title('Train')
    axs[1].plot(actual_labels[i][0])
    axs[1].plot(predicted_labels[i][0], linestyle = '--')
    axs[1].set_title('Test')
    plt.show()