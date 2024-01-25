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
import scipy 
import math


#Definition of the constants
SAMPLE_RATE = 44100
lambda1 = 0.995
lambda2 = 0.005
epochs = 1
n = 20

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
        input_waveform = input_waveform[0]

        # Load label audio file
        label_waveform, label_sr = torchaudio.load(self.file_paths[index]['label_audio'])
        label_waveform = label_waveform[0]

        # Apply any necessary transformations
        if self.transform1:
            input_waveform = self.transform1(input_waveform)
            label_waveform = self.transform1(label_waveform)
        
        if self.transform2:
            input_waveform = self.transform2(input_waveform)
            label_waveform = self.transform2(label_waveform)

        if self.transform3:
            input_fft = self.transform3(input_waveform)
        
        if self.transform4:
            label_fft = self.transform4(label_waveform)

        print(input_fft.shape)

        return input_fft, label_fft

directory = "dataset"
file_list = [
    {'input_audio': os.path.join(directory, f'input_audio_{i}.wav'), 'label_audio': os.path.join(directory, f'label_audio_{i}.wav')} for i in range(200)
]

def convert_to_mono(audio_waveform):
    if audio_waveform.shape[0] > 1:
        audio_waveform = torch.mean(audio_waveform, dim=0, keepdim=True)
    return audio_waveform

def fft(audio_waveform):
    fft = torch.fft.fft(audio_waveform)
    freq = torch.fft.fftfreq(len(audio_waveform), d=1/SAMPLE_RATE)
    input = torch.cat((freq, fft.abs(), fft.angle()), dim=0)
    return input


def freq_max_sorted(audio_waveform):
    fft = torch.fft.fft(audio_waveform) 
    freq = torch.fft.fftfreq(len(audio_waveform), d=1/SAMPLE_RATE)

    fft_abs = torch.abs(fft)

    r = fft_abs[1:]
    l = fft_abs[:-1]
    mask_l = r > l
    mask_r = l > r
    mask = mask_l[:-1] * mask_r[1:]
    mask = torch.cat((torch.tensor([True]), mask, torch.tensor([False])))

    list_max = torch.zeros(len(fft_abs))
    list_max[mask] = fft_abs[mask]
    ind = torch.argsort(list_max, descending=True)[:n]

    label = torch.cat((freq[ind], fft[ind].abs(), fft[ind].arg()), dim=0)
    return label

transform1 = torchaudio.transforms.Resample(orig_freq=SAMPLE_RATE, new_freq=SAMPLE_RATE)
transform2 = convert_to_mono
transform3 = fft
transform4 = freq_max_sorted

# Creation of the dataset
custom_dataset = CustomDataset(file_list, transform1, transform2, transform3, transform4)

# Split the dataset into train and test
train_ratio = 0.8
train_size = int(train_ratio * len(custom_dataset))
test_size = len(custom_dataset) - train_size

train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])

# Creation of the DataLoader for train and test datasets
batch_size = 16
shuffle = True

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

print(train_loader)
# Define the network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.sequential = nn.Sequential(
        nn.Linear(372, 372), 
        nn.ReLU(),
        nn.Linear(372, 186),
        nn.ReLU(),
        nn.Linear(186, 186),
        nn.Sigmoid(),
        nn.Linear(186, 93),
        nn.Sigmoid(),
        nn.Linear(93, 93))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.sequential(x)
        return logits
    
net = Net()

# Define a Loss function and optimizer
loss = nn.MSELoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train the network
for i in range (epochs):  # loop over the dataset multiple times

    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)

        # Calcul de la loss
        loss = loss(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 50 == 49:    # print every 100 mini-batches
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

    # Append the predicted and actual labels to their respective lists
    predicted_labels.append(outputs)
    actual_labels.append(labels)
    actuals_train.append(inputs)
