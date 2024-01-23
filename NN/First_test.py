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

#Definition of the constants
SAMPLE_RATE = 32000
lambda1 = 0.995
lambda2 = 0.005

class CustomAudioDataset(Dataset):
    def __init__(self, file_paths, transform1=None, transform2 = None):
        self.file_paths = file_paths
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # Load input audio file
        input_waveform, _ = torchaudio.load(self.file_paths[index]['input_audio'])

        # Load label audio file
        label_waveform, _ = torchaudio.load(self.file_paths[index]['label_audio'])

        # Apply any necessary transformations
        if self.transform1:
            input_waveform = self.transform1(input_waveform)
            label_waveform = self.transform1(label_waveform)
        
        if self.transform2:
            input_waveform = self.transform2(input_waveform)
            label_waveform = self.transform2(label_waveform)

        return *input_waveform, *label_waveform

directory = "dataset"
file_list = [
    {'input_audio': os.path.join(directory, f'input_audio_{i}.wav'), 'label_audio': os.path.join(directory, f'label_audio_{i}.wav')} for i in range(20000)
]

def convert_to_mono(audio_waveform):
    if audio_waveform.shape[0] > 1:
        audio_waveform = torch.mean(audio_waveform, dim=0, keepdim=True)
    return audio_waveform

transform1 = torchaudio.transforms.Resample(orig_freq=44100, new_freq=SAMPLE_RATE)
transform2 = convert_to_mono

# Creation of the dataset
custom_audio_dataset = CustomAudioDataset(file_list, transform1, transform2)

print(custom_audio_dataset[0][1].size())

# Split the dataset into train and test
train_ratio = 0.8
train_size = int(train_ratio * len(custom_audio_dataset))
test_size = len(custom_audio_dataset) - train_size

train_dataset, test_dataset = random_split(custom_audio_dataset, [train_size, test_size])

# Creation of the DataLoader for train and test datasets
batch_size = 16
shuffle = True

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

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
loss1 = nn.MSELoss(reduction='mean')
loss2 = nn.MSELoss(reduction='mean')

optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train the network
for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)



        # Conversion des tenseurs PyTorch en tableaux NumPy
        outputs_np = torch.concat((inputs, outputs), dim = 1).detach().numpy()
        labels_np = torch.concat((inputs, labels), dim = 1).detach().numpy()

        # Calcul des spectrogrammes
        spectrogram_outputs = np.array([sc.spectrogram(outputs_np[i], fs=SAMPLE_RATE, window = 'tukey', nperseg = 256, noverlap = 2, nfft = 512, mode = 'magnitude')[2] for i in range(len(outputs_np))])
        spectrogram_labels = np.array([sc.spectrogram(labels_np[i], fs=SAMPLE_RATE, window = 'tukey', nperseg = 256, noverlap = 2, nfft = 512, mode = 'magnitude')[2] for i in range(len(labels_np))])

        # Conversion des tableaux NumPy en tenseurs PyTorch
        spectrogram_outputs = torch.tensor(spectrogram_outputs)
        spectrogram_labels = torch.tensor(spectrogram_labels)

        # Calcul de la loss
        loss = lambda1 * loss1(outputs, labels) + lambda2 * loss2(spectrogram_outputs, spectrogram_labels)
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


fig, axs = plt.subplots(2)

axs[0].plot(actuals_train[0].detach().numpy()[0], color='blue')
axs[1].plot(predicted_labels[0].detach().numpy()[0], color='red')
axs[1].plot(actual_labels[0].detach().numpy()[0], color='blue')
plt.show()

