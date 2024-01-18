import os
import scipy.io.wavfile as wav

train_size = 512
test_size = 128
sample_size = train_size + test_size

song_directory = "songs"
dataset_directory = "dataset"

files = os.listdir(song_directory)    
files_path = []

for file in files:
    files_path.append(os.path.join(song_directory, file))

print(files_path)

for file_path in files_path : 
    sr, song = wav.read(file_path)
    print(song)
    for i in range (len(song)//sample_size):
        wav.write(os.path.join(dataset_directory, f'input_audio_{i}.wav'), sr, song[i*sample_size:i*sample_size + train_size])
        wav.write(os.path.join(dataset_directory, f'label_audio_{i}.wav'), sr, song[i*sample_size + train_size:(i+1)*sample_size])
