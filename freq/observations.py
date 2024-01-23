import os
import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt


#### Constantes ####
sample_size = 2048
SampleRate = 44100
nb_coeffs = 16

song_directory = r"C:\Users\nicor\Documents\Mines\Data Sophia\Projet\songs"

files = os.listdir(song_directory)    
files_path = []

songs = []

for file in files:
    files_path.append(os.path.join(song_directory, file))

for i in range (len(files_path)) : 
    sr, song = wav.read(files_path[i])
    if len(song.shape) == 2 : 
        song = np.mean(song, axis = 1)
    song = signal.resample(song, int(len(song)/sr*SampleRate))
    songs.append(song)

index = [np.random.randint(0, len(songs[0])-sample_size) for _ in range(1)]

samples = [np.array(songs[0][i:i+sample_size]) for i in index]
samples = np.array(samples)


def freq_max_sorted(fft, n):

    '''Renvoie les n plus grandes valeurs de fft, en mettant à 0 les autres valeurs'''

    fft_copy = np.copy(fft)
    rfft = np.abs(fft_copy.real)
    imfft = np.abs(fft_copy.imag)
    # Renvoie une liste de booléen de même taille que values, avec True si la valeur correspondante est un maximum local
    rmask_1 = rfft[1:]>rfft[:-1]
    rmask_2 = rfft[:-1]>rfft[1:]

    imask_1 = imfft[1:]>imfft[:-1]
    imask_2 = imfft[:-1]>imfft[1:]

    max_locaux = [False for _ in range (len(fft_copy))]

    for i in range(len(fft_copy)-2):
        if rmask_1[i] and rmask_2[i+1] :
            max_locaux[i+1] = True
        if imask_1[i] and imask_2[i+1] :
            max_locaux[i+1] = True

    if rmask_2[0] :
        max_locaux[0] = True
    if rmask_1[-1] :
        max_locaux[-1] = True
    if imask_2[0] :
        max_locaux[0] = True
    if imask_1[-1] :
        max_locaux[-1] = True

    # On inverse la liste de booléen pour avoir True si la valeur correspondante n'est pas un maximum local
    inversed_max_locaux = [not i for i in max_locaux]
    
    # On met à 0 les valeurs qui ne sont pas des maximums locaux
    fft_copy[inversed_max_locaux] = 0

    # On trie les valeurs
    rindexes_sorted = np.argsort(rfft)[n//2::-1]
    imindexes_sorted = np.argsort(imfft)[n//2::-1]

    # On concatène les deux listes d'indexes
    indexes_sorted = np.concatenate((rindexes_sorted, imindexes_sorted))

    # On met à 0 les valeurs qui ne sont pas parmi les n plus grandes
    fft_copy[indexes_sorted] = 0

    return fft_copy

def freq_max_sorted_2(fft, n):

    '''Renvoie les n plus grandes valeurs de fft, en mettant à 0 les autres valeurs'''

    fft_copy = np.copy(fft)
    fft_copy = np.abs(fft_copy)
    # Renvoie une liste de booléen de même taille que values, avec True si la valeur correspondante est un maximum local
    mask_1 = fft_copy[1:]>fft_copy[:-1]
    mask_2 = fft_copy[:-1]>fft_copy[1:]

    max_locaux = [False for _ in range (len(fft_copy))]

    for i in range(len(fft_copy)-2):
        if mask_1[i] and mask_2[i+1] :
            max_locaux[i+1] = True
    if mask_2[0] :
        max_locaux[0] = True
    if mask_1[-1] :
        max_locaux[-1] = True


    # On inverse la liste de booléen pour avoir True si la valeur correspondante n'est pas un maximum local
    inversed_max_locaux = [not i for i in max_locaux]
    
    # On met à 0 les valeurs qui ne sont pas des maximums locaux
    fft_copy[inversed_max_locaux] = 0

    # On trie les valeurs
    indexes_sorted = np.argsort(fft)[n::-1]

    # On met à 0 les valeurs qui ne sont pas parmi les n plus grandes
    fft_copy[indexes_sorted] = 0

    return fft_copy

def rmse(signal1, signal2):
    return np.sqrt(np.mean(np.square(signal1 - signal2)))

def optim(signal, nb_coeffs):
    sigmas = np.linspace(0.00001, 0.01, 1000)
    RMSE = []
    x = np.linspace(-1, 1, sample_size)
    for sigma in sigmas : 
        gaussian_vector = np.exp(-np.square(x)/sigma)
        fft = np.fft.fft(signal)

        fft_1 = freq_max_sorted(fft[:int(sample_size/2)], nb_coeffs)
        fft_1_real = np.convolve(np.real(fft_1), gaussian_vector, mode = 'same')
        fft_1_imag = np.convolve(np.imag(fft_1), gaussian_vector, mode = 'same')
        fft_1 = fft_1_real + 1j*fft_1_imag
        ifft_1 = np.real(np.fft.ifft(fft_1, n = sample_size))*2
        RMSE.append(rmse(ifft_1, samples[0]))

    if sigmas[np.argmin(RMSE)] == 0.00001 : 
        print("On n'effectue pas de convolution")
        fft_1 = freq_max_sorted(fft[:int(sample_size/2)], nb_coeffs)
        ifft_1 = np.real(np.fft.ifft(fft_1, n = sample_size))*2
        #plt.plot(ifft_1, color = 'red' , label = 'reconstructed')
        #plt.plot(signal, color = 'blue', label = 'original', linestyle = '--', alpha = 0.5)
        #plt.show()
        return ifft_1
    
    else : 
        print("On effectue une convolution")
        gaussian_vector = np.exp(-np.square(x)/sigmas[np.argmin(RMSE)])
        fft_1 = freq_max_sorted(fft[:int(sample_size/2)], nb_coeffs)
        fft_1_real = np.convolve(np.real(fft_1), gaussian_vector, mode = 'same')
        fft_1_imag = np.convolve(np.imag(fft_1), gaussian_vector, mode = 'same')
        fft_1 = fft_1_real + 1j*fft_1_imag
        ifft_1 = np.real(np.fft.ifft(fft_1, n = sample_size))*2
        return ifft_1
    
def optim_2(signal, nb_coeffs):
    fft = np.fft.fft(signal)
    fft_1 = freq_max_sorted(fft[:int(sample_size/2)], nb_coeffs)
    fft_2 = freq_max_sorted_2(fft[:int(sample_size/2)], nb_coeffs)
    ifft_1 = np.real(np.fft.ifft(fft_1, n = sample_size))*2
    ifft_2 = np.real(np.fft.ifft(fft_2, n = sample_size))*2
    if rmse(ifft_1, signal) < rmse(ifft_2, signal) :
        print("On prend la première méthode")

for sample in samples :
        fft = np.fft.fft(sample)
        fft_1 = freq_max_sorted(fft[:int(sample_size/2)], nb_coeffs)
        ifft_1 = np.real(np.fft.ifft(fft_1, n = sample_size))*2
        plt.plot(ifft_1, color = 'red' , label = 'reconstructed')
        plt.plot(sample, color = 'blue', label = 'original', linestyle = '--', alpha = 0.5)
        plt.legend()
        plt.show()