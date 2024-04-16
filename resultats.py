import numpy as np
import matplotlib.pyplot as plt

import pickle 
from statsmodels.tsa.ar_model import AutoReg
import scipy
import scipy.io.wavfile as wav
from AR_hybride import AR_freq
from persistance_freq_env import AR_freq_env

### Définitions des constantes ###
Resample_rate = 32000
context_length = 0.04      #### Durée en seconde de la fenêtre de contexte
predict_length = 0.02       #### Durée en seconde de la fenêtre de prédiction
context_size = int(context_length * Resample_rate)     #### Taille en nombre de points de la fenêtre de contexte
predict_size = int(predict_length * Resample_rate)     #### Taille en nombre de points de la fenêtre de prédiction

#### Importation des données ####

with open('time_series\data.pkl', 'rb') as fichier:
    data = pickle.load(fichier)

sample_rate = data[2][0]                #### On récupère le sample rate
song_1 = np.array(data[2][1])             #### On prend la première série temporelle

print('Le sample rate est : ', sample_rate)

if sample_rate != Resample_rate :
    song_1 = scipy.signal.resample(song_1, int(len(song_1)*Resample_rate/sample_rate))  #### On resample la série temporelle si le sample rate n'est pas celui désiré

song_1 = np.mean(song_1, axis=1)

pos_loss_packets = np.random.randint(2000, len(song_1) - 2 * predict_size, 1000)  #### On choisit n positions de paquets perdus au hasard
pos_loss_packets = np.sort(pos_loss_packets)
mask = pos_loss_packets[1:]>(pos_loss_packets[:-1]+context_size*2)
mask = [True]+mask.tolist()
pos_loss_packets = pos_loss_packets[mask]  #### On enlève les positions trop proches
print('Il y a ', len(pos_loss_packets), 'paquets perdus')

#### Fonction pour le crossfade ####

def f(x):
    return 1/(1+np.exp(3*x))

f = np.vectorize(f)

#### On remplace les paquets perdus avec diverses méthodes ####

#### Méthode vide ####

def vide(song, pos_loss_packets, predict_size):
    print('On effectue la méthode vide')
    song_predicted = song.copy()
    for pos in pos_loss_packets :
        song_predicted[pos: pos + predict_size] = np.zeros(predict_size)
    return song_predicted

#### Méthode persistance ####

def persistance(song, pos_loss_packets, predict_size):
    print('On effectue la méthode persistance')
    song_predicted = song.copy()
    for pos in pos_loss_packets :
        song_predicted[pos: pos + predict_size] = song[pos - predict_size: pos]
    return song_predicted

#### Méthode AR, option crossfade, option nombre de lags adaptatif, option context adaptatif ####

def AR(song, pos_loss_packets, predict_size, context_size, lags, crossfade = False, adaptatif = False, context_variable = False, threshold = 1.2):
    
    print('On effectue la méthode AR')

    song_predicted = song.copy()

    if crossfade :
        print('Avec un crossfade')
        crossfade_size = predict_size // 20
        crossfade_window = f(np.linspace(-1/2,1/2,crossfade_size))
    else : 
        crossfade_size = 0

    if adaptatif :
        print('Avec un nombre de lags adaptatif')
        if context_variable :
            print('Et un contexte variable')
    
    nb_lags = lags

    for pos in pos_loss_packets :
        context = song[pos - context_size: pos]
        context = context.astype(np.int16)
        # train autoregression

        model = AutoReg(context, lags)
        model_fit = model.fit()
        # make predictions
        predictions = model_fit.predict(start=context_size, end=context_size + predict_size + crossfade_size - 1)

        if adaptatif :
            if context_variable :
                while (np.max(np.abs(predictions)) > threshold * np.max(np.abs(context))) and (nb_lags > 32):
                    nb_lags = nb_lags // 2
                    context = context[len(context)//2:]
                    model = AutoReg(context, nb_lags)
                    model_fit = model.fit()
                    predictions = model_fit.predict(start=context_size, end=context_size + predict_size + crossfade_size - 1, dynamic=False)
            else : 
                while (np.max(np.abs(predictions)) > threshold * np.max(np.abs(context))) and (nb_lags > 32):
                    nb_lags = nb_lags // 2
                    model = AutoReg(context, nb_lags)
                    model_fit = model.fit()
                    predictions = model_fit.predict(start=context_size, end=context_size + predict_size + crossfade_size - 1, dynamic=False) 
        nb_lags = lags

        if np.max(predictions) > 32767 : 
            predictions = 32767 * np.ones(len(predictions))
        elif np.min(predictions) < -32767 :
            predictions  = -32767 * np.ones(len(predictions))

        # complete song   
        if crossfade :
            song_predicted[pos: pos + predict_size] = predictions[:predict_size]
            song_predicted[pos + predict_size: pos + predict_size + crossfade_size] = (crossfade_window * predictions[predict_size:] + (1-crossfade_window) * song[pos + predict_size: pos + predict_size + crossfade_size])
        else :     
            song_predicted[pos: pos + predict_size] = predictions

    return song_predicted

#### Méthode persistance fréquentielle ####

def finding_local_max(list):
    r = list[1:]
    l = list[:-1]
    mask_l = r > l
    mask_r = l > r
    mask = mask_l[:-1] * mask_r[1:]
    mask = np.concatenate((np.array([False]), mask, np.array([False])))
    list_max = np.zeros(len(list))
    list_max[mask] = list[mask]
    ind = np.argsort(list_max)[::-1]
    return ind

def freq_max_sorted(fft, n):

    fft_abs = np.abs(fft)
    r = fft_abs[1:]
    l = fft_abs[:-1]
    mask_l = r > l
    mask_r = l > r
    mask = mask_l[:-1] * mask_r[1:]
    mask = np.concatenate((np.array([True]), mask, np.array([False])))

    list_max = np.zeros(len(fft_abs))
    list_max[mask] = fft_abs[mask]
    ind = np.argsort(list_max)[:-n]

    fft[ind] = 0
    return fft

def persistance_freq(song, pos_loss_packets, predict_size, context_size, lags, crossfade = False):

    print('On effectue la méthode persistance fréquentielle')

    song_predicted = song.copy()
    temps_paquet = context_size/sample_rate

    if crossfade : 
        print('Avec un crossfade')
        crossfade_size = predict_size // 20
        crossfade_window = f(np.linspace(-1/2,1/2,crossfade_size))
    else :
        crossfade_size = 0

    for pos in pos_loss_packets :
        context = song[pos - context_size: pos]
        context_fft = scipy.fft.fft(context)
        context_fft = freq_max_sorted(context_fft, lags)
        predictions = np.real(scipy.fft.ifft(context_fft))

        autocor = np.correlate(context, context, mode = 'same')
        ind = finding_local_max(autocor)
        period = np.abs(ind[0]- ind[1])/sample_rate

        #Fixing parameters needed
        x = temps_paquet / period
        t_phased =  temps_paquet- np.floor(x)*period

        #dephasing the context_fft
        freq = scipy.fft.fftfreq(context_size, d=1/sample_rate)
        dephasage = np.exp(1j*2*np.pi*t_phased*freq)
        fft_signal = context_fft * dephasage
        
        predictions = np.real(scipy.fft.ifft(fft_signal))

        # complete song        
        if crossfade and t_phased * sample_rate < predict_size + crossfade_size: ### Si le paquet est trop petit pour faire un crossfade on ne le fait pas
            song_predicted[pos: pos + predict_size] = predictions[:predict_size]
            song_predicted[pos + predict_size: pos + predict_size + crossfade_size] = (crossfade_window * predictions[predict_size:predict_size+crossfade_size] + (1-crossfade_window) * song[pos + predict_size: pos + predict_size + crossfade_size])
        else :
            song_predicted[pos: pos + predict_size] = predictions[:predict_size]
            
    return song_predicted

def AR_hybrid(song, pos_loss_packets, predict_size, context_size, lags, crossfade = False):
    
    print('On effectue la méthode AR_hybride')

    song_predicted = song.copy()

    if crossfade :
        print('Avec un crossfade')
        crossfade_size = predict_size // 10
        crossfade_window = f(np.linspace(-1/2,1/2,crossfade_size))
    else : 
        crossfade_size = 0

    AR = AR_freq(song, sample_rate, sample_rate, nb_lags = lags, nb_freq_kept = 20, train_size = context_size)
    n=0
    for pos in pos_loss_packets :
        AR.fit(pos)
        AR.predict(predict_size+crossfade_size)
        predictions = AR.pred

        if AR.diverge == False :
            n+=1
            # complete song   
            if crossfade :
                song_predicted[pos: pos + predict_size] = predictions[:predict_size]
                song_predicted[pos + predict_size: pos + predict_size + crossfade_size] = (crossfade_window * predictions[predict_size:] + (1-crossfade_window) * song[pos + predict_size: pos + predict_size + crossfade_size])
            else :     
                song_predicted[pos: pos + predict_size] = predictions[:predict_size]
        else : 
            # complete song
            pos_loss_packets = np.delete(pos_loss_packets, np.where(pos_loss_packets == pos))
            song_predicted[pos: pos + predict_size] = song[pos: pos + predict_size]

    print(f"Il y a {n/len(pos_loss_packets)*100} % de paquets qui n'ont pas divergé")
    return song_predicted

#### Méthode fréquentielle ####

def Persistance_freq(song, pos_loss_packets, predict_size, context_size, lags, crossfade = False):
    
    print('On effectue la méthode AR_hybride')

    song_predicted = song.copy()

    if crossfade :
        print('Avec un crossfade')
        crossfade_size = predict_size // 10
        crossfade_window = f(np.linspace(-1/2,1/2,crossfade_size))
    else : 
        crossfade_size = 0

    AR = AR_freq(song, sample_rate, sample_rate, nb_lags = lags, nb_freq_kept = 20, train_size = context_size)
    for pos in pos_loss_packets :
        AR.fit(pos-crossfade_size)
        AR.predict(predict_size+2*crossfade_size)
        predictions = AR.pred
  
        if crossfade :
            song_predicted[pos-crossfade_size: pos] = (crossfade_window * song[pos-crossfade_size:pos] + (1-crossfade_window) * predictions[:crossfade_size])
            song_predicted[pos: pos + predict_size] = predictions[crossfade_size:predict_size+crossfade_size]
            song_predicted[pos + predict_size: pos + predict_size + crossfade_size] = (crossfade_window * predictions[predict_size+crossfade_size:] + (1-crossfade_window) * song[pos + predict_size: pos + predict_size + crossfade_size])
        else :     
            song_predicted[pos: pos + predict_size] = predictions[crossfade_size:predict_size]
    return song_predicted

#### Méthode fréquentielle avec enveloppe ####

def AR_hybrid_env(song, pos_loss_packets, predict_size, train_size, crossfade = False):
    
    print('On effectue la méthode AR_hybride_env')

    song_predicted = song.copy()

    if crossfade :
        print('Avec un crossfade')
        crossfade_size = predict_size // 20
        crossfade_window = f(np.linspace(-1/2,1/2,crossfade_size))
    else : 
        crossfade_size = 0

    AR = AR_freq_env(song, sample_rate, sample_rate, nb_freq_kept = 50, train_size = train_size)
    
    for pos in pos_loss_packets :
        AR.fit(pos)
        AR.predict(predict_size+crossfade_size)
        predictions = AR.pred
        # complete song   
        if crossfade :
            song_predicted[pos: pos + crossfade_size] = (crossfade_window * song[pos:pos+crossfade_size] + (1-crossfade_window) * predictions[:crossfade_size])
            song_predicted[pos + crossfade_size: pos + predict_size] = predictions[crossfade_size:predict_size]
            song_predicted[pos + predict_size: pos + predict_size + crossfade_size] = (crossfade_window * predictions[predict_size:] + (1-crossfade_window) * song[pos + predict_size: pos + predict_size + crossfade_size])
        else :     
            song_predicted[pos: pos + predict_size] = predictions[:predict_size]

    return song_predicted

#### Résultats ####
predictions_AR_128 = AR(song_1, pos_loss_packets, predict_size, context_size, lags = 128, crossfade=True)
predictions_AR_256 = AR(song_1, pos_loss_packets, predict_size, context_size, lags = 256, crossfade=True)
#predictions_AR_hybride = AR_hybrid(song_1, pos_loss_packets, predict_size, context_size, lags = 256)
#predictions_AR_hybride_crossfade = AR_hybrid(song_1, pos_loss_packets, predict_size, context_size, lags = 256, crossfade = True)
#predictions_vide = vide(song_1, pos_loss_packets, predict_size)
#predictions_persistance = persistance(song_1, pos_loss_packets, predict_size)
#predictions_AR = AR(song_1, pos_loss_packets , predict_size, context_size, lags = 256)
#predictions_AR_crossfade = AR(song_1, pos_loss_packets, predict_size, context_size, 256, crossfade = True)
#predictions_AR_adaptatif = AR(song_1, pos_loss_packets, predict_size, context_size, 256, adaptatif = True)
#predictions_AR_adaptatif_crossfade = AR(song_1, pos_loss_packets, predict_size, context_size, 256, crossfade = True, adaptatif = True)
#predictions_AR_adaptatif_crossfade_variable = AR(song_1, pos_loss_packets, predict_size, context_size, 256, crossfade = True, adaptatif = True, context_variable = True)
#predictions_persistance_freq = Persistance_freq(song_1, pos_loss_packets, predict_size, context_size, lags = 0)
#predictions_persistance_freq_crossfade = Persistance_freq(song_1, pos_loss_packets, predict_size, context_size, lags = 0, crossfade = True)
#predictions_AR_hybride_env = AR_hybrid_env(song_1, pos_loss_packets, predict_size, context_size)
#predictions_AR_hybride_env_crossfade = AR_hybrid_env(song_1, pos_loss_packets, predict_size, context_size, crossfade = True)

# Save the predicted songs
names = ['predictions_vide', 'predictions_persistance', 'predictions_AR', 'predictions_AR_crossfade', 'predictions_AR_adaptatif', 'predictions_AR_adaptatif_crossfade', 'predictions_AR_adaptatif_crossfade_variable', 'predictions_persistance_freq', 'predictions_persistance_freq_crossfade', 'predictions_AR_hybride', 'predictions_AR_hybride_crossfade', 'predictions_AR_hybride_env', 'predictions_AR_hybride_env_crossfade']
for i, predictions_name in enumerate(['predictions_AR_128', 'predictions_AR_256']):
    wav.write(f"predictions/{predictions_name}.wav", Resample_rate, globals()[predictions_name].astype(np.int16))

wav.write(f"predictions/original.wav", Resample_rate, song_1.astype(np.int16))

'''RMSE_vide = []
RMSE_persistance = []
RMSE_AR = []
RMSE_AR_crossfade = []
RMSE_AR_adaptatif = []
RMSE_AR_adaptatif_crossfade = []
RMSE_AR_adaptatif_crossfade_variable = []
RMSE_persistance_freq = []
RMSE_persistance_freq_crossfade = []
RMSE_AR_hybride = []
RMSE_AR_hybride_crossfade = []
RMSE_AR_hybride_env = []
RMSE_AR_hybride_env_crossfade = []

AME_vide = []
AME_persistance = []
AME_AR = []
AME_AR_crossfade = []
AME_AR_adaptatif = []
AME_AR_adaptatif_crossfade = []
AME_AR_adaptatif_crossfade_variable = []
AME_persistance_freq = []
AME_persistance_freq_crossfade = []
AME_AR_hybride = []
AME_AR_hybride_crossfade = []   
AME_AR_hybride_env = []
AME_AR_hybride_env_crossfade = []

def rmse(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())

def ame(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.mean(np.abs(predictions - targets))

for pos in pos_loss_packets : 
    RMSE_vide.append(rmse(song_1[pos: pos + predict_size], predictions_vide[pos: pos + predict_size]))
    RMSE_persistance.append(rmse(song_1[pos: pos + predict_size], predictions_persistance[pos: pos + predict_size]))
    RMSE_AR.append(rmse(song_1[pos: pos + predict_size], predictions_AR[pos: pos + predict_size]))
    RMSE_AR_crossfade.append(rmse(song_1[pos: pos + predict_size], predictions_AR_crossfade[pos: pos + predict_size]))
    RMSE_AR_adaptatif.append(rmse(song_1[pos: pos + predict_size], predictions_AR_adaptatif[pos: pos + predict_size]))
    RMSE_AR_adaptatif_crossfade.append(rmse(song_1[pos: pos + predict_size], predictions_AR_adaptatif_crossfade[pos: pos + predict_size]))
    RMSE_AR_adaptatif_crossfade_variable.append(rmse(song_1[pos: pos + predict_size], predictions_AR_adaptatif_crossfade_variable[pos: pos + predict_size]))
    RMSE_persistance_freq.append(rmse(song_1[pos: pos + predict_size], predictions_persistance_freq[pos: pos + predict_size]))
    RMSE_persistance_freq_crossfade.append(rmse(song_1[pos: pos + predict_size], predictions_persistance_freq_crossfade[pos: pos + predict_size]))
    RMSE_AR_hybride.append(rmse(song_1[pos: pos + predict_size], predictions_AR_hybride[pos: pos + predict_size]))
    RMSE_AR_hybride_crossfade.append(rmse(song_1[pos: pos + predict_size], predictions_AR_hybride_crossfade[pos: pos + predict_size]))    
    RMSE_AR_hybride_env.append(rmse(song_1[pos: pos + predict_size], predictions_AR_hybride_env[pos: pos + predict_size]))
    RMSE_AR_hybride_env_crossfade.append(rmse(song_1[pos: pos + predict_size], predictions_AR_hybride_env_crossfade[pos: pos + predict_size]))


    AME_vide.append(ame(song_1[pos: pos + predict_size], predictions_vide[pos: pos + predict_size]))
    AME_persistance.append(ame(song_1[pos: pos + predict_size], predictions_persistance[pos: pos + predict_size]))
    AME_AR.append(ame(song_1[pos: pos + predict_size], predictions_AR[pos: pos + predict_size]))
    AME_AR_crossfade.append(ame(song_1[pos: pos + predict_size], predictions_AR_crossfade[pos: pos + predict_size]))
    AME_AR_adaptatif.append(ame(song_1[pos: pos + predict_size], predictions_AR_adaptatif[pos: pos + predict_size]))
    AME_AR_adaptatif_crossfade.append(ame(song_1[pos: pos + predict_size], predictions_AR_adaptatif_crossfade[pos: pos + predict_size]))
    AME_AR_adaptatif_crossfade_variable.append(ame(song_1[pos: pos + predict_size], predictions_AR_adaptatif_crossfade_variable[pos: pos + predict_size]))
    AME_persistance_freq.append(ame(song_1[pos: pos + predict_size], predictions_persistance_freq[pos: pos + predict_size]))
    AME_persistance_freq_crossfade.append(ame(song_1[pos: pos + predict_size], predictions_persistance_freq_crossfade[pos: pos + predict_size]))
    AME_AR_hybride.append(ame(song_1[pos: pos + predict_size], predictions_AR_hybride[pos: pos + predict_size]))
    AME_AR_hybride_crossfade.append(ame(song_1[pos: pos + predict_size], predictions_AR_hybride_crossfade[pos: pos + predict_size]))
    AME_AR_hybride_env.append(ame(song_1[pos: pos + predict_size], predictions_AR_hybride_env[pos: pos + predict_size]))
    AME_AR_hybride_env_crossfade.append(ame(song_1[pos: pos + predict_size], predictions_AR_hybride_env_crossfade[pos: pos + predict_size]))

# Create a list of all the RMSE lists
rmse_lists = [RMSE_vide, RMSE_persistance, RMSE_AR, RMSE_AR_crossfade, RMSE_AR_adaptatif, RMSE_AR_adaptatif_crossfade, RMSE_AR_adaptatif_crossfade_variable, RMSE_persistance_freq, RMSE_persistance_freq_crossfade, RMSE_AR_hybride, RMSE_AR_hybride_crossfade, RMSE_AR_hybride_env, RMSE_AR_hybride_env_crossfade]
ame_lists = [AME_vide, AME_persistance, AME_AR, AME_AR_crossfade, AME_AR_adaptatif, AME_AR_adaptatif_crossfade, AME_AR_adaptatif_crossfade_variable, AME_persistance_freq, AME_persistance_freq_crossfade, AME_AR_hybride, AME_AR_hybride_crossfade, AME_AR_hybride_env, AME_AR_hybride_env_crossfade]

# Create a figure and axis
fig, ax = plt.subplots(2)

# Create the boxplot
ax[0].boxplot(rmse_lists)
ax[1].boxplot(ame_lists)

# Set the labels for the x-axis
labels = ['Vide', 'Persistance', 'AR', 'AR Crossfade', 'AR Adaptatif', 'AR Adaptatif Crossfade', 'AR Adaptatif Crossfade Variable', 'Persistance Freq', 'Persistance Freq Crossfade', 'AR Hybride', 'AR Hybride Crossfade', 'AR Hybride Env', 'AR Hybride Env Crossfade']
ax[0].set_xticklabels(labels, rotation=45)
ax[1].set_xticklabels(labels, rotation=45)

# Set the title and labels for the plot
ax[0].set_title('Boxplots of RMSE')
ax[0].set_xlabel('Methods')
ax[0].set_ylabel('RMSE')

ax[1].set_title('Boxplots of AME')
ax[1].set_xlabel('Methods')
ax[1].set_ylabel('AME')

plt.ylim(0, 5*10e2)
# Show the plot
plt.show()

#### Visualisation des résultats ####

time = np.linspace(0, 1.2*predict_size/Resample_rate, int(1.2*predict_size))

#### AR_hybride_env ####
for pos in pos_loss_packets : 
    _, axs = plt.subplots(3)

    axs[0].plot(time,song_1[pos:pos+predict_size+predict_size//5], color = 'blue', label = 'original', linestyle = '--', alpha = 0.5)
    axs[0].legend()
    axs[0].plot(time, predictions_AR_adaptatif[pos:pos+predict_size+predict_size//5], color = 'red', label = 'AR_adaptatif', linestyle = '--', alpha = 0.5)
    axs[0].legend()
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('AR_adaptatif')

    axs[1].plot(time,song_1[pos:pos+predict_size+predict_size//5], color = 'blue', label = 'original', linestyle = '--', alpha = 0.5)
    axs[1].legend()
    axs[1].plot(time,predictions_AR_hybride_crossfade[pos:pos+predict_size+predict_size//5], color = 'red', label = 'AR_hybride_crossfade', linestyle = '--', alpha = 0.5)
    axs[1].legend()
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('AR_hybride_crossfade')

    axs[2].plot(time,song_1[pos:pos+predict_size+predict_size//5], color = 'blue', label = 'original', linestyle = '--', alpha = 0.5)
    axs[2].legend()
    axs[2].plot(time,predictions_persistance_freq_crossfade[pos:pos+predict_size+predict_size//5], color = 'red', label = 'persistance_freq_crossfade', linestyle = '--', alpha = 0.5)
    axs[2].legend()
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Amplitude')
    axs[2].set_title('persistance_freq_crossfade')

    plt.tight_layout()
    plt.show()'''