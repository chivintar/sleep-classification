import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch


print('Running feature_extraction.py')

def extract_time_features(epoch_data):
    ''' 
        Takes epochs object (shape: (number_of_channels, number_of_epochs) ).
        Computes the first four statistical moments of each epoch, 
        together with the root mean square and the peak-to-peak amplitude. 
        Returns a flat list of the features of each channel for individual epoch (shape: number_of_channels, 7).
    '''
    features = []
    for ch_data in epoch_data:
        features.extend([
            ch_data.mean(),
            ch_data.std(),
            skew(ch_data),
            kurtosis(ch_data),
            np.sqrt(np.mean(ch_data**2)),  # RMS
            np.ptp(ch_data),               # Peak to peak
            ((ch_data[:-1] * ch_data[1:]) < 0).sum()  # Zero crossings
        ])
    return features



def bandpower(data, sf, band, window_seconds=4):
    ''' 
    Returns a scalar that depicts the average band power in the given frequency range (Delta, Theta, Alpha, Beta)
    with the Welch method, a spectral density estimation method. https://en.wikipedia.org/wiki/Welch%27s_method, 
    using the build-in scipy function: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html#welch

    The widnow_seconds are set to 4 (typical for EEG) so that for 100Hz sampling frequency we have 400samples for each segment. 
    More seconds lead to finer frequency-bins and fewwer seconds lead to more stable segments. Function adapted from Raphael Vallat.
    '''
    band = np.asarray(band)
    freqs, psd = welch(data, sf, nperseg=window_seconds * sf) # default with Hann tapering and 50% overlapping
    freq_res = freqs[1] - freqs[0]  # for approximation of the integral 
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1]) #this is to select only the freq bins inside the desired band
    return psd[idx_band].sum() * freq_res #the approximate integration




def extract_frequency_features(epoch_data, sf):
    ''' 
    Returns a flat list for individual epoch with the bandpower values as columns 
    for each channel-band pair, (ch1_delta, ch1_theta,..., ch2_beta)
    '''
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30)] # delta, theta, alpha, beta respectively
    features = []
    for ch_data in epoch_data: #for every channel
        for band in bands:
            features.append(bandpower(ch_data, sf, band))
    return features




def extract_all_features(epoch_data, sf):
    ''' Extracts both time and frequency-domain features. Features include: first four statistical moments, RMS, number of xero crossings, average bandpower in theta,delta,alpha,beta bands
    '''
    return extract_time_features(epoch_data) + extract_frequency_features(epoch_data, sf)
    