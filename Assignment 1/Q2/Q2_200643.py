import cv2
import numpy as np
import librosa

def load_audio(audio_path):
    # Load the audio file and get its sample rate
    audio_signal, sample_rate = librosa.load(audio_path, sr=None)
    return audio_signal, sample_rate

def calculate_mel_spectrogram(audio_signal, sample_rate):
    # Define parameters for mel spectrogram calculation
    window = 2048
    stride = 512
    
    # Calculate the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_signal, sr=sample_rate, n_fft=window, hop_length=stride, fmax=22000
    )
    return mel_spectrogram

def calculate_weighted_mean_frequencies(mel_spectrogram, mel_frequencies):
    # Ensure that mel_spectrogram and mel_frequencies have compatible shapes
    assert mel_spectrogram.shape[0] == len(mel_frequencies), "Input shapes mismatch"

    # Initialize an array to store the weighted mean frequencies
    mean_frequencies = []

    # Iterate through each frame in the mel spectrogram
    for frame in mel_spectrogram.T:
        # Calculate the weighted mean for the current frame
        weighted_mean = np.sum(mel_frequencies * frame) / np.sum(frame) if np.sum(frame) > 0 else 0
        mean_frequencies.append(weighted_mean)

    return mean_frequencies

def solution(audio_path):
    # Load the audio
    audio_signal, sample_rate = load_audio(audio_path)
    
    # Calculate the mel spectrogram
    mel_spectrogram = calculate_mel_spectrogram(audio_signal, sample_rate)
    
    # Calculate mel frequencies
    mel_frequencies = librosa.mel_frequencies(n_mels=mel_spectrogram.shape[0], fmax=22000)
    
    # Calculate weighted mean frequencies
    mean_frequencies = calculate_weighted_mean_frequencies(mel_spectrogram, mel_frequencies)
    
    # Calculate overall mean frequency
    overall_mean_frequency = np.mean(mean_frequencies)
    
    # Calculate an index based on the overall mean frequency
    mel_frequency_index = int((overall_mean_frequency * mel_spectrogram.shape[0]) / 22000)
    
    # Split the mel spectrogram into upper and lower halves
    upper_half = mel_spectrogram[mel_frequency_index:, :]
    lower_half = mel_spectrogram[:mel_frequency_index, :]
    
    # Calculate the mean of the upper and lower halves
    upper_half_mean = np.mean(upper_half)
    lower_half_mean = np.mean(lower_half)
    
    # Determine the class based on the difference between means
    audio_class = 'metal' if upper_half_mean > lower_half_mean else 'cardboard'
   
    return audio_class