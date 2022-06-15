import torch
import torchaudio
import torchvision
import librosa
import numpy as np
from data.parameters import CLASSES_MAPPING

class remove_elements(
    object
    ):
    
    def __call__(
        self, 
        inputs
        ):
      
        waveform = inputs[0]
        sample_rate = inputs[1]
        class_label = inputs[2]
        return waveform, sample_rate, class_label

class process_labels(
    object
    ):
    
    def __init__(
        self, 
        classes_mapping=CLASSES_MAPPING
        ):
      
        self.classes_mapping = classes_mapping

    def __call__(
        self, 
        inputs
        ):
      
        waveform = inputs[0]
        sample_rate = inputs[1]
        class_label = inputs[2]
        class_label = torch.tensor(self.classes_mapping[class_label])
        return waveform, sample_rate, class_label

class shift_pitch(
    object
    ):

    def __init__(
        self,
        limits=[-1, 1]
        ):
      
        self.limits = limits

    def __call__(
        self,
        inputs
        ):
      
        waveform = inputs[0]
        sample_rate = inputs[1]
        class_label = inputs[2]
        pitch_shift = np.random.randint(self.limits[0], self.limits[1] + 1)
        waveform = torch.tensor(librosa.effects.pitch_shift(waveform.squeeze().numpy(), sample_rate, pitch_shift)).view(1, -1)
        return waveform, sample_rate, class_label

class stretch_time(
    object
    ):

    def __init__(
        self,
        limits=[0.9, 1.1]
        ):
      
        self.limits = limits

    def __call__(
        self,
        inputs
        ):
      
        waveform = inputs[0]
        sample_rate = inputs[1]
        class_label = inputs[2]
        time_stretch = np.random.random() * (self.limits[1] - self.limits[0]) + self.limits[0]
        waveform = torch.tensor(librosa.effects.time_stretch(waveform.squeeze().numpy(), time_stretch)).view(1, -1)
        return waveform, sample_rate, class_label

class add_noise(
    object
    ):
  
    def __init__(
        self,
        limits=[20, 100]
        ):
      
        self.limits = limits

    def __call__(
        self,
        inputs
        ):
      
        waveform = inputs[0]
        sample_rate = inputs[1]
        class_label = inputs[2]
        snr_dB = np.random.random() * (self.limits[1] - self.limits[0]) + self.limits[0]
        snr = 10 ** (snr_dB / 20)
        noise = torch.tensor(np.random.randn(waveform.shape[1]) * np.sqrt(np.mean(waveform.numpy() ** 2)) / snr).view(1, -1)
        waveform += noise
        return waveform, sample_rate, class_label

class resample(
    object
    ):
    
    def __init__(
        self, 
        new_sample_rate=8000
        ):
      
        self.new_sample_rate = new_sample_rate

    def __call__(
        self, 
        inputs
        ):
      
        waveform = inputs[0]
        orig_sample_rate = inputs[1]
        class_label = inputs[2]
        waveform = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=self.new_sample_rate)(waveform)
        return waveform, self.new_sample_rate, class_label

class mel_spectrogram(
    object
    ):

    def __init__(
        self, 
        num_channels=3,
        win_length=[200, 400, 800], 
        hop_length=[80, 200, 400]
        ):
      
        self.num_channels = num_channels
        self.win_length = win_length
        self.hop_length = hop_length
    
    def __call__(
        self, 
        inputs
        ):
      
        waveform = inputs[0]
        sample_rate = inputs[1]
        class_label = inputs[2]
        spectrogram = []
        for i in range(self.num_channels):
            spectrogram_ = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=2205, win_length=self.win_length[i], hop_length=self.hop_length[i], n_mels=128)(waveform)
            spectrogram_ = torchvision.transforms.Resize((128, 250))(spectrogram_)
            spectrogram_ = np.log(spectrogram_.numpy() + 1e-6)
            spectrogram.append(spectrogram_)
        spectrogram = torch.tensor(np.squeeze((spectrogram)))
        return spectrogram, class_label