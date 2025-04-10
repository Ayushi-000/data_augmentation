# augmentation/audio_aug.py

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
import numpy as np
import soundfile as sf

class AudioAugmentor:
    def __init__(self):
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5)
        ])

    def augment(self, input_path, output_path):
        samples, sample_rate = sf.read(input_path)
        augmented = self.augment(samples=samples, sample_rate=sample_rate)
        sf.write(output_path, augmented, sample_rate)
        return output_path
