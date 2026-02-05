"""
Audio Features Data Model
"""
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class AudioFeatures:
    """Container for extracted audio features"""
    mel_spectrogram: np.ndarray
    mfcc: np.ndarray
    pitch_contour: np.ndarray
    spectral_centroid: np.ndarray
    spectral_rolloff: np.ndarray
    zero_crossing_rate: np.ndarray
    prosody_features: Dict[str, float]
    duration_seconds: float