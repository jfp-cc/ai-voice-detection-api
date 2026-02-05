"""
Feature Extractor Component
Extracts acoustic features from audio for classification
"""
import io
import tempfile
from typing import Dict
import numpy as np
import librosa
import soundfile as sf
from features.audio_features import AudioFeatures


class FeatureExtractionError(Exception):
    """Exception raised when feature extraction fails"""
    pass


class FeatureExtractor:
    """Extracts comprehensive acoustic features from audio"""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize FeatureExtractor
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
    
    def extract_features(self, audio, sr=None) -> AudioFeatures:
        """
        Extracts comprehensive acoustic features from audio.
        
        Args:
            audio: Audio data (bytes or numpy array)
            sr: Sample rate (if audio is numpy array)
            
        Returns:
            AudioFeatures object containing all extracted features
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        try:
            # Handle different input types
            if isinstance(audio, bytes):
                y = self.preprocess_audio(audio)
            else:
                y = audio
                if sr is not None:
                    # Resample if needed
                    if sr != self.sample_rate:
                        y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            
            # Calculate duration
            duration = librosa.get_duration(y=y, sr=self.sample_rate)
            
            # Extract mel-spectrogram (64 bins for simple robust model)
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=self.sample_rate,
                n_mels=128,  # Changed back to 128 to match local
                n_fft=2048,
                hop_length=512,
                fmax=8000
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=self.sample_rate,
                n_mfcc=40
            )
            
            # Extract pitch contour (F0)
            pitch_contour = librosa.yin(
                y,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            
            # Extract spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=y,
                sr=self.sample_rate
            )[0]
            
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=y,
                sr=self.sample_rate
            )[0]
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # Extract prosody features
            prosody_features = self._extract_prosody_features(y, pitch_contour)
            
            return AudioFeatures(
                mel_spectrogram=mel_spec_db,
                mfcc=mfcc,
                pitch_contour=pitch_contour,
                spectral_centroid=spectral_centroid,
                spectral_rolloff=spectral_rolloff,
                zero_crossing_rate=zero_crossing_rate,
                prosody_features=prosody_features,
                duration_seconds=duration
            )
            
        except FeatureExtractionError:
            raise
        except Exception as e:
            raise FeatureExtractionError(f"Feature extraction failed: {str(e)}")
    
    def preprocess_audio(self, audio_bytes: bytes) -> np.ndarray:
        """
        Preprocesses audio for feature extraction.
        
        Args:
            audio_bytes: Raw audio data
            
        Returns:
            Normalized audio waveform as numpy array
        """
        try:
            # Try librosa first
            audio_io = io.BytesIO(audio_bytes)
            y, sr = librosa.load(audio_io, sr=self.sample_rate, mono=True)
            
            if len(y) == 0:
                raise FeatureExtractionError("Audio waveform is empty")
            
            # Normalize audio
            y = librosa.util.normalize(y)
            
            return y
            
        except Exception as e:
            raise FeatureExtractionError(f"Audio preprocessing failed: {str(e)}")
    
    def _extract_prosody_features(
        self,
        y: np.ndarray,
        pitch_contour: np.ndarray
    ) -> Dict[str, float]:
        """
        Extracts prosody features from audio.
        
        Args:
            y: Audio waveform
            pitch_contour: F0 pitch contour
            
        Returns:
            Dictionary of prosody features
        """
        # Calculate speaking rate (rough estimate based on zero crossings)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        speaking_rate = float(np.mean(zcr))
        
        # Calculate pitch statistics
        valid_pitch = pitch_contour[pitch_contour > 0]
        if len(valid_pitch) > 0:
            pitch_mean = float(np.mean(valid_pitch))
            pitch_std = float(np.std(valid_pitch))
            pitch_range = float(np.max(valid_pitch) - np.min(valid_pitch))
        else:
            pitch_mean = 0.0
            pitch_std = 0.0
            pitch_range = 0.0
        
        # Calculate energy statistics
        rms_energy = librosa.feature.rms(y=y)[0]
        energy_mean = float(np.mean(rms_energy))
        energy_std = float(np.std(rms_energy))
        
        return {
            'speaking_rate': speaking_rate,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'pitch_range': pitch_range,
            'energy_mean': energy_mean,
            'energy_std': energy_std
        }