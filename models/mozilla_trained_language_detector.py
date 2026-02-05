#!/usr/bin/env python3
"""
Mozilla-Trained Language Detector
Trains on actual Mozilla Common Voice data instead of hardcoded bullshit
"""
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.audio_features import AudioFeatures
from features.feature_extractor import FeatureExtractor


class MozillaTrainedLanguageDetector:
    """Language detector trained on actual Mozilla Common Voice data"""
    
    def __init__(self):
        """Initialize the detector"""
        self.model = None
        self.scaler = None
        self.feature_extractor = FeatureExtractor()
        self.language_codes = ['en', 'hi', 'ml', 'te', 'ta']
        self.model_path = Path("models/mozilla_language_model.pkl")
        self.scaler_path = Path("models/mozilla_language_scaler.pkl")
        
        # Try to load existing model
        if self.model_path.exists() and self.scaler_path.exists():
            self.load_model()
    
    def extract_language_features(self, features: AudioFeatures) -> np.ndarray:
        """Extract features relevant for language detection"""
        feature_vector = []
        
        # 1. Spectral features
        if len(features.spectral_centroid) > 0:
            feature_vector.extend([
                np.mean(features.spectral_centroid),
                np.std(features.spectral_centroid),
                np.median(features.spectral_centroid),
                np.percentile(features.spectral_centroid, 25),
                np.percentile(features.spectral_centroid, 75)
            ])
        else:
            feature_vector.extend([0, 0, 0, 0, 0])
        
        if len(features.spectral_rolloff) > 0:
            feature_vector.extend([
                np.mean(features.spectral_rolloff),
                np.std(features.spectral_rolloff),
                np.median(features.spectral_rolloff),
                np.percentile(features.spectral_rolloff, 25),
                np.percentile(features.spectral_rolloff, 75)
            ])
        else:
            feature_vector.extend([0, 0, 0, 0, 0])
        
        # 2. MFCC features (first 13 coefficients)
        if len(features.mfcc) > 0:
            mfcc_array = np.array(features.mfcc)
            if mfcc_array.ndim > 1:
                # Take mean of each MFCC coefficient across time
                mfcc_means = np.mean(mfcc_array[:13], axis=1) if mfcc_array.shape[0] >= 13 else np.mean(mfcc_array, axis=1)
                # Pad or truncate to exactly 13 features
                if len(mfcc_means) < 13:
                    mfcc_means = np.pad(mfcc_means, (0, 13 - len(mfcc_means)))
                else:
                    mfcc_means = mfcc_means[:13]
                feature_vector.extend(mfcc_means.tolist())
            else:
                # Fallback for 1D MFCC
                feature_vector.extend([np.mean(mfcc_array)] + [0] * 12)
        else:
            feature_vector.extend([0] * 13)
        
        # 3. Pitch and prosody features
        prosody = features.prosody_features
        feature_vector.extend([
            prosody.get('pitch_mean', 0),
            prosody.get('pitch_std', 0),
            prosody.get('pitch_range', 0),
            prosody.get('speaking_rate', 0),
            prosody.get('energy_mean', 0),
            prosody.get('energy_std', 0)
        ])
        
        # 4. Zero crossing rate
        if len(features.zero_crossing_rate) > 0:
            feature_vector.extend([
                np.mean(features.zero_crossing_rate),
                np.std(features.zero_crossing_rate)
            ])
        else:
            feature_vector.extend([0, 0])
        
        # 5. Duration
        feature_vector.append(features.duration_seconds)
        
        return np.array(feature_vector)
    
    def detect_language(self, features: AudioFeatures) -> str:
        """Detect language using trained model"""
        if self.model is None or self.scaler is None:
            print("⚠️  Model not trained! Using fallback...")
            return "en"  # Default fallback
        
        try:
            # Extract features
            feature_vector = self.extract_language_features(features)
            
            # Scale features
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Predict
            prediction = self.model.predict(feature_vector_scaled)[0]
            
            return prediction
            
        except Exception as e:
            print(f"Language detection error: {e}")
            return "en"  # Default fallback
    
    def get_language_confidence(self, features: AudioFeatures) -> Dict[str, float]:
        """Get confidence scores for all languages"""
        if self.model is None or self.scaler is None:
            return {lang: 0.2 for lang in self.language_codes}  # Equal probability
        
        try:
            # Extract features
            feature_vector = self.extract_language_features(features)
            
            # Scale features
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Get probabilities
            probabilities = self.model.predict_proba(feature_vector_scaled)[0]
            
            # Map to language codes
            confidence_dict = {}
            for i, lang in enumerate(self.model.classes_):
                confidence_dict[lang] = probabilities[i]
            
            # Fill in missing languages with low probability
            for lang in self.language_codes:
                if lang not in confidence_dict:
                    confidence_dict[lang] = 0.01
            
            return confidence_dict
            
        except Exception as e:
            print(f"Confidence calculation error: {e}")
            return {lang: 0.2 for lang in self.language_codes}
    
    def load_model(self):
        """Load trained model and scaler"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            print("✅ Loaded existing Mozilla language model")
            
        except Exception as e:
            print(f"⚠️  Could not load existing model: {e}")
            self.model = None
            self.scaler = None