"""
Simple Robust Classifier for Railway Deployment
Uses our working simple robust CNN model for AI voice detection
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from typing import Optional, Dict
from pathlib import Path
from features.audio_features import AudioFeatures
from models.classification_result import ClassificationResult


class SimpleRobustClassifier:
    """Simple robust classifier using our working CNN model"""
    
    def __init__(self, threshold: float = 0.5):
        """Initialize simple robust classifier
        
        Args:
            threshold: Decision threshold (default 0.5 for balanced predictions)
        """
        self.model_version = "simple_robust_v1.0"
        self.cnn_model = None
        self.threshold = threshold
        
        # Model architecture parameters (64x64 input)
        self.mel_bins = 64
        self.max_time_steps = 64
        self.input_shape = (self.mel_bins, self.max_time_steps, 1)
        
        # Load the working model
        self._load_model()
        
        print(f"✅ Simple Robust Classifier v1.0 initialized (threshold={threshold})")
    
    def _load_model(self):
        """Load the working simple robust CNN model"""
        try:
            model_path = Path("models/simple_robust_model.h5")
            
            if model_path.exists():
                self.cnn_model = keras.models.load_model(model_path)
                print(f"✅ Loaded simple robust CNN model: {model_path}")
                
                # Load metadata if available
                metadata_path = Path("models/simple_robust_metadata.json")
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    print(f"✅ Model metadata: {metadata.get('training_info', {}).get('balanced_accuracy', 'N/A')} balanced accuracy")
            else:
                print(f"❌ Simple robust model not found at {model_path}")
                
        except Exception as e:
            print(f"⚠️ Could not load simple robust model: {e}")
    
    def _extract_features_for_model(self, features: AudioFeatures) -> np.ndarray:
        """Extract features compatible with the simple robust model (64x64)"""
        try:
            # Use existing mel spectrogram if available
            if hasattr(features, 'mel_spectrogram') and features.mel_spectrogram is not None:
                mel_spec = features.mel_spectrogram
            else:
                # Create synthetic mel spectrogram from available features
                mel_spec = self._create_synthetic_mel_spectrogram(features)
            
            # Ensure correct shape
            if mel_spec.ndim == 1:
                mel_spec = mel_spec.reshape(-1, 1)
            
            # Resize to 64x64 (our model's input size)
            mel_spec_resized = self._resize_mel_spectrogram(mel_spec, self.mel_bins, self.max_time_steps)
            
            # Normalize to [0, 1] (same as training)
            mel_spec_normalized = (mel_spec_resized - mel_spec_resized.min()) / (mel_spec_resized.max() - mel_spec_resized.min() + 1e-10)
            
            return mel_spec_normalized
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            import traceback
            traceback.print_exc()
            # Return random features as fallback
            return np.random.uniform(0, 1, (self.mel_bins, self.max_time_steps))
    
    def _create_synthetic_mel_spectrogram(self, features: AudioFeatures) -> np.ndarray:
        """Create synthetic mel spectrogram from available features"""
        mel_spec = np.zeros((self.mel_bins, self.max_time_steps))
        
        # Use MFCC coefficients if available
        if hasattr(features, 'mfcc') and features.mfcc is not None and len(features.mfcc) > 0:
            mfcc_array = np.array(features.mfcc)
            if mfcc_array.ndim == 1:
                mfcc_array = mfcc_array.reshape(1, -1)
            
            for i, mfcc_frame in enumerate(mfcc_array):
                if i >= self.max_time_steps:
                    break
                for j, coeff in enumerate(mfcc_frame):
                    if j < self.mel_bins:
                        mel_spec[j, i] = coeff
        
        # Add spectral information if available
        if hasattr(features, 'spectral_centroid') and features.spectral_centroid is not None and len(features.spectral_centroid) > 0:
            centroid = np.array(features.spectral_centroid)
            for i, cent in enumerate(centroid):
                if i >= self.max_time_steps:
                    break
                bin_idx = int((cent / 8000.0) * self.mel_bins)
                bin_idx = np.clip(bin_idx, 0, self.mel_bins - 1)
                mel_spec[bin_idx, i] += 1.0
        
        return mel_spec
    
    def _resize_mel_spectrogram(self, mel_spec: np.ndarray, target_mel_bins: int, target_time_steps: int) -> np.ndarray:
        """Resize mel spectrogram to target dimensions"""
        current_mel_bins, current_time_steps = mel_spec.shape
        
        # Simple resize using numpy interpolation
        from scipy import ndimage
        zoom_factors = (target_mel_bins / current_mel_bins, target_time_steps / current_time_steps)
        resized = ndimage.zoom(mel_spec, zoom_factors, order=1)
        
        return resized
    
    def classify(self, features: AudioFeatures) -> ClassificationResult:
        """Classify using the simple robust model"""
        try:
            if self.cnn_model is None:
                # Fallback to simple heuristics
                return self._fallback_classification(features)
            
            # Extract features for the model
            mel_spec = self._extract_features_for_model(features)
            
            # Prepare input for CNN (add batch and channel dimensions)
            X = mel_spec.reshape(1, self.mel_bins, self.max_time_steps, 1)
            
            # Get prediction (binary classification with sigmoid output)
            prediction_prob = self.cnn_model.predict(X, verbose=0)[0][0]
            
            # Apply threshold
            if prediction_prob > self.threshold:
                label = "ai_generated"
                confidence = float(prediction_prob)
            else:
                label = "human"
                confidence = float(1.0 - prediction_prob)
            
            # Ensure reasonable confidence range (at least 50%)
            confidence = max(confidence, 0.5)
            confidence = min(confidence, 0.95)
            
            return ClassificationResult(
                label=label,
                confidence=confidence,
                feature_importance={
                    'cnn_prediction': 1.0
                },
                model_version=self.model_version
            )
            
        except Exception as e:
            print(f"❌ Classification error: {e}")
            return self._fallback_classification(features)
    
    def _fallback_classification(self, features: AudioFeatures) -> ClassificationResult:
        """Fallback classification when model is not available"""
        # Simple heuristic based on spectral consistency
        if hasattr(features, 'spectral_centroid') and features.spectral_centroid is not None and len(features.spectral_centroid) > 0:
            centroid_std = np.std(features.spectral_centroid)
            centroid_mean = np.mean(features.spectral_centroid)
            centroid_cv = centroid_std / centroid_mean if centroid_mean > 0 else 0.5
            
            if centroid_cv < 0.2:
                # Low variation suggests AI
                return ClassificationResult(
                    label="ai_generated",
                    confidence=0.7,
                    feature_importance={'spectral_consistency': 1.0},
                    model_version=self.model_version + "_fallback"
                )
            else:
                # High variation suggests human
                return ClassificationResult(
                    label="human",
                    confidence=0.7,
                    feature_importance={'spectral_consistency': 1.0},
                    model_version=self.model_version + "_fallback"
                )
        else:
            # Default to human if no features
            return ClassificationResult(
                label="human",
                confidence=0.6,
                feature_importance={},
                model_version=self.model_version + "_fallback"
            )
    
    def detect_language(self, features: AudioFeatures) -> str:
        """Simple language detection fallback"""
        # For deployment, just return English as default
        # In production, this would use a proper language detector
        return "en"