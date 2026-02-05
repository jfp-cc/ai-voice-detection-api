"""
Classification Result Data Model
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class ClassificationResult:
    """Container for classification results"""
    label: str  # "ai_generated" or "human"
    confidence: float  # 0.0 to 1.0
    feature_importance: Dict[str, float]
    model_version: str