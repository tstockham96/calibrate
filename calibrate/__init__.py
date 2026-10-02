"""Calibrate (prototype): log typed decisions, join ground truth, watch accuracy/calibration/drift.

Pure standard library. The bundled demo uses clearly-labelled SYNTHETIC data and a mock provider.
"""
from .monitor import Monitor
from .providers import MockProvider, TypeSafeProvider, get_default_provider

__all__ = ["Monitor", "MockProvider", "TypeSafeProvider", "get_default_provider"]
__version__ = "0.0.1"
