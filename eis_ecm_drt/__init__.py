"""EIS to ECM parameters and DRT curves."""

from .circuit import CircuitModel
from .drt import DRTConfig, solve_drt
from .ecm import ECMConfig, ecm_configs_from_dict, ecm_model_presets, fit_ecm
from .pipeline import analyze_batch, analyze_file

__version__ = "0.3.2"

__all__ = [
    "CircuitModel",
    "DRTConfig",
    "ECMConfig",
    "analyze_batch",
    "analyze_file",
    "ecm_configs_from_dict",
    "ecm_model_presets",
    "fit_ecm",
    "solve_drt",
]
