"""Scoring package: registration + integration + combined."""
from .registration import compute_registration_accuracy, geodesic_rotation_error, translation_error
from .integration import compute_integration_score, check_embedding_collapse
from .combined import compute_final_score

__all__ = [
    "compute_registration_accuracy",
    "geodesic_rotation_error",
    "translation_error",
    "compute_integration_score",
    "check_embedding_collapse",
    "compute_final_score",
]
