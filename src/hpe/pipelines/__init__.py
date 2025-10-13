"""
Pipeline modules package.

This package contains all preprocessing pipeline implementations.
Each pipeline implements a common interface: process(image: np.ndarray) -> np.ndarray
"""

__all__ = ['pipeline_a', 'pipeline_b', 'pipeline_c', 'pipeline_d']
