"""
Pipeline A: Baseline++ preprocessing.

This pipeline applies artifact-robust tissue masking, grayscale conversion, 
masking, and CLAHE enhancement.
"""

import numpy as np


def process(image: np.ndarray) -> np.ndarray:
    """
    Process an image using Pipeline A (Baseline++).
    
    Args:
        image: Input RGB image as numpy ndarray with shape (H, W, 3)
        
    Returns:
        Processed grayscale image with CLAHE applied
        
    Raises:
        TypeError: If image is None or not a numpy ndarray
        ValueError: If image has invalid dimensions or dtype
    """
    # Validate input
    if image is None:
        raise TypeError("Input image cannot be None")
    
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input must be a numpy ndarray, got {type(image)}")
    
    # Check dimensions
    if image.ndim not in [2, 3]:
        raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D array")
    
    # For now, return a simple stub implementation
    # This will be replaced with actual implementation in next steps
    if image.ndim == 3:
        # Simple grayscale conversion as stub
        gray = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray = image.astype(np.uint8)
    
    return gray
