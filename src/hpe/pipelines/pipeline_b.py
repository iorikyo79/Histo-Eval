"""
Pipeline B: Nuclei-focused preprocessing.

This pipeline extracts the hematoxylin channel (nuclei) through color normalization
and deconvolution, with optional LoG filtering for nucleus center enhancement.

References:
    - Reinhard et al. "Color transfer between images." IEEE CGA, 2001.
    - Ruifrok & Johnston. "Quantification of histochemical staining by color
      deconvolution." Analytical and Quantitative Cytology and Histology, 2001.
"""

import numpy as np
import histomicstk as htk
from histomicstk.preprocessing.color_normalization import reinhard


# Reference statistics for Reinhard normalization (typical H&E stained tissue)
# These values are from Reinhard et al. paper for standard tissue appearance
_REINHARD_REF_MU = [8.63234435, -0.11501964, 0.03868433]
_REINHARD_REF_SIGMA = [0.57506023, 0.10403329, 0.01364062]

# Standard H&E stain matrix from Ruifrok & Johnston
# Columns represent: [Hematoxylin, Eosin, Residual]
# Rows represent RGB channels
_HE_STAIN_MATRIX = np.array([
    [0.650, 0.072, 0],  # Red channel
    [0.704, 0.990, 0],  # Green channel  
    [0.286, 0.105, 0]   # Blue channel
])

# Maximum transmitted light intensity (white background)
_MAX_TRANSMITTED_LIGHT = 255


def process(image: np.ndarray, use_clog: bool = False) -> np.ndarray:
    """
    Process an image using Pipeline B (Nuclei-focused).
    
    Workflow:
    1. Apply Reinhard color normalization to standardize staining
    2. Perform color deconvolution to separate stains
    3. Extract hematoxylin channel (represents nuclei)
    4. Normalize to 0-255 range and convert to uint8
    5. Optional: Apply LoG filter for nucleus center enhancement
    
    Args:
        image: Input RGB image as numpy ndarray with shape (H, W, 3)
        use_clog: If True, apply LoG filter to enhance nucleus centers
        
    Returns:
        Processed hematoxylin channel image (2D grayscale, uint8)
        
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
    
    # If grayscale input, just return as uint8
    if image.ndim == 2:
        return image.astype(np.uint8)
    
    # Step 1: Apply Reinhard color normalization
    # Standardizes color appearance across different slides
    normalized = reinhard(
        image,
        target_mu=_REINHARD_REF_MU,
        target_sigma=_REINHARD_REF_SIGMA
    )
    
    # Step 2: Perform color deconvolution to separate stains
    stains = htk.preprocessing.color_deconvolution.color_deconvolution(
        normalized,
        _HE_STAIN_MATRIX,
        _MAX_TRANSMITTED_LIGHT
    )
    
    # Step 3: Extract hematoxylin channel (first channel, index 0)
    # Hematoxylin stains nuclei (typically appears purple/blue)
    hematoxylin = stains.Stains[:, :, 0]
    
    # Step 4: Normalize to 0-255 range
    # Deconvolution output is optical density, need to convert to intensity
    h_min = hematoxylin.min()
    h_max = hematoxylin.max()
    
    if h_max > h_min:
        # Normalize to [0, 255]
        hematoxylin_normalized = (hematoxylin - h_min) / (h_max - h_min) * 255
    else:
        # Uniform image, set to middle gray
        hematoxylin_normalized = np.full_like(hematoxylin, 128)
    
    # Convert to uint8
    result = hematoxylin_normalized.astype(np.uint8)
    
    # Step 5: Optional LoG filter (to be implemented in item 7)
    if use_clog:
        # Placeholder: LoG filter will be implemented in next TODO item
        # For now, just return the hematoxylin channel
        pass
    
    return result
