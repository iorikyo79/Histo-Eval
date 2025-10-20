"""
Pipeline C: Eosin-focused preprocessing.

This pipeline extracts the eosin channel (cytoplasm) through color normalization
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
    Process an image using Pipeline C (Eosin-focused).
    
    Workflow:
    1. Apply Reinhard color normalization to standardize staining
    2. Perform color deconvolution to separate stains
    3. Extract eosin channel (represents cytoplasm/background)
    4. Normalize to 0-255 range and convert to uint8
    5. Optional: Apply LoG filter for nucleus center enhancement
    
    Args:
        image: Input RGB image as numpy ndarray with shape (H, W, 3)
        use_clog: If True, apply LoG (Laplacian of Gaussian) filter to 
                  enhance nucleus centers. This highlights blob-like structures
                  at specific scales (nucleus sizes).
        
    Returns:
        Processed eosin channel image (2D grayscale, uint8).
        
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
    
    # Step 3: Extract eosin channel (second channel, index 1)
    # Eosin stains cytoplasm (typically appears pink)
    #Stains shape: (64, 64, 3)
    #Channel 0 = Hematoxylin (nuclei)
    #Channel 1 = Eosin (cytoplasm)
    #Channel 2 = Residual
    eosin = stains.Stains[:, :, 1]
    
    # Step 4: Normalize to 0-255 range
    # Deconvolution output is optical density, need to convert to intensity
    e_min = eosin.min()
    e_max = eosin.max()

    if e_max > e_min:
        # Normalize to [0, 255]
        eosin_normalized = (eosin - e_min) / (e_max - e_min) * 255
    else:
        # Uniform image, set to middle gray
        eosin_normalized = np.full_like(eosin, 128)


    # Convert to uint8
    result = eosin_normalized.astype(np.uint8)

    return result
