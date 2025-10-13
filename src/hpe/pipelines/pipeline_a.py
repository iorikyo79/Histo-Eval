"""
Pipeline A: Baseline++ preprocessing.

This pipeline applies artifact-robust tissue masking, grayscale conversion, 
masking, and CLAHE enhancement.
"""

import numpy as np
import cv2


def process(image: np.ndarray) -> np.ndarray:
    """
    Process an image using Pipeline A (Baseline++).
    
    Implementation follows PRD specification:
    1. Get tissue mask robust to artifacts (using histomicstk if available)
    2. Convert to grayscale
    3. Apply mask to grayscale image
    4. Apply CLAHE enhancement
    
    Args:
        image: Input RGB image as numpy ndarray with shape (H, W, 3)
        
    Returns:
        Processed grayscale image with CLAHE applied (2D uint8)
        
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
    
    # Handle grayscale input
    if image.ndim == 2:
        im_gray = image.astype(np.uint8)
        # For grayscale, create a simple threshold-based mask
        im_mask = np.where(im_gray > 10, 255, 0).astype(np.uint8)
    else:
        # RGB input - full Baseline++ pipeline
        try:
            import histomicstk.saliency.tissue_detection as htk_td
            
            # 1. Get tissue mask robust to artifacts
            im_mask = htk_td.get_tissue_mask(
                image, 
                deconvolve_first=True, 
                n_thresholding_steps=1, 
                sigma=1.5
            )[0]
        except (ImportError, AttributeError, Exception) as e:
            # Fallback if histomicstk is not available or fails
            # Use Otsu's thresholding on grayscale
            gray_temp = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, im_mask = cv2.threshold(gray_temp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Clean up the mask with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            im_mask = cv2.morphologyEx(im_mask, cv2.MORPH_CLOSE, kernel)
            im_mask = cv2.morphologyEx(im_mask, cv2.MORPH_OPEN, kernel)
        
        # 2. Convert to grayscale
        im_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 3. Apply mask
    im_masked_gray = im_gray * (im_mask / 255.0)
    
    # 4. Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    im_clahe = clahe.apply(im_masked_gray.astype(np.uint8))
    
    return im_clahe
