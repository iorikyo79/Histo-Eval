"""
Pipeline D: Edge-based preprocessing.

This pipeline emphasizes tissue structural edges through edge detection,
with optional simple background suppression (avoiding early tissue masking).

Workflow:
1. Validate input
2. Convert to grayscale
3. Apply CLAHE for contrast enhancement
4. Apply Gaussian blur for noise reduction
5. Edge detection (Sobel or Canny)
6. Threshold to binary edge map
7. Optional simple background suppression (non-HTK based)
8. Return binary edge map (0 or 255)

References:
    - Canny, J. "A Computational Approach to Edge Detection." IEEE TPAMI, 1986.
    - Sobel, I. "An Isotropic 3x3 Image Gradient Operator." 1968.
"""

import numpy as np
import cv2
from skimage.feature import canny


def process(
    image: np.ndarray, 
    use_tissue_mask: bool = False,
    edge_method: str = "sobel"
) -> np.ndarray:
    """
    Process an image using Pipeline D (Edge-based).
    
    Workflow:
    1. Validate input (image type, dimensions, edge_method parameter)
    2. Convert to grayscale
    3. Apply CLAHE for contrast enhancement (clipLimit=2.0, tileGridSize=8x8)
    4. Apply Gaussian blur (sigma=1.2) for noise reduction
    5. Edge detection (Sobel with Otsu threshold or Canny with adaptive threshold)
    6. Background suppression (remove edges in white regions >245 or use tissue mask)
    7. Convert to binary uint8 (0 or 255)
    
    Args:
        image: Input RGB image as numpy ndarray with shape (H, W, 3) or (H, W)
        use_tissue_mask: If True, use histomicstk tissue mask (optional, slower).
                        Default False uses simple background suppression.
        edge_method: Edge detection method to use. Options: "sobel" (default), "canny"
        
    Returns:
        Processed edge image (2D binary, values 0 or 255)
        
    Raises:
        TypeError: If image is None or not a numpy ndarray
        ValueError: If image has invalid dimensions, dtype, or edge_method
    """
    # Validate input
    if image is None:
        raise TypeError("Input image cannot be None")
    
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input must be a numpy ndarray, got {type(image)}")
    
    # Check dimensions
    if image.ndim not in [2, 3]:
        raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D array")
    
    # Validate edge_method
    if edge_method not in ["sobel", "canny"]:
        raise ValueError(f"edge_method must be 'sobel' or 'canny', got '{edge_method}'")
    
    # Step 1: Convert to grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.uint8)
    
    # Step 2: Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Step 3: Gaussian blur for noise reduction (after CLAHE)
    blurred = cv2.GaussianBlur(enhanced, ksize=(0, 0), sigmaX=1.2)
    
    # Step 4 & 5: Edge detection and thresholding
    if edge_method == "sobel":
        # Sobel gradient calculation
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize to 0-255
        gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)
        
        # Otsu threshold for binary edge map
        _, edges_binary = cv2.threshold(gradient_magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = edges_binary > 0
        
    else:  # canny
        # Convert to float for Canny (expects 0-1 range)
        gray_norm = blurred.astype(np.float32) / 255.0
        
        # Use median-based adaptive threshold
        median_val = float(np.median(gray_norm))
        
        # Conservative thresholds for tissue images
        low_threshold = np.clip(0.66 * median_val, 0.05, 0.3)
        high_threshold = np.clip(1.33 * median_val, low_threshold + 0.05, 0.5)
        
        # Canny edge detection
        edges = canny(
            gray_norm,
            sigma=1.5,
            low_threshold=low_threshold,
            high_threshold=high_threshold
        )
    
    # Step 6: Background suppression
    if use_tissue_mask:
        # Use histomicstk tissue mask (optional, more accurate but slower)
        try:
            import histomicstk.saliency.tissue_detection as htk_td
            tissue_mask = htk_td.get_tissue_mask(
                image if image.ndim == 3 else np.stack([image]*3, axis=-1),
                deconvolve_first=True,
                n_thresholding_steps=1,
                sigma=1.5
            )[0]
            # Apply mask to edges
            edges = edges & (tissue_mask > 0)
        except (ImportError, AttributeError, Exception):
            # Fallback to simple background suppression
            use_tissue_mask = False
    
    if not use_tissue_mask:
        # Simple background suppression: remove edges in near-white regions
        # White background typically > 245 (out of 255) - more conservative
        background_mask = enhanced > 245  # Use enhanced image for consistency
        # Remove edges in background
        edges = edges & (~background_mask)
    
    # Step 7: Convert boolean to uint8 (0 or 255)
    result = (edges * 255).astype(np.uint8)
    
    return result
