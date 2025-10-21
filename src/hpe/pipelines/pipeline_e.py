"""
Pipeline E: 노이즈 필터링된 경계 기반 preprocessing.

This pipeline builds on Pipeline D by removing small noisy edge fragments,
keeping only large structural edges through Connected Components analysis
or morphological operations. Optionally blends edges with CLAHE-enhanced
original image for better feature detection.

Workflow:
1. Validate input
2. Apply Pipeline D (edge detection)
3. Filter small components (noise removal)
   - Method A: Connected Components Analysis (size-based filtering)
   - Method B: Morphological Opening (shape-based filtering)
4. (Optional) Blend filtered edges with CLAHE-enhanced original
5. Return clean edge map (binary 0/255) or blended image (grayscale 0-255)

References:
    - Connected Components: Suzuki, S. "Topological Structural Analysis". CVGIP, 1985.
    - Morphological Operations: Serra, J. "Image Analysis and Mathematical Morphology". 1982.
"""

import numpy as np
import cv2
from src.hpe.pipelines.pipeline_d import process as pipeline_d_process


def process(
    image: np.ndarray,
    edge_method: str = "sobel",
    filter_method: str = "morphology",
    min_component_size: int = 100,
    use_tissue_mask: bool = False,
    blend_with_original: bool = False,
    blend_alpha: float = 0.7
) -> np.ndarray:
    """
    Process an image using Pipeline E (Noise-filtered edge-based).
    
    Workflow:
    1. Validate input (image type, dimensions, method parameters)
    2. Apply Pipeline D to get initial edges (internally processes with CLAHE+Blur)
    3. Filter small noisy components based on size
       - connected_components: Keep only components >= min_component_size pixels
       - morphology: Apply morphological opening to remove small structures
    4. (Optional) Blend filtered edges with CLAHE-enhanced original for better features
    5. Return clean edge map or blended image
    
    Args:
        image: Input RGB image as numpy ndarray with shape (H, W, 3) or (H, W)
        edge_method: Edge detection method for Pipeline D ('sobel' or 'canny')
        filter_method: Noise filtering method. Options:
                      'connected_components': Size-based component filtering
                      'morphology' (default): Morphological opening operation
        min_component_size: Minimum component size in pixels (default: 100)
                           Components smaller than this are removed
        use_tissue_mask: If True, use histomicstk tissue mask in Pipeline D (optional)
        blend_with_original: If True, blend edges with CLAHE-enhanced grayscale (default: False)
                            Useful for SuperPoint feature detection on pathology images
        blend_alpha: Blending weight for original image (default: 0.7)
                    Result = alpha * original_enhanced + (1-alpha) * edges
                    Higher alpha = more original texture, lower alpha = more edges
        
    Returns:
        If blend_with_original=False: Clean edge image (2D binary, values 0 or 255)
        If blend_with_original=True: Blended image (2D grayscale, values 0-255)
        
    Raises:
        TypeError: If image is None or not a numpy ndarray
        ValueError: If image has invalid dimensions, or invalid method parameters
    """
    # Validate input
    if image is None:
        raise TypeError("Input image cannot be None")
    
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input must be a numpy ndarray, got {type(image)}")
    
    # Check dimensions
    if image.ndim not in [2, 3]:
        raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D array")
    
    # Validate filter_method
    if filter_method not in ["connected_components", "morphology"]:
        raise ValueError(
            f"filter_method must be 'connected_components' or 'morphology', "
            f"got '{filter_method}'"
        )
    
    # Validate blend_alpha
    if not (0.0 <= blend_alpha <= 1.0):
        raise ValueError(f"blend_alpha must be between 0.0 and 1.0, got {blend_alpha}")
    
    # Step 1: Prepare CLAHE-enhanced grayscale (for blending if needed)
    # This replicates Pipeline D's preprocessing to get the enhanced image
    enhanced_gray = None
    if blend_with_original:
        # Convert to grayscale
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur for noise reduction
        enhanced_gray = cv2.GaussianBlur(enhanced, ksize=(0, 0), sigmaX=1.2)
    
    # Step 2: Get edges from Pipeline D
    edges = pipeline_d_process(
        image,
        edge_method=edge_method,
        use_tissue_mask=use_tissue_mask
    )
    
    # Step 3: Filter small noise components
    if filter_method == "connected_components":
        # Connected Components Analysis
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            edges, connectivity=8
        )
        
        # Create filtered image by keeping only large components
        filtered = np.zeros_like(edges)
        for i in range(1, num_labels):  # Skip background (label 0)
            component_size = stats[i, cv2.CC_STAT_AREA]
            if component_size >= min_component_size:
                filtered[labels == i] = 255
                
    else:  # morphology
        # Morphological Opening: Erosion followed by Dilation
        # Kernel size is derived from min_component_size
        # Approximate: kernel_size ≈ sqrt(min_component_size) / 2
        kernel_size = max(3, int(np.sqrt(min_component_size) / 2))
        if kernel_size % 2 == 0:  # Ensure odd number
            kernel_size += 1
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_size, kernel_size)
        )
        filtered = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    
    # Step 4: Blend with CLAHE-enhanced original if requested
    if blend_with_original and enhanced_gray is not None:
        # Weighted blending: alpha * enhanced_gray + (1-alpha) * edges
        # Convert edges to same range as enhanced_gray (0-255)
        result = (
            blend_alpha * enhanced_gray.astype(np.float32) +
            (1 - blend_alpha) * filtered.astype(np.float32)
        )
        result = np.clip(result, 0, 255).astype(np.uint8)
    else:
        # Return clean binary edge map
        result = filtered.astype(np.uint8)
    
    return result
