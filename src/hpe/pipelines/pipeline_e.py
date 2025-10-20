"""
Pipeline E: 노이즈 필터링된 경계 기반 preprocessing.

This pipeline builds on Pipeline D by removing small noisy edge fragments,
keeping only large structural edges through Connected Components analysis
or morphological operations.

Workflow:
1. Validate input
2. Apply Pipeline D (edge detection)
3. Filter small components (noise removal)
   - Method A: Connected Components Analysis (size-based filtering)
   - Method B: Morphological Opening (shape-based filtering)
4. Return clean binary edge map (0 or 255)

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
    filter_method: str = "connected_components",
    min_component_size: int = 50,
    use_tissue_mask: bool = False
) -> np.ndarray:
    """
    Process an image using Pipeline E (Noise-filtered edge-based).
    
    Workflow:
    1. Validate input (image type, dimensions, method parameters)
    2. Apply Pipeline D to get initial edges
    3. Filter small noisy components based on size
       - connected_components: Keep only components >= min_component_size pixels
       - morphology: Apply morphological opening to remove small structures
    4. Return clean binary edge map
    
    Args:
        image: Input RGB image as numpy ndarray with shape (H, W, 3) or (H, W)
        edge_method: Edge detection method for Pipeline D ('sobel' or 'canny')
        filter_method: Noise filtering method. Options:
                      'connected_components' (default): Size-based component filtering
                      'morphology': Morphological opening operation
        min_component_size: Minimum component size in pixels (default: 50)
                           Components smaller than this are removed
        use_tissue_mask: If True, use histomicstk tissue mask in Pipeline D (optional)
        
    Returns:
        Processed clean edge image (2D binary, values 0 or 255)
        
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
    
    # Step 1: Get edges from Pipeline D
    edges = pipeline_d_process(
        image,
        edge_method=edge_method,
        use_tissue_mask=use_tissue_mask
    )
    
    # Step 2: Filter small noise components
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
    
    # Step 3: Ensure output is uint8
    result = filtered.astype(np.uint8)
    
    return result
