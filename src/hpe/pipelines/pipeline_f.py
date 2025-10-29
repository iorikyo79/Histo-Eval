"""
Pipeline F: Nuclei-focused preprocessing with CLAHE and intermediate saving.

This pipeline is a variant of Pipeline B. It extracts the hematoxylin channel
through color normalization and deconvolution. It allows saving the color-normalized
intermediate image and uses CLAHE for contrast enhancement instead of simple
min-max normalization.
"""

from typing import Optional
import numpy as np
import histomicstk as htk
from histomicstk.preprocessing.color_normalization import reinhard
from skimage.exposure import equalize_adapthist
from skimage.io import imsave
from skimage.util import img_as_ubyte


# Reference statistics for Reinhard normalization (typical H&E stained tissue)
_REINHARD_REF_MU = [8.63234435, -0.11501964, 0.03868433]
_REINHARD_REF_SIGMA = [0.57506023, 0.10403329, 0.01364062]

# Standard H&E stain matrix from Ruifrok & Johnston
_HE_STAIN_MATRIX = np.array([
    [0.650, 0.072, 0],  # Red channel
    [0.704, 0.990, 0],  # Green channel
    [0.286, 0.105, 0]   # Blue channel
])

# Maximum transmitted light intensity (white background)
_MAX_TRANSMITTED_LIGHT = 255


def process(
    image: np.ndarray,
    use_clog: bool = False,
    normalized_save_path: Optional[str] = None
) -> np.ndarray:
    """
    Process an image using Pipeline F (Nuclei-focused with CLAHE).

    Workflow:
    1. Apply Reinhard color normalization to standardize staining.
    2. (Optional) Save the color-normalized image if a path is provided.
    3. Perform color deconvolution to separate stains.
    4. Extract hematoxylin channel (represents nuclei).
    5. Apply CLAHE for local contrast enhancement.
    6. Optional: Apply LoG filter for nucleus center enhancement.

    Args:
        image: Input RGB image as numpy ndarray with shape (H, W, 3).
        use_clog: If True, apply LoG (Laplacian of Gaussian) filter to
                  enhance nucleus centers.
        normalized_save_path: If provided, the color-normalized intermediate
                              image is saved to this path.

    Returns:
        Processed hematoxylin channel image (2D grayscale, uint8).

    Raises:
        TypeError: If image is None or not a numpy ndarray.
        ValueError: If image has invalid dimensions or dtype.
    """
    # Validate input
    if image is None:
        raise TypeError("Input image cannot be None")

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input must be a numpy ndarray, got {type(image)}")

    if image.ndim not in [2, 3]:
        raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D array")

    if image.ndim == 2:
        return image.astype(np.uint8)

    # Step 1: Apply Reinhard color normalization
    normalized = reinhard(
        image,
        target_mu=_REINHARD_REF_MU,
        target_sigma=_REINHARD_REF_SIGMA
    )

    # Step 2: (Optional) Save intermediate normalized image
    if normalized_save_path:
        imsave(normalized_save_path, img_as_ubyte(normalized))

    # Step 3: Perform color deconvolution
    stains = htk.preprocessing.color_deconvolution.color_deconvolution(
        normalized,
        _HE_STAIN_MATRIX,
        _MAX_TRANSMITTED_LIGHT
    )

    # Step 4: Extract hematoxylin channel
    hematoxylin = stains.Stains[:, :, 0]

    # Step 5: Apply CLAHE instead of simple normalization
    h_min, h_max = hematoxylin.min(), hematoxylin.max()
    if h_max > h_min:
        hematoxylin_scaled = (hematoxylin - h_min) / (h_max - h_min)
    else:
        hematoxylin_scaled = np.full_like(hematoxylin, 0.5)

    hematoxylin_enhanced = img_as_ubyte(equalize_adapthist(hematoxylin_scaled))

    # Step 6: Optional LoG (Laplacian of Gaussian) filter
    if use_clog:
        im_mask = np.ones(hematoxylin_enhanced.shape, dtype=bool)
        result_tuple = htk.filters.shape.clog(
            hematoxylin_enhanced,
            im_mask,
            sigma_min=2 * np.sqrt(2),
            sigma_max=4 * np.sqrt(2)
        )
        hematoxylin_log = result_tuple[0] if isinstance(result_tuple, tuple) else result_tuple

        log_min, log_max = hematoxylin_log.min(), hematoxylin_log.max()
        if log_max > log_min:
            final_result = (hematoxylin_log - log_min) / (log_max - log_min) * 255
        else:
            final_result = np.full_like(hematoxylin_log, 128)
        
        return final_result.astype(np.uint8)

    return hematoxylin_enhanced.astype(np.uint8)
