import histomicstk.saliency.tissue_detection as htk_td

import numpy as np
import cv2

def test_get_tissue_mask():
    # Create a dummy image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[30:70, 30:70] = [255, 0, 0]  # Add a red square

    # Call the function
    labeled, mask = htk_td.get_tissue_mask(image)

    # Check the outputs
    assert labeled is not None
    assert mask is not None
    assert mask.shape == (100, 100)


def print_get_tissue_mask_doc():
    doc = htk_td.get_tissue_mask.__doc__
    print("Documentation for histomicstk.saliency.tissue_detection.get_tissue_mask:")
    print(doc)

if __name__ == "__main__":
   # test_get_tissue_mask()
    print_get_tissue_mask_doc()
   # print("All tests passed.")