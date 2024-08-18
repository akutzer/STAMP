from typing import Tuple
import numpy as np
import cv2
import PIL


def canny_filter(patch: np.array) -> bool:
    h, w = patch.shape[:2]
    if h * w == 0: return False

    patch_img = PIL.Image.fromarray(patch)
    patch_gray = np.array(patch_img.convert("L"))
    # tile_to_grayscale is an PIL.Image.Image with image mode L
    # Note: If you have an L mode image, that means it is
    # a single channel image - normally interpreted as grayscale.
    # The L means that is just stores the Luminance.
    # It is very compact, but only stores a grayscale, not color.

    # hardcoded thresholds
    edge = cv2.Canny(patch_gray, 40, 100, L2gradient=True)
    edge = edge / (np.max(edge) + 1e-8)
    edge = (np.sum(np.sum(edge)) / (h * w) * 100)

    # hardcoded limit. Less or equal to 2 edges will be rejected (i.e., not saved)
    return edge >= 2


def filter_background(patches: np.array, patches_coords: np.array) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the patches which do not only contain background."""
    n = len(patches)
    has_tissue = np.zeros((n,), dtype=np.bool_)

    for k, patch in enumerate(patches):
        has_tissue[k] = canny_filter(patch)
   
    patches = patches[has_tissue]
    patches_coords = patches_coords[has_tissue]

    return patches, patches_coords, np.sum(~has_tissue).item()
