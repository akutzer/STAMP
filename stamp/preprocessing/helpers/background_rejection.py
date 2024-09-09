from typing import Tuple
import numpy as np
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def canny_filter(tile: np.ndarray, low_threshold: int = 40, high_threshold: int = 100, edge_threshold: float = 2.0) -> bool:
    """Determine if a patch contains significant edges, indicating tissue."""
    h, w = tile.shape[:2]
    if h * w == 0:
        return False
    
    tile_gray = np.array(Image.fromarray(tile).convert("L"))

    # tile_gray = cv2.GaussianBlur(tile_gray, (5, 5), 0)
    edges = cv2.Canny(tile_gray, low_threshold, high_threshold, L2gradient=True)
    edges = edges / (np.max(edges) + 1e-8)
    edges = np.sum(edges) / (h * w) * 100

    return edges >= edge_threshold


def filter_background(patches: np.ndarray, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Filter out patches that only contain background based on edge detection."""
    has_edges = np.array([canny_filter(patch) for patch in patches], dtype=np.bool_)
    # with ThreadPoolExecutor(max_workers=8) as executor:
    #     has_edges = np.array(list(executor.map(canny_filter, patches)), dtype=np.bool_)

    patches = patches[has_edges]
    coordinates = coordinates[has_edges]
    num_rejected = np.sum(~has_edges).item()

    return patches, coordinates, num_rejected
