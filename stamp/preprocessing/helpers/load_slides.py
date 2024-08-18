import re
from typing import Tuple
from concurrent import futures
import openslide
from tqdm import tqdm
import numpy as np
from PIL import Image

from .exceptions import MPPExtractionError

Image.MAX_IMAGE_PIXELS = None



def _load_tile(
        slide: openslide.OpenSlide, pos: Tuple[int, int], stride: Tuple[int, int], target_size: Tuple[int, int]
    ) -> np.ndarray:
    # Loads part of a WSI. Used for parallelization with ThreadPoolExecutor
    tile = slide.read_region(pos, 0, stride).convert('RGB').resize(target_size)
    return np.array(tile)


def load_slide(slide: openslide.OpenSlide, target_mpp: float = 256/224, cores: int = 8) -> np.ndarray:
    """Loads a slide into a numpy array."""
    # We load the slides in chunks to:
    #  1. parallelize the loading process using Threads since it's IO heavy
    #  2. not use too much data when then scaling down the tiles from their
    #     initial size
    chunks = np.ceil(np.array(slide.dimensions) / 4096).astype(int)
    stride = np.ceil(np.array(slide.dimensions) / chunks).astype(int)
    slide_mpp = float(get_slide_mpp(slide))
    tile_size = np.round(stride * slide_mpp / target_mpp).astype(int) # (width, height) for openslide
    # print(chunks, stride, tile_size)
    # return np.random.randn(5, 5)

    with futures.ThreadPoolExecutor(cores) as executor:
        # map from future to its (row, col) index
        future_coords: dict[futures.Future, Tuple[int, int]] = {}
        for i in range(chunks[1]):  # row
            for j in range(chunks[0]):  # column
                future = executor.submit(
                    _load_tile, slide, (stride*(j, i)), stride, tile_size)
                future_coords[future] = (i, j)

        # write the loaded tiles into an array as soon as they are loaded
        n_tiles_w, n_tiles_h = tile_size * chunks
        img = np.zeros((n_tiles_h, n_tiles_w, 3), dtype=np.uint8)
        for tile_future in tqdm(futures.as_completed(future_coords), total=chunks[0]*chunks[1], desc='Reading WSI tiles', leave=False):
            i, j = future_coords[tile_future]
            tile = tile_future.result()
            x, y = tile_size * (j, i)    # switch (w,h) to (h,w) for numpy
            img[y:y+tile_size[1], x:x+tile_size[0], :] = tile
    return img


def get_slide_mpp(slide: openslide.OpenSlide) -> float:
    try:
        slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        print(f"Slide MPP successfully retrieved from metadata: {slide_mpp}")
    except KeyError:
        # Try out the missing MPP handlers
        try:
            slide_mpp = extract_mpp_from_comments(slide)
            if slide_mpp:
                print(f"MPP retrieved from comments after initial failure: {slide_mpp}")
            else:
                print(f"MPP is missing in the comments of this file format, attempting to extract from metadata...")
                slide_mpp = extract_mpp_from_metadata(slide)
                print(f"MPP re-matched from metadata after initial failure: {slide_mpp}")
        except:
            raise MPPExtractionError("MPP could not be loaded from the slide!")
    return slide_mpp


def extract_mpp_from_metadata(slide: openslide.OpenSlide) -> float:
    import xml.dom.minidom as minidom
    xml_path = slide.properties['tiff.ImageDescription']
    doc = minidom.parseString(xml_path)
    collection = doc.documentElement
    images = collection.getElementsByTagName("Image")
    pixels = images[0].getElementsByTagName("Pixels")
    mpp = float(pixels[0].getAttribute("PhysicalSizeX"))
    return mpp


def extract_mpp_from_comments(slide: openslide.OpenSlide) -> float:
    slide_properties = slide.properties.get('openslide.comment')
    pattern = r'<PixelSizeMicrons>(.*?)</PixelSizeMicrons>'
    match = re.search(pattern, slide_properties)
    if match:
        return float(match.group(1))
    else:
        return None
