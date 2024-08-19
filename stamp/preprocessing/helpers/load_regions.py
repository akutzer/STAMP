import re
import math
from concurrent import futures
from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
import openslide
from PIL import Image
from .exceptions import MPPExtractionError


# Disable the maximum pixel limit in PIL for large images
Image.MAX_IMAGE_PIXELS = None


class AsyncRegionLoader:
    def __init__(self, slide: openslide.OpenSlide, target_microns: int = 256, target_tile_size: int = 224,
                 tiles_per_region: int = 8, max_workers: int = 4):
        """
        Asynchronous loader to load regions from a whole slide image (WSI) at a specified resolution.

        Parameters:
            slide (openslide.OpenSlide): The whole slide image object.
            target_microns (int): The target physical size of each region in microns.
            target_tile_size (int): The target tile size in pixels for each region.
            tiles_per_region (int): Number of tiles per region (region size = tiles_per_region * tile_size).
            max_workers (int): Number of threads to use for loading regions.
        """
        self.slide = slide
        self.target_microns = target_microns
        self.target_tile_size = target_tile_size
        self.tiles_per_region = tiles_per_region
        self.max_workers = max_workers

        self.slide_mpp = float(get_slide_mpp(slide))
        self.target_mpp = self.target_microns / self.target_tile_size
        
        self.height = math.ceil(self.slide_mpp * slide.dimensions[1] / self.target_mpp)
        self.width = math.ceil(self.slide_mpp * slide.dimensions[0] / self.target_mpp)

        # Calculate tile and region sizes in the original resolution
        # These tiles will then get resized to the desired size and MPP value
        self._tile_size = math.ceil(self.target_microns / self.slide_mpp)
        self._region_size = self.tiles_per_region * self._tile_size
        self._target_region_size = self.tiles_per_region * self.target_tile_size

        self._regions = np.ceil(np.array(slide.dimensions) / self._region_size).astype(int)
        self.length = self._regions[0] * self._regions[1]
        
        self.executor = None
        self.future_to_pos = {}
            
    def __iter__(self) -> 'AsyncRegionLoader':
        """Initialize the executor and prepare to load regions."""
        self.executor = futures.ThreadPoolExecutor(max_workers=self.max_workers)
        for h in range(self._regions[1]):  # Rows
            for w in range(self._regions[0]):  # Columns
                position = (w * self._region_size, h * self._region_size)
                size = (self._region_size, self._region_size)
                target_size = (self._target_region_size, self._target_region_size)
                transform_position = (w * self._target_region_size, h * self._target_region_size)
                future = self.executor.submit(self.load_region, self.slide, position, size, target_size)
                self.future_to_pos[future] = transform_position[::-1]
        return self

    def __len__(self) -> int:
        return self.length

    def __next__(self) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """Return the next region and its (y, x) position in the target space."""
        if not self.future_to_pos:
            self.executor.shutdown(wait=True)
            raise StopIteration
        
        future = next(futures.as_completed(self.future_to_pos))
        pos = self.future_to_pos.pop(future)
        try:
            data = future.result()
        except Exception as exc:
            print(f'{pos} generated an exception: {exc}')
            return None, (-1, -1)
        return data, pos

    @staticmethod
    def load_region(slide: openslide.OpenSlide, position: Tuple[int, int], size: Tuple[int, int],
                    target_size: Tuple[int, int]) -> np.ndarray:
        """
        Load a region from the slide, resize it to the target size, and convert it to an RGB array.

        Parameters:
            slide (openslide.OpenSlide): The whole slide image object.
            position (Tuple[int, int]): The (x, y) position of the top-left corner of the region.
            size (Tuple[int, int]): The size (width, height) of the region to load.
            target_size (Tuple[int, int]): The size (width, height) to which the region should be resized.

        Returns:
            np.ndarray: The loaded region as a numpy array.
        """
        # if the region goes outside the slide then openslide fills these values with 0
        region = slide.read_region(position, 0, size).convert('RGB')
        region = region.resize(target_size)
        return np.array(region)


class JPEGRegionLoader:
    def __init__(self, path: Union[Path, str], tile_size: int = 224, tiles_per_region: int = 8):
        """
        Loader to divide a JPEG image into regions and tiles.

        Parameters:
            path (Union[Path, str]): Path to the JPEG image file.
            tile_size (int): The size of each tile in pixels (default is 224).
            tiles_per_region (int): Number of tiles per region  (default is 8).
        """
        self.img = Image.open(path)
        self.tile_size = tile_size
        self.tiles_per_region = tiles_per_region
        
        self.height = self.img.height
        self.width = self.img.width

        self.region_size = self.tiles_per_region * self.tile_size
        self._regions = np.ceil(np.array([self.width, self.height]) / self.region_size).astype(int)
        self.length = self._regions[0] * self._regions[1]

        self.current_region = 0

    def __iter__(self) -> 'JPEGRegionLoader':
        return self
    
    def __len__(self) -> int:
        return self.length

    def __next__(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Return the next region and its (x, y) position in the image."""
        if self.current_region >= self.length:
            raise StopIteration
        
        h = self.current_region // self._regions[0]
        w = self.current_region % self._regions[0]
        
        position = (w * self.region_size, h * self.region_size)
        size = (self.region_size, self.region_size)
        
        region = self.img.crop((position[0], position[1], position[0] + size[0], position[1] + size[1]))
        region_array = np.array(region)
        
        self.current_region += 1
        
        return region_array, position[::-1]


def get_slide_mpp(slide: openslide.OpenSlide) -> float:
    try:
        slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        print(f"Slide MPP successfully retrieved from metadata: {slide_mpp}")
    except KeyError:
        slide_mpp = _attempt_mpp_retrieval(slide)
    return slide_mpp


def _attempt_mpp_retrieval(slide: openslide.OpenSlide) -> float:
    slide_mpp = extract_mpp_from_comments(slide)
    if slide_mpp:
        print(f"MPP retrieved from comments after initial failure: {slide_mpp}")
    else:
        print("MPP is missing in the comments of this file format, attempting to extract from metadata...")
        slide_mpp = extract_mpp_from_metadata(slide)
        if not slide_mpp:
            raise MPPExtractionError("MPP could not be loaded from the slide!")
        print(f"MPP extracted from metadata: {slide_mpp}")
    return slide_mpp


def extract_mpp_from_metadata(slide: openslide.OpenSlide) -> float:
    from xml.dom import minidom

    try:
        xml_data = slide.properties['tiff.ImageDescription']
        doc = minidom.parseString(xml_data)
        collection = doc.documentElement
        images = collection.getElementsByTagName("Image")
        pixels = images[0].getElementsByTagName("Pixels")
        mpp = float(pixels[0].getAttribute("PhysicalSizeX"))
        return mpp
    except (KeyError, IndexError, AttributeError) as e:
        print(f"Failed to extract MPP from metadata: {e}")
        return None


def extract_mpp_from_comments(slide: openslide.OpenSlide) -> float:
    slide_properties = slide.properties.get('openslide.comment', '')
    pattern = r'<PixelSizeMicrons>(.*?)</PixelSizeMicrons>'
    match = re.search(pattern, slide_properties)
    return float(match.group(1)) if match else None
