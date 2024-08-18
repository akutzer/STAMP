import re
from typing import Tuple
import concurrent
import openslide
import numpy as np
from PIL import Image
import math
from PIL import Image
from typing import Union, Tuple
from pathlib import Path
import numpy as np

from .exceptions import MPPExtractionError


Image.MAX_IMAGE_PIXELS = None


class AsyncRegionLoader:
    def __init__(self, slide: openslide.OpenSlide, target_microns: int = 256, target_tile_size: int = 224, tiles_per_region: int = 8, max_workers: int = 4):
        self.slide = slide
        self.target_microns = target_microns
        self.target_tile_size = target_tile_size
        self.tiles_per_region = tiles_per_region
        self.max_workers = max_workers
        
        self.slide_mpp = float(get_slide_mpp(slide))
        self.target_mpp = self.target_microns / self.target_tile_size
        self.height = math.ceil(self.slide_mpp * slide.dimensions[1] / self.target_mpp)
        self.width = math.ceil(self.slide_mpp * slide.dimensions[0] / self.target_mpp)

        # calculate tile and region size for the source MPP
        # these tiles will then get resized to the desired size and MPP value
        self._tile_size = math.ceil(self.target_microns / self.slide_mpp)
        self._region_size = self.tiles_per_region * self._tile_size
        self._target_region_size = self.tiles_per_region * self.target_tile_size

        self._regions = np.ceil(np.array(slide.dimensions) / self._region_size).astype(int)
        self.length = self._regions[0] * self._regions[1]
            
    def __iter__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.future_to_pos = {}
        for h in range(self._regions[1]):  # row
            for w in range(self._regions[0]):  # column
                position = (w * self._region_size, h * self._region_size)
                size = (self._region_size, self._region_size)
                target_size = (self._target_region_size, self._target_region_size)
                transform_position = (w * self._target_region_size, h * self._target_region_size)
                self.future_to_pos[self.executor.submit(self.load_region, self.slide, position, size, target_size)] = transform_position[::-1]
        return self

    def __len__(self) -> int:
        return self.length
        
    def __next__(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        if not self.future_to_pos:
            self.executor.shutdown(wait=True)
            raise StopIteration
        
        future = next(concurrent.futures.as_completed(self.future_to_pos))
        pos = self.future_to_pos.pop(future)
        try:
            data = future.result()
        except Exception as exc:
            print(f'{pos} generated an exception: {exc}')
            return None, -1
        else:
            return data, pos
    
    @staticmethod
    def load_region(slide, position, size, target_size):
        region = slide.read_region(position, 0, size).convert('RGB')
        # if the region goes outside the slide then openslide fills these values with 0
        region = region.resize(target_size)
        
        region = np.array(region)
        return region


class JPEGRegionLoader:
    def __init__(self, path: Union[Path, str], tile_size: int = 224, tiles_per_region: int = 8):
        """
        Initializes the JPEGRegionLoader. Stores the entire JPEG in memory!

        Parameters:
            path (Union[Path, str]): Path to the JPEG image file.
            tile_size (int): The size of each tile (default is 224).
            tiles_per_region (int): Number of tiles per region (default is 8).
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

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, Tuple[int, int]]:
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

    def __len__(self) -> int:
        return self.length



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
