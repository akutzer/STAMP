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


def view_as_tiles(
    chunk: np.ndarray, 
    tile_size: Tuple[int, int],
    position: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits a chunk of an image into smaller tiles.

    Parameters:
        chunk (np.ndarray): The chunk to be split into tiles.
        tile_size (Tuple[int, int]): The size of each tile (height, width).
        position (Tuple[int, int]): The (row, column) position of the top-left corner of the chunk.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Array of tiles and their respective positions in the original image.
    """
    h, w, c = chunk.shape
    tile_h, tile_w = tile_size
    
    assert h % tile_h == 0, f"Region height {h} is not divisible by tile height {tile_h}"
    assert w % tile_w == 0, f"Region width {w} is not divisible by tile width {tile_w}"
    
    n_tiles_h = h // tile_h
    n_tiles_w = w // tile_w
    

    tiles = chunk.reshape(n_tiles_h, tile_h, n_tiles_w, tile_w, c)
    tiles = tiles.transpose(0, 2, 1, 3, 4)
    tiles = tiles.reshape(-1, tile_h, tile_w, c)

    # Calculate the positions of the top-left corner of each tile in the original image
    x, y = np.arange(n_tiles_w), np.arange(n_tiles_h)
    xx, yy = np.meshgrid(x, y)
    tile_indices = np.vstack([yy.ravel(), xx.ravel()]).T
    tile_positions = tile_size * tile_indices + position
    
    return tiles, tile_positions


class AsyncChunkLoader:
    def __init__(self, slide: openslide.OpenSlide, target_microns: int = 256, target_tile_size: int = 224,
                 tiles_per_chunk: int = 8, max_workers: int = 4):
        """
        Asynchronous loader to load chunks from a whole slide image (WSI) at a specified resolution.

        Parameters:
            slide (openslide.OpenSlide): The whole slide image object.
            target_microns (int): The target physical size of each chunk in microns.
            target_tile_size (int): The target tile size in pixels for each chunk.
            tiles_per_chunk (int): Number of tiles per chunk (chunk size = tiles_per_chunk * tile_size).
            max_workers (int): Number of threads to use for loading chunks.
        """
        self.slide = slide
        self.target_microns = target_microns
        self.target_tile_size = target_tile_size
        self.tiles_per_chunk = tiles_per_chunk
        self.max_workers = max_workers

        self.slide_mpp = float(get_slide_mpp(slide))
        self.target_mpp = self.target_microns / self.target_tile_size
        
        self.height = math.ceil(self.slide_mpp * slide.dimensions[1] / self.target_mpp)
        self.width = math.ceil(self.slide_mpp * slide.dimensions[0] / self.target_mpp)

        # Calculate tile and chunk sizes in the original resolution
        # These tiles will then get resized to the desired size and MPP value
        self._tile_size = math.ceil(self.target_microns / self.slide_mpp)
        self._chunk_size = self.tiles_per_chunk * self._tile_size
        self._target_chunk_size = self.tiles_per_chunk * self.target_tile_size

        self._chunks = np.ceil(np.array(slide.dimensions) / self._chunk_size).astype(int)
        self.length = self._chunks[0] * self._chunks[1]
        
        self.executor = None
        self.future_to_pos = {}
            
    def __iter__(self) -> 'AsyncChunkLoader':
        """Initialize the executor and prepare to load chunks."""
        self.executor = futures.ThreadPoolExecutor(max_workers=self.max_workers)
        for h in range(self._chunks[1]):  # Rows
            for w in range(self._chunks[0]):  # Columns
                position = (w * self._chunk_size, h * self._chunk_size)
                size = (self._chunk_size, self._chunk_size)
                target_size = (self._target_chunk_size, self._target_chunk_size)
                transform_position = (w * self._target_chunk_size, h * self._target_chunk_size)
                future = self.executor.submit(self.load_chunk, self.slide, position, size, target_size)
                self.future_to_pos[future] = transform_position[::-1]
        return self

    def __len__(self) -> int:
        return self.length

    def __next__(self) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """Return the next chunk and its (row, column) position in the target space."""
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
    def load_chunk(slide: openslide.OpenSlide, position: Tuple[int, int], size: Tuple[int, int],
                    target_size: Tuple[int, int]) -> np.ndarray:
        """
        Load a chunk from the slide, resize it to the target size, and convert it to an RGB array.

        Parameters:
            slide (openslide.OpenSlide): The whole slide image object.
            position (Tuple[int, int]): The (column, row) position of the top-left corner of the chunk.
            size (Tuple[int, int]): The size (width, height) of the chunk to load.
            target_size (Tuple[int, int]): The size (width, height) to which the chunk should be resized.

        Returns:
            np.ndarray: The loaded chunk as a numpy array.
        """
        # if the chunk goes outside the slide then openslide fills these values with 0
        chunk = slide.read_region(position, 0, size).convert('RGB')
        chunk = chunk.resize(target_size)
        return np.array(chunk)


class JPEGChunkLoader:
    def __init__(self, path: Union[Path, str], tile_size: int = 224, tiles_per_chunk: int = 8):
        """
        Loader to divide a JPEG image into chunks and tiles.

        Parameters:
            path (Union[Path, str]): Path to the JPEG image file.
            tile_size (int): The size of each tile in pixels (default is 224).
            tiles_per_chunk (int): Number of tiles per chunk  (default is 8).
        """
        self.img = Image.open(path)
        self.tile_size = tile_size
        self.tiles_per_chunk = tiles_per_chunk
        
        self.height = self.img.height
        self.width = self.img.width

        self.chunk_size = self.tiles_per_chunk * self.tile_size
        self._chunks = np.ceil(np.array([self.width, self.height]) / self.chunk_size).astype(int)
        self.length = self._chunks[0] * self._chunks[1]

        self.current_chunk = 0

    def __iter__(self) -> 'JPEGChunkLoader':
        return self
    
    def __len__(self) -> int:
        return self.length

    def __next__(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Return the next chunk and its (row, column) position in the image."""
        if self.current_chunk >= self.length:
            raise StopIteration
        
        h = self.current_chunk // self._chunks[0]
        w = self.current_chunk % self._chunks[0]
        
        position = (w * self.chunk_size, h * self.chunk_size)
        size = (self.chunk_size, self.chunk_size)
        
        chunk = self.img.crop((position[0], position[1], position[0] + size[0], position[1] + size[1]))
        chunk_array = np.array(chunk)
        
        self.current_chunk += 1
        
        return chunk_array, position[::-1]
