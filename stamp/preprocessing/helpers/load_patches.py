import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
from tempfile import mkdtemp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor




def extract_patches(
    img: np.ndarray, patch_size: Tuple[int, int], pad: bool = False, drop_empty: bool = False, overlap: bool = False
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Splits a the whole slide image into smaller patches.
    If `drop_empty`=True completly black patches are removed. This is useful
    when `img` was obtained from an JPEG of an already Canny+normed WSI. 
    """
    kernel_size = np.array(patch_size)
    img_size = np.array(img.shape)[:2]
    stride = kernel_size if not overlap else kernel_size // 2
    if pad:
        rows, cols = np.ceil((img_size - kernel_size) / stride + 1).astype(int)
    else:  # if pad=False, then too small patches at the right and bottom border are getting discarded
        rows, cols = (img_size - kernel_size) // stride + 1
    n_max = rows * cols

    # overestimate the number of non-empty patches
    patches = np.zeros((n_max, kernel_size[0], kernel_size[1], img.shape[-1]), dtype=np.uint8)
    # patches_coords stores the (height, width)-coordinate of the top-left corner for each patch
    patches_coords = np.zeros((n_max, 2), dtype=np.int32)

    k = 0
    for i in range(rows):
        for j in range(cols):
            x, y = (i, j) * stride
            patch = img[x : x + kernel_size[0], y : y + kernel_size[1]]
            
            if drop_empty and not patch.any():  # skip empty/black patches
                continue
            
            # pad on the left and bottom so patches on the edges of the WSI have the same size
            if pad and ((real_shape := np.array(patch.shape[:2])) < kernel_size).any():
                padding = kernel_size - real_shape
                patch = np.pad(
                    patch,
                    pad_width=((0, padding[0]), (0, padding[1]), (0, 0)),
                    mode="mean",
                )

            patches[k] = patch
            patches_coords[k] = (x, y)
            k += 1

    return patches[:k], patches_coords[:k], n_max


def view_as_tiles(region, tile_size, position):
    h, w, c = region.shape
    h_tile, w_tile = tile_size
    
    # Ensure that the height and width are divisible by the tile size
    assert h % h_tile == 0, f"Region height {h} is not divisible by tile height {h_tile}"
    assert w % w_tile == 0, f"Region width {w} is not divisible by tile width {w_tile}"
    
    # Number of tiles along each dimension
    n_tiles_h = h // h_tile
    n_tiles_w = w // w_tile
    
    # Reshape and transpose to get the tiles
    tiles = region.reshape(n_tiles_h, h_tile, n_tiles_w, w_tile, c)
    tiles = tiles.transpose(0, 2, 1, 3, 4)
    
    # Combine the first two dimensions to get the list of tiles
    tiles = tiles.reshape(-1, h_tile, w_tile, c)


    x = np.arange(n_tiles_w) 
    y = np.arange(n_tiles_h)
    xx, yy = np.meshgrid(x, y)
    tile_idx = np.vstack([yy.ravel(), xx.ravel()]).T
    tile_positions = tile_size * tile_idx + position
    
    return tiles, tile_positions


# def reconstruct_from_patches(
#     patches: np.ndarray, patches_coords: np.array, img_shape: Tuple[int, int]
# ) -> np.ndarray:
#     """
#     Reconstruct the WSI from the patches.
#     `patches` is of shape (num_patches, patch_height, patch_width, channels)
#     """
#     img_h, img_w = img_shape
#     patch_h, patch_w = patches.shape[1:3]
#     img = Image.new("RGB", (img_w, img_h))
#     for (x, y), patch in zip(patches_coords, patches):  # (x, y) = (height, width)
#         img.paste(
#             Image.fromarray(patch[:patch_h, :patch_w]),
#             (y, x, y + patch_w, x + patch_h)
#         )
#     return img



class AsyncMemmapImage:
    def __init__(self, shape: Tuple[int, ...], dtype: str = 'uint8', mode: str = 'w+', max_workers: Optional[int] = None):
        self._filename = Path(mkdtemp()) / 'memmap.dat'
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self.fp = np.memmap(self._filename, dtype=dtype, mode=mode, shape=shape)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def write_region(self, data: np.ndarray, position: Tuple[int, int]) -> None:
        """
        Asynchronously writes a chunk of data to the memmap.

        Parameters:
            data (np.ndarray): The data to be written.
            position (Tuple[int, int]): The top-left (h, w) position where the data will be written.
        """
        def write_chunk(fp: np.memmap, data: np.ndarray, position: Tuple[int, int]):
            h_start, w_start = position
            h_end = min(h_start + data.shape[0], self.shape[0])
            w_end = min(w_start + data.shape[1], self.shape[1])

            # slice data if it exceeds the memmap boundaries
            fp[h_start:h_end, w_start:w_end] = data[:h_end - h_start, :w_end - w_start]
            fp.flush()
        
        self.executor.submit(write_chunk, self.fp, data, position)
    
    def write_tiles(self, tiles: np.ndarray, positions: np.ndarray) -> None:
        """
        Asynchronously writes multiple tiles of data to the memmap.

        Parameters:
            tiles (np.ndarray): A 3D or 4D array where each element is a 3D or 2D tile to be written.
            positions (np.ndarray): A 2D array where each element is a (row, column) position for the corresponding tile.
        """
        def write_chunk(fp: np.memmap, tiles: np.ndarray, positions: np.ndarray):
            for tile, position in zip(tiles, positions):
                h_start, w_start = position
                h_end = min(h_start + tile.shape[0], self.shape[0])
                w_end = min(w_start + tile.shape[1], self.shape[1])

                # slice data if it exceeds the memmap boundaries
                fp[h_start:h_end, w_start:w_end] = tile[:h_end - h_start, :w_end - w_start]
            fp.flush()
        
        self.executor.submit(write_chunk, self.fp, tiles, positions)

    def save(self, output_path: Union[Path, str]) -> None:
        """
        Saves the entire memmap data as an image file.

        Parameters:
            output_path (Union[Path, str]): The path where the image file will be saved.
        """
        # ensure all threads are completed before saving the image
        self.executor.shutdown(wait=True)

        if len(self.shape) == 3 and self.shape[2] == 3:  # RGB data
            image_data = self.fp.astype('uint8')
        elif len(self.shape) == 2 or (len(self.shape) == 3 and self.shape[2] == 1):  # Grayscale data
            image_data = self.fp.squeeze().astype('uint8')
        else:
            raise ValueError("Unsupported shape for image conversion. Must be 2D or 3D with 1 or 3 channels.")
        
        image = Image.fromarray(image_data)
        image.save(output_path)

    def close(self) -> None:
        """
        Shuts down the thread pool executor and closes the memmap file.
        """
        self.executor.shutdown(wait=True)
        del self.fp
