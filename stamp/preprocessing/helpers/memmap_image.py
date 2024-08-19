from pathlib import Path
from tempfile import mkdtemp
from typing import Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image



class AsyncMemmapImage:
    def __init__(
        self, 
        shape: Tuple[int, ...], 
        dtype: str = 'uint8', 
        mode: str = 'w+', 
        max_workers: Optional[int] = None
    ):
        self._filename = Path(mkdtemp()) / 'memmap.dat'
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self.memmap = np.memmap(self._filename, dtype=dtype, mode=mode, shape=shape)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def write_chunk(self, data: np.ndarray, position: Tuple[int, int]) -> None:
        """
        Asynchronously write a chunk of data to the memmap.

        Parameters:
            data (np.ndarray): Data chunk to be written.
            position (Tuple[int, int]): (row, column) position in the image where the data will be written.
        """
        def _write(fp: np.memmap, data: np.ndarray, position: Tuple[int, int]):
            h_start, w_start = position
            h_end = min(h_start + data.shape[0], self.shape[0])
            w_end = min(w_start + data.shape[1], self.shape[1])

            # slice data if it exceeds the memmap boundaries
            fp[h_start:h_end, w_start:w_end] = data[:h_end - h_start, :w_end - w_start]
            fp.flush()
        
        self.executor.submit(_write, self.memmap, data, position)
    
    def write_tiles(self, tiles: np.ndarray, positions: np.ndarray) -> None:
        """
        Asynchronously write multiple tiles of data to the memmap.

        Parameters:
            tiles (np.ndarray): Array of tiles to be written.
            positions (np.ndarray): Array of (row, column) positions for each tile.
        """
        def _write(fp: np.memmap, tiles: np.ndarray, positions: np.ndarray):
            for tile, position in zip(tiles, positions):
                h_start, w_start = position
                h_end = min(h_start + tile.shape[0], self.shape[0])
                w_end = min(w_start + tile.shape[1], self.shape[1])

                # slice data if it exceeds the memmap boundaries
                fp[h_start:h_end, w_start:w_end] = tile[:h_end - h_start, :w_end - w_start]
            fp.flush()
        
        self.executor.submit(_write, self.memmap, tiles, positions)

    def save(self, output_path: Union[Path, str]) -> None:
        """
        Save the entire memmap data as an image file.

        Parameters:
            output_path (Union[Path, str]): Path where the image file will be saved.
        """
        # Ensure all threads complete before saving the image
        print("Waiting for shutdown")
        self.executor.shutdown(wait=True)
        print("All shutdown")

        if len(self.shape) == 3 and self.shape[2] == 3:  # RGB data
            image_data = self.memmap.astype('uint8')
        elif len(self.shape) == 2 or (len(self.shape) == 3 and self.shape[2] == 1):  # Grayscale data
            image_data = self.memmap.squeeze().astype('uint8')
        else:
            raise ValueError("Unsupported shape for image conversion. Must be 2D or 3D with 1 or 3 channels.")
        
        image = Image.fromarray(image_data)
        image.save(output_path)

    def close(self) -> None:
        """
        Shut down the thread pool executor and close the memmap file.
        """
        self.executor.shutdown(wait=True)
        del self.memmap
