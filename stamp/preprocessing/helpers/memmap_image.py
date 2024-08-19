from pathlib import Path
from tempfile import mkdtemp
from typing import Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import threading
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
        print(self._filename)
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self.memmap = np.memmap(self._filename, dtype=dtype, mode=mode, shape=shape)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()

    def write_chunk(self, data: np.ndarray, position: Tuple[int, int]) -> None:
        """
        Asynchronously write a chunk of data to the memmap.

        Parameters:
            data (np.ndarray): Data chunk to be written.
            position (Tuple[int, int]): (row, column) position in the image where the data will be written.
        """
        def _write(memmap: np.memmap, data: np.ndarray, position: Tuple[int, int]):
            h_start, w_start = position
            h_end = min(h_start + data.shape[0], self.shape[0])
            w_end = min(w_start + data.shape[1], self.shape[1])

            with self.lock:
                # slice data if it exceeds the memmap boundaries
                memmap[h_start:h_end, w_start:w_end] = data[:h_end - h_start, :w_end - w_start]
                memmap.flush()
        
        self.executor.submit(_write, self.memmap, data, position)
    
    def write_tiles(self, tiles: np.ndarray, positions: np.ndarray, use_threading: bool = True) -> None:
        """
        Write multiple tiles of data to the memmap, either asynchronously (with threading) or synchronously.

        Parameters:
            tiles (np.ndarray): Array of tiles to be written.
            positions (np.ndarray): Array of (row, column) positions for each tile.
            use_threading (bool): Whether to write asynchronously using threads. Defaults to True.
        """
        def _write(memmap: np.memmap, tiles: np.ndarray, positions: np.ndarray):
            for tile, position in zip(tiles, positions):
                h_start, w_start = position
                h_end = min(h_start + tile.shape[0], self.shape[0])
                w_end = min(w_start + tile.shape[1], self.shape[1])

                with self.lock:
                    # slice data if it exceeds the memmap boundaries
                    memmap[h_start:h_end, w_start:w_end] = tile[:h_end - h_start, :w_end - w_start]
            with self.lock:
                memmap.flush()
        
        if use_threading:
            self.executor.submit(_write, self.memmap, tiles, positions)
        else:
            _write(self.memmap, tiles, positions)

    def save(self, output_path: Union[Path, str]) -> None:
        """
        Save the entire memmap data as an image file.

        Parameters:
            output_path (Union[Path, str]): Path where the image file will be saved.
        """
        # Ensure all threads complete before saving the image
        self.executor.shutdown(wait=True)

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
        self._filename.unlink(missing_ok=True)
        self._filename.parent.rmdir()
