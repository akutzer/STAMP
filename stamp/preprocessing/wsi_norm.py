__author__ = "Omar El Nahhas"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Omar"]
__email__ = "omar.el_nahhas@tu-dresden.de"

import os
from pathlib import Path
import logging
from contextlib import contextmanager
import time
from datetime import timedelta
from typing import Optional, List

import openslide
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch

from .normalizer.normalizer import MacenkoNormalizer
from .extractor.feature_extractors import FeatureExtractor, store_features, store_metadata
from .helpers.common import supported_extensions
from .helpers.exceptions import MPPExtractionError
from .helpers.load_regions import AsyncRegionLoader, JPEGRegionLoader
from .helpers.load_patches import AsyncMemmapImage, view_as_tiles
from .helpers.background_rejection import filter_background

from memory_profiler import profile


Image.MAX_IMAGE_PIXELS = None


def remove_lockfile(file: Path) -> None:
    if file.exists():  # Catch collision cases
        file.unlink()


@contextmanager
def lock_file(slide_path: Path):
    """Create a lock file for a slide to prevent concurrent processing."""
    try:
        (slide_path.with_suffix('.lock')).touch()
    except OSError:
        pass  # No write permissions for WSI directory
    try:
        yield
    finally:
        remove_lockfile(slide_path.with_suffix('.lock'))


def check_write_permissions(directory: Path) -> None:
    testfile = directory / f"test_{os.getpid()}.tmp"
    try:
        testfile.touch()
    except OSError:
        logging.warning(
            "No write permissions for WSI directory! If multiple stamp processes are running "
            "in parallel, the final summary may show an incorrect number of slides processed."
        )
    finally:
        remove_lockfile(testfile)


@profile
def preprocess(
    wsi_dir: Path, output_dir: Path, model_path: Path, cache_dir: Path,
    feature_extractor: str = "ctp", device: str = "cuda", batch_size: int = 64,
    target_microns: int = 256, tile_size: int = 224, cores: int = 8,
    normalize: bool = False, normalization_template: Optional[Path] = None,
    cache: bool = False, keep_dir_structure: bool = False, use_cache: bool = False,
    delete_slide: bool = False
) -> None:
    
    # Remove old lock files
    for lockfile in wsi_dir.glob("**/*.lock"):
        if time.time() - lockfile.stat().st_mtime > 60 * 60:
            remove_lockfile(lockfile)
    check_write_permissions(wsi_dir)
    
    target_mpp = target_microns / tile_size
    tile_size = (tile_size, tile_size)  # (224, 224) by default


    # Initialize the feature extraction model
    print(f"Initializing feature extractor {feature_extractor}...")
    has_gpu = torch.cuda.is_available()
    device = torch.device(device) if "cuda" in device and has_gpu else torch.device("cpu")

    if feature_extractor == "ctp":
        extractor = FeatureExtractor.init_ctranspath(model_path, device)
    elif feature_extractor == "uni":
        extractor = FeatureExtractor.init_uni(device)
    else:
        raise ValueError(f"Invalid feature extractor '{feature_extractor}' selected")


    # Create cache and output directories
    if cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    norm_method = "STAMP_macenko_" if normalize else "STAMP_raw_"
    model_name_norm = Path(norm_method + extractor.model_name)
    output_file_dir = output_dir / model_name_norm
    output_file_dir.mkdir(parents=True, exist_ok=True)


    # Set up logging
    logfile_name = f"logfile_{time.strftime('%Y-%m-%d_%H-%M-%S')}_{os.getpid()}"
    logdir = output_file_dir / logfile_name
    logging.basicConfig(filename=logdir, force=True, level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())
    
    logging.info(f"Preprocessing started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Normalize: {normalize} | Target Microns: {target_microns} | Tile Size: {tile_size} | MPP: {target_mpp}")
    logging.info(f"Model: {extractor.model_name}\n")

    print(f"Current working directory: {os.getcwd()}")
    print(f"Stored logfile in {logdir}")
    print(f"Number of CPUs in the system: {os.cpu_count()}")
    print(f"Number of CPU cores used: {cores}")
    print(f"GPU is available: {has_gpu}")
    if has_gpu:
        print(f"Number of GPUs in the system: {torch.cuda.device_count()}, using device {device}")
    
    if normalize:
        raise NotImplementedError("Normalization is not implemented yet.")
        if normalization_template is None:
            raise ValueError("`normalization_template` can't be None if `normalize` is True")
        
        logging.info("Initializing Macenko normalizer...")
        logging.info(f"Reference: {normalization_template}")
        target_image = Image.open(normalization_template).convert("RGB")
        normalizer = MacenkoNormalizer().fit(np.array(target_image))


    total_start_time = time.time()
    # TODO: remove
    # img_name = "norm_slide.jpg" if norm else "canny_slide.jpg"


    # Scan for existing feature files
    logging.info("Scanning for existing feature files...")
    existing_slides = [f.stem for f in output_file_dir.glob("**/*.h5")] if output_file_dir.exists() else []

    slides = [slide for ext in supported_extensions for slide in wsi_dir.glob(f"**/*{ext}")]
    slides_to_process = [slide for slide in slides if slide.stem not in existing_slides]

    num_total = len(slides)
    num_processed, num_skipped = len(existing_slides), 0
    error_slides: List[str] = []

    if existing_slides:
        logging.info(f"Skipping {len(existing_slides)} already processed slides out of {num_total} total slides...")


    store_metadata(
        outdir=output_file_dir,
        extractor_name=extractor.name,
        tile_size=tile_size,
        target_microns=target_microns,
        normalized=normalize
    )
    
    # random.shuffle(slides_to_process)
    for slide_path in tqdm(slides_to_process, desc="\nPreprocessing progress", leave=False, miniters=1, mininterval=0):
        slide_name = slide_path.stem
        slide_cache_dir = cache_dir / slide_name
        if cache:
            slide_cache_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"\n\n===== Processing slide {slide_name} =====")
        slide_subdir = slide_path.parent.relative_to(wsi_dir)

        if not keep_dir_structure or slide_subdir == Path("."):
            feature_output_path = output_file_dir / slide_name
        else:
            (output_file_dir / slide_subdir).mkdir(parents=True, exist_ok=True)
            feature_output_path = output_file_dir / slide_subdir / slide_name


        if not feature_output_path.with_suffix('.h5').exists() and not slide_path.with_suffix('.lock').exists():
            start_loading_time = time.time()
            with lock_file(slide_path):
                if (slide_jpg := slide_cache_dir / "canny_slide.jpg").exists() and use_cache:
                    try:
                        region_loader = JPEGRegionLoader(slide_jpg, tile_size=tile_size[0])
                        print("JPEGRegionLoader")
                    except Exception as e:
                        logging.error(f"Failed loading cached slide, trying to load WSI... Error: {e}")
                        if use_cache:
                            error_slides.append(slide_name)
                            continue
                
                else:
                    try:
                        slide = openslide.OpenSlide(slide_path)
                        original_slide_size = slide.dimensions
                        region_loader = AsyncRegionLoader(
                            slide, target_microns=target_microns, target_tile_size=tile_size[0], max_workers=cores * 2
                        )
                    except openslide.lowlevel.OpenSlideUnsupportedFormatError:
                        logging.error("Unsupported format for slide, skipping...")
                        error_slides.append(slide_name)
                        continue
                    except Exception as e:
                        logging.error(f"Failed loading slide, skipping... Error: {e}")
                        error_slides.append(slide_name)
                        continue
                
                try:
                    slide_size = (region_loader.height, region_loader.width)
                    embeddings, coords = [], []

                    if cache and not slide_jpg.exists():
                        canny_img = AsyncMemmapImage(shape=(*slide_size, 3), max_workers=cores * 2)

                    total_rejected, total_tiles = 0, 0

                    for region, position in tqdm(region_loader, leave=False):
                        # `region`: 3D numpy array of shape (region_height, region_width, 3) representing the current region of the WSI
                        # `position`: (y, x) tuple representing tje position of the top-left corner of a region

                        if region is None:
                            continue
                            
                        # Break the region into small tiles and get their coordinates
                        # `tiles`: (num_tiles, tile_height, tile_width, 3)
                        # `tile_coords`: (num_tiles, 2) where each row represents (y, x) position of the top-left corner of a tile
                        tiles, tile_coords = view_as_tiles(region, tile_size, position)
                        
                        # Remove completely black tiles (i.e., tiles where all pixel values are 0 across all channels)
                        # These tiles may be outside the slide's boundaries or have been rejected by previous processing steps (in the case of reading from a canny_slide.jpg)
                        non_empty_tiles = tiles.any(axis=(-3, -2, -1))
                        tiles = tiles[non_empty_tiles, ...]
                        tile_coords = tile_coords[non_empty_tiles, ...]
                        total_tiles += tiles.shape[0]

                        # Filter out tiles that are considered background by Canny filter
                        tiles, tile_coords, num_rejected = filter_background(tiles, tile_coords)
                        total_rejected += num_rejected

                        if tiles.shape[0] == 0:
                            continue
                        
                        if cache and not slide_jpg.exists():
                            canny_img.write_tiles(tiles, tile_coords)

                        embeddings.append(extractor.extract(tiles, cores=0, batch_size=min(tiles.shape[0], batch_size)))
                        coords.append(tile_coords)
                
                except MPPExtractionError:
                    if delete_slide:
                        logging.error("MPP missing in slide metadata, deleting slide and skipping...")
                        slide_path.unlink(missing_ok=True)
                    else:
                        logging.error("MPP missing in slide metadata, skipping...")
                    error_slides.append(slide_name)
                    continue
                except openslide.lowlevel.OpenSlideError as e:
                    logging.error(f"Failed loading slide, skipping... Error: {e}")
                    error_slides.append(slide_name)
                    continue
                
                del region_loader
                if not slide_jpg.exists():
                    del slide
            
                embeddings = np.concatenate(embeddings, axis=0)
                coords = np.concatenate(coords, axis=0) 

                if embeddings.shape[0] > 0:
                    # Order embeddings row and then column-wise
                    max_width = coords[:, 1].max()
                    idx = coords[:, 0] * max_width + coords[:, 1]
                    sort_indices = np.argsort(idx)
                    embeddings, coords = embeddings[sort_indices], coords[sort_indices]

                    store_features(feature_output_path, embeddings, coords, extractor.name)
                    num_processed += 1
                else:
                    logging.warning("No tiles remain for feature extraction after preprocessing. Skipping...")
                    continue
                
                if cache and not slide_jpg.exists():
                    canny_img.save(slide_cache_dir / "canny_slide.jpg")
                    del canny_img

                logging.info(f"Successfully preprocessed slide ({time.time() - start_loading_time:.2f} seconds)")
                if not slide_jpg.exists():
                    logging.info(f"Reshaped original WSI of shape {original_slide_size} -> {slide_size}")
                logging.info(f"Canny background rejection: rejected {total_rejected}/{total_tiles} tiles")
                logging.info(f"Embedded {total_tiles - total_rejected} tiles in total")

        else:
            if feature_output_path.with_suffix('.h5').exists():
                logging.info(".h5 file for this slide already exists. Skipping...")
            else:
                logging.info("Slide is already being processed. Skipping...")
            num_skipped += 1
            if delete_slide:
                print("Deleting slide from local folder...")
                slide_path.unlink(missing_ok=True)

    logging.info(f"\n\n\n===== End-to-end processing time of {num_total} slides: {str(timedelta(seconds=(time.time() - total_start_time)))} =====")
    logging.info(f"Summary: Processed {num_processed} slides, encountered {len(error_slides)} errors, skipped {num_skipped} slides")
    if error_slides:
        logging.info("The following slides were not processed due to errors:\n  " + "\n  ".join(error_slides))
