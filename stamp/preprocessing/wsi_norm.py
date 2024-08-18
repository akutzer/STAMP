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
from typing import Optional
import random

import openslide
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch

from .normalizer.normalizer import MacenkoNormalizer
from .extractor.feature_extractors import FeatureExtractor, store_features, store_metadata
from .helpers.common import supported_extensions
from .helpers.exceptions import MPPExtractionError
from .helpers.load_regions import  AsyncRegionLoader, JPEGRegionLoader
from .helpers.load_patches import AsyncMemmapImage, view_as_tiles
from .helpers.background_rejection import filter_background


from memory_profiler import profile



Image.MAX_IMAGE_PIXELS = None


def clean_lockfile(file):
    if os.path.exists(file): # Catch collision cases
        os.remove(file)

@contextmanager
def lock_file(slide_path: Path):
    try:
        Path(f"{slide_path}.lock").touch()
    except OSError:
        pass # No write permissions for wsi directory
    try:
        yield
    finally:
        clean_lockfile(f"{slide_path}.lock")


def test_wsidir_write_permissions(wsi_dir: Path):
    try:
        testfile = wsi_dir/f"test_{str(os.getpid())}.tmp"
        Path(testfile).touch()
    except OSError:
        logging.warning("No write permissions for wsi directory! If multiple stamp processes are running "
                        "in parallel, the final summary may show an incorrect number of slides processed.")
    finally:
        clean_lockfile(testfile)


@profile
def preprocess(output_dir: Path, wsi_dir: Path, model_path: Path, cache_dir: Path,
               cache: bool = False, norm: bool = False, normalization_template: Optional[Path] = None,
               del_slide: bool = False, only_feature_extraction: bool = False,
               keep_dir_structure: bool = False, cores: int = 8, target_microns: int = 256,
               patch_size: int = 224, batch_size: int = 64, device: str = "cuda",
               feat_extractor: str = "ctp"
               ):

    # Clean up potentially old leftover .lock files
    for lockfile in wsi_dir.glob("**/*.lock"):
        if time.time() - os.path.getmtime(lockfile) > 60 * 60:
            clean_lockfile(lockfile)
    test_wsidir_write_permissions(wsi_dir)
    
    target_mpp = target_microns / patch_size
    patch_size = (patch_size, patch_size) # (224, 224) by default

    # Initialize the feature extraction model
    print(f"Initialising feature extractor {feat_extractor}...")
    has_gpu = torch.cuda.is_available()
    device = torch.device(device) if "cuda" in device and has_gpu else torch.device("cpu")
    if feat_extractor == "ctp":
        extractor = FeatureExtractor.init_ctranspath(model_path, device)
    elif feat_extractor == "uni":
        extractor = FeatureExtractor.init_uni(device)
    else:
        raise ValueError(f"Invalid feature extractor '{feat_extractor}' selected")

    # Create cache and output directories
    if cache:
        cache_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_method = "STAMP_macenko_" if norm else "STAMP_raw_"
    model_name_norm = Path(norm_method + extractor.model_name)
    output_file_dir = output_dir/model_name_norm
    output_file_dir.mkdir(parents=True, exist_ok=True)
    if cache:
        cache_dir.mkdir(exist_ok=True, parents=True)

    # Create logfile and set up logging
    logfile_name = "logfile_" + time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(os.getpid())
    logdir = output_file_dir/logfile_name
    logging.basicConfig(filename=logdir, force=True, level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Preprocessing started at: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info(f"Norm: {norm} | Target_microns: {target_microns} | Patch_size: {patch_size} | MPP: {target_mpp}")
    logging.info(f"Model: {extractor.model_name}\n")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Stored logfile in {logdir}")
    print(f"Number of CPUs in the system: {os.cpu_count()}")
    print(f"Number of CPU cores used: {cores}")
    print(f"GPU is available: {has_gpu}")
    if has_gpu:
        print(f"Number of GPUs in the system: {torch.cuda.device_count()}, using device {device}")

    if norm:
        raise NotImplementedError()
        assert normalization_template is not None, "`normalization_template` can't be None if `norm`=True"
        logging.info("Initializing Macenko normalizer...")
        logging.info(f"Reference: {normalization_template}")
        target = Image.open(normalization_template).convert("RGB")
        normalizer = MacenkoNormalizer().fit(np.array(target))

    total_start_time = time.time()
    img_name = "norm_slide.jpg" if norm else "canny_slide.jpg"

    # Get list of slides, filter out slides that have already been processed
    logging.info("Scanning for existing feature files...")
    existing = [f.stem for f in output_file_dir.glob("**/*.h5")] if output_file_dir.exists() else []

    img_dir = [svs for ext in supported_extensions for svs in wsi_dir.glob(f"**/*{ext}")]
    existing = [f for f in existing if f in [f.stem for f in img_dir]]
    img_dir = [f for f in img_dir if f.stem not in existing]

    # random.shuffle(img_dir)
    num_processed, num_total = 0, len(img_dir) + len(existing)
    error_slides = []
    if len(existing):
        logging.info(f"For {len(existing)} out of {num_total} slides in the wsi directory feature files were found, skipping these slides...")

    
    store_metadata(
        outdir=output_file_dir,
        extractor_name=extractor.name,
        patch_size=patch_size,
        target_microns=target_microns,
        normalized=norm
    )

    for slide_url in tqdm(img_dir, desc="\nPreprocessing progress", leave=False, miniters=1, mininterval=0):
        slide_name = slide_url.stem
        slide_cache_dir = cache_dir/slide_name
        if cache: slide_cache_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"\n\n===== Processing slide {slide_name} =====")
        slide_subdir = slide_url.parent.relative_to(wsi_dir)
        if not keep_dir_structure or slide_subdir == Path("."):
            feat_out_dir = output_file_dir/slide_name
        else:
            (output_file_dir/slide_subdir).mkdir(parents=True, exist_ok=True)
            feat_out_dir = output_file_dir/slide_subdir/slide_name
        if not os.path.exists((f"{feat_out_dir}.h5")) and not os.path.exists(f"{slide_url}.lock"):
            with lock_file(slide_url):

                if using_cache := (slide_jpg := slide_cache_dir/"canny_slide.jpg").exists():
                    try:
                        region_loader = JPEGRegionLoader(slide_jpg, tile_size=patch_size[0])
                    except Exception as e:
                        logging.error(f"Failed loading slide, continuing... Error: {e}")
                        error_slides.append(slide_name)
                        continue
                else:
                    try:
                        slide = openslide.OpenSlide(slide_url)
                        original_slide_size = slide.dimensions
                        region_loader = AsyncRegionLoader(slide, target_microns=target_microns, target_tile_size=patch_size[0], max_workers=cores*2)
                    except openslide.lowlevel.OpenSlideUnsupportedFormatError:
                        logging.error("Unsupported format for slide, continuing...")
                        error_slides.append(slide_name)
                        continue
                    except Exception as e:
                        logging.error(f"Failed loading slide, continuing... Error: {e}")
                        error_slides.append(slide_name)
                        continue
                    
                start_loading = time.time()
                try:
                    slide_size = (region_loader.height, region_loader.width)
                    embeddings, coords = [], []

                    if cache and not using_cache:
                        canny_img = AsyncMemmapImage(shape=(*slide_size, 3), max_workers=cores*2)

                    total_rejected, total_tiles = 0, 0
                    for region, position in tqdm(region_loader, leave=False):
                        if region is None: continue

                        tiles, tile_coords = view_as_tiles(region, patch_size, position)

                        non_empty = tiles.any(axis=(-3, -2, -1))
                        tiles = tiles[non_empty, ...]
                        tile_coords = tile_coords[non_empty, ...]
                        total_tiles += tiles.shape[0]

                        tiles, tile_coords, num_rejected = filter_background(tiles, tile_coords)
                        total_rejected += num_rejected

                        if tiles.shape[0] == 0: continue
                        if cache and not using_cache:
                            canny_img.write_tiles(tiles, tile_coords)
    
                        embeddings.append(extractor.extract(tiles, cores=0, batch_size=min(tiles.shape[0], batch_size)))
                        coords.append(tile_coords)
                    
                except MPPExtractionError:
                    if del_slide:
                        logging.error("MPP missing in slide metadata, deleting slide and continuing...")
                        if os.path.exists(slide_url):
                            os.remove(slide_url)
                    else:
                        logging.error("MPP missing in slide metadata, continuing...")
                    error_slides.append(slide_name)
                    continue
                except openslide.lowlevel.OpenSlideError as e:
                    logging.error(f"Failed loading slide, continuing... Error: {e}")
                    error_slides.append(slide_name)
                    continue
                
                del region_loader
                if not using_cache: del slide
            
                embeddings = np.concatenate(embeddings, axis=0)
                coords = np.concatenate(coords, axis=0)
                if embeddings.shape[0] > 0:
                    # TODO: order embeddings?
                    store_features(feat_out_dir, embeddings, coords, extractor.name)
                    num_processed += 1
                else:
                    logging.warning("No tiles remain for feature extraction after pre-processing. Continuing...")
                    continue

                if cache and not using_cache:
                    canny_img.save(slide_cache_dir/"canny_slide.jpg")
                    del canny_img

                logging.info(f"Successfully preprocessed slide ({time.time() - start_loading:.2f} seconds)")
                if not using_cache:
                    logging.info(f"Reshaped original WSI of shape {original_slide_size} -> {slide_size}")
                logging.info(f"Canny background rejection, rejected {total_rejected}/{total_tiles} tiles")
                logging.info(f"Embedded {total_tiles - total_rejected} tiles in total")
    
        else:
            if os.path.exists((f"{feat_out_dir}.h5")):
                logging.info(".h5 file for this slide already exists. Skipping...")
            else:
                logging.info("Slide is already being processed. Skipping...")
            existing.append(slide_name)
            if del_slide:
                print("Deleting slide from local folder...")
                if os.path.exists(slide_url):
                    os.remove(slide_url)

    logging.info(f"\n\n\n===== End-to-end processing time of {num_total} slides: {str(timedelta(seconds=(time.time() - total_start_time)))} =====")
    logging.info(f"Summary: Processed {num_processed} slides, encountered {len(error_slides)} errors, skipped {len(existing)} readily-processed slides")
    if len(error_slides):
        logging.info("The following slides were not processed due to errors:\n  " + "\n  ".join(error_slides))
