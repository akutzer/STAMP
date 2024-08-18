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
from .helpers.load_regions import  AsyncRegionLoader
from .helpers.load_patches import extract_patches, view_as_tiles
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
    except PermissionError:
        pass # No write permissions for wsi directory
    try:
        yield
    finally:
        clean_lockfile(f"{slide_path}.lock")


def test_wsidir_write_permissions(wsi_dir: Path):
    try:
        testfile = wsi_dir/f"test_{str(os.getpid())}.tmp"
        Path(testfile).touch()
    except PermissionError:
        logging.warning("No write permissions for wsi directory! If multiple stamp processes are running "
                        "in parallel, the final summary may show an incorrect number of slides processed.")
    finally:
        clean_lockfile(testfile)


def save_image(image, path: Path):
    width, height = image.size
    if width > 65500 or height > 65500:
        logging.warning(f"Image size ({width}x{height}) exceeds maximum size of 65500x65500, resizing {path.name} before saving...")
        ratio = 65500 / max(width, height)
        image = image.resize((int(width * ratio), int(height * ratio)))
    image.save(path)


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
        raise Exception(f"Invalid feature extractor '{feat_extractor}' selected")

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
    test_wsidir_write_permissions(wsi_dir)

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
    if not only_feature_extraction:
        img_dir = [svs for ext in supported_extensions for svs in wsi_dir.glob(f"**/*{ext}")]
        existing = [f for f in existing if f in [f.stem for f in img_dir]]
        img_dir = [f for f in img_dir if f.stem not in existing]
    else:
        raise NotImplementedError()
        if not cache_dir.exists():
            logging.error("Cache directory does not exist, cannot extract features from cached slides!")
            exit(1)
        img_dir = [jpg for jpg in cache_dir.glob(f"**/*/{img_name}")]
        existing = [f for f in existing if f in [f.parent.name for f in img_dir]]
        img_dir = [f for f in img_dir if f.parent.name not in existing]

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

    for slide_url in tqdm(img_dir, "\nPreprocessing progress", leave=False, miniters=1, mininterval=0):
        slide_name = slide_url.stem if not only_feature_extraction else slide_url.parent.name
        slide_cache_dir = cache_dir/slide_name
        if cache:
            # raise NotImplementedError()
            slide_cache_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"\n\n===== Processing slide {slide_name} =====")
        slide_subdir = slide_url.parent.relative_to(wsi_dir)
        if not keep_dir_structure or slide_subdir == Path("."):
            feat_out_dir = output_file_dir/slide_name
        else:
            (output_file_dir/slide_subdir).mkdir(parents=True, exist_ok=True)
            feat_out_dir = output_file_dir/slide_subdir/slide_name
        if not os.path.exists((f"{feat_out_dir}.h5")) and not os.path.exists(f"{slide_url}.lock"):
            with lock_file(slide_url):
                if (
                    (only_feature_extraction and (slide_jpg := slide_url).exists()) or \
                    (slide_jpg := slide_cache_dir/"norm_slide.jpg").exists()
                ):
                    raise NotImplementedError()
                    slide_array = np.array(Image.open(slide_jpg))
                    patches, patches_coords, n = extract_patches(slide_array, patch_size, pad=False, drop_empty=True, overlap=False)
                    logging.info(f"Loaded {img_name}, {patches.shape[0]}/{n} tiles remain")
                    # note that due to being stored as an JPEG rejected patches which
                    # neighbor accepted patches will most likely also be loaded
                    # thus we again apply a background filtering
                    patches, patches_coords = filter_background(patches, patches_coords, cores)
                    # patches.shape = (n_patches, patch_h, patch_w, 3)
                    # patches_coords.shape = (n_patches, 2)
                else:
                    try:
                        slide = openslide.OpenSlide(slide_url)
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
                        region_loader = AsyncRegionLoader(slide, target_microns=target_microns, target_tile_size=patch_size[0], max_workers=cores)
                        embeddings, coords = [], []

                        if cache:
                            canny_img = Image.new(mode="RGB", size=(region_loader.width, region_loader.height))
                            # canny_img = Image.new(mode="L", size=(region_iter.width, region_iter.height))

                        total_rejected, total_tiles = 0, 0
                        for region, position in tqdm(region_loader, leave=False):
                            if region is None: continue

                            tiles, tile_coords = view_as_tiles(region, patch_size, position)

                            non_empty = tiles.any(axis=(-3, -2, -1))
                            tiles = tiles[non_empty, ...]
                            tile_coords = tile_coords[non_empty, ...]
                            total_tiles += tiles.shape[0]

                            tiles, tile_coords, num_rejected = filter_background(tiles, tile_coords, cores)
                            total_rejected += num_rejected
                            # logging.info(f"Finished Canny background rejection, rejected {n_tiles-tiles.shape[0]}/{n_tiles} tiles.")

                            if cache:
                                for tile, tile_coord in zip(tiles, tile_coords):
                                    tile_img = Image.fromarray(tile)
                                    # convert (h, w) -> (w, h) for PIL
                                    tile_coord = tuple(tile_coord[::-1])

                                    # import cv2
                                    # tile_lumen = np.array(tile_img.convert("L"))
                                    # tile_img = Image.fromarray(cv2.Canny(tile_lumen, 40, 100, L2gradient=True))
         
                                    canny_img.paste(tile_img, tile_coord)
                            
                            if tiles.shape[0] > 0:
                                embeddings.append(extractor.extract(tiles, cores=0, batch_size=min(tiles.shape[0], 64)))
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
                    # del slide   # Remove .SVS from memory
                
                    if cache:
                        canny_img.save(slide_cache_dir/"canny_slide.jpg")
                        del canny_img
                    
                    embeddings = np.concatenate(embeddings, axis=0)
                    coords = np.concatenate(coords, axis=0)
                    if embeddings.shape[0] > 0:
                        store_features(feat_out_dir, embeddings, coords, extractor.name)
                        num_processed += 1
                    else:
                        logging.warning("No tiles remain for feature extraction after pre-processing. Continuing...")

                    logging.info(f"Successfully preprocessed slide ({time.time() - start_loading:.2f} seconds)")
                    logging.info(f"Reshaped original WSI of shape {slide.dimensions} -> {(region_loader.width, region_loader.height)}")
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
