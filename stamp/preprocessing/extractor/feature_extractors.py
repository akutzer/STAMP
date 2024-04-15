import json
from pathlib import Path
from typing import Tuple
import hashlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from torchvision.transforms import ToTensor
import numpy as np
import h5py
from tqdm import tqdm
import os
import uni

from stamp.preprocessing.extractor.swin_transformer import swin_tiny_patch4_window7_224, ConvStem


__version__ = "1.0.3_15-04-2024"


def get_digest(file: str):
    sha256 = hashlib.sha256()
    with open(file, 'rb') as f:
        while True:
            data = f.read(1 << 16)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


class FeatureExtractor:
    """Extracts features from slide tiles."""

    def __init__(self, model: str, model_name: str, transform = None, device: str = "cpu"):
        self.model_name = model_name
        self.model_type = "CTransPath"
        self.name = f"STAMP-extract-{__version__}_{model_name}"

        self.model = model
        self.model.to(device)
        self.model.eval()
        self.transform = transform

        self.device = torch.device(device)
        self.dtype = next(self.model.parameters()).dtype

    @classmethod
    def init_ctranspath(cls, checkpoint_path: str, device: str) -> "FeatureExtractor":
        # loading the checkpoint weights
        digest = get_digest(checkpoint_path)
        assert digest == "7c998680060c8743551a412583fac689db43cec07053b72dfec6dcd810113539"

        model_name = f"xiyuewang-ctranspath-{digest[:8]}"
        ctranspath_weights = torch.load(checkpoint_path, map_location=torch.device("cpu"))

        # initializing the model and updating the weights
        model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
        model.head = nn.Identity()
        model.load_state_dict(ctranspath_weights["model"], strict=True)

        transform = transforms.Compose([
            transforms.ToImage(),  # convert to tensor, only needed for PIL images
            transforms.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
            transforms.Resize(size=(224, 224), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),  # normalize expects float input
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization
        ])

        extractor = cls(model, model_name, transform, device)
        print("CTransPath model successfully initialized...\n")

        return extractor
    
    @classmethod
    def init_uni(cls, device: str, **kwargs) -> "FeatureExtractor":
        """Extracts features from slide tiles. 
        Requirements: 
            Permission from authors via huggingface: https://huggingface.co/MahmoodLab/UNI
            Huggingface account with valid login token
        On first model initialization, you will be prompted to enter your login token. The token is
        then stored in ./home/<user>/.cache/huggingface/token. Subsequent inits do not require you to re-enter the token. 

        Args:
            device: "cuda" or "cpu"
        """
        
        # loading the checkpoint weights
        asset_dir = f"{os.environ['STAMP_RESOURCES_DIR']}/uni"
        digest = get_digest(f"{asset_dir}/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin")
        model_name = f"mahmood-uni-{digest[:8]}"

        # initializing the model and updating the weights
        model, transform = uni.get_encoder(enc_name="uni", device=device, assets_dir=asset_dir, center_crop=False, **kwargs)
        # UNI implementations still uses transformsV1, which expects
        # either a PILImage or Tensor as input, not a numpy array like in the current code,
        # thus we have to move the ToTensor transform to the beginning 
        transform.transforms = list(filter(lambda x: not isinstance(x, ToTensor), transform.transforms))
        transform.transforms.insert(0, ToTensor())

        extractor = cls(model, model_name, transform, device)
        print("UNI model successfully initialized...\n")

        return extractor

    def extract(
        self, patches: np.ndarray, cores: int = 8, batch_size: int = 64
    ) -> np.ndarray:
        """Extracts features from slide tiles.

        Args:
            patches:  Array of shape (n_patches, patch_h, patch_w, 3)
            cores:  Number of cores for dataloader
        """
        dataset = SlideTileDataset(patches, self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cores,
            drop_last=False,
            pin_memory=self.device != torch.device("cpu"),
        )

        features = []
        with torch.inference_mode():
            for patches_batch in tqdm(dataloader, leave=False):
                patches_batch = patches_batch.to(dtype=self.dtype, device=self.device)
                features_batch = self.model(patches_batch).half().cpu()
                features.append(features_batch)

        features = torch.concat(features, dim=0).numpy()
        return features


def store_metadata(
    outdir: Path,
    extractor_name: str,
    patch_size: Tuple[int, int],
    target_microns: int,
    normalized: bool,
):
    with open(outdir.parent / "info.json", "w") as f:
        json.dump({
            "extractor": extractor_name,
            "augmented_repetitions": 0,
            "patches_normalized": normalized,
            "microns": target_microns,
            "patch_size": patch_size,
        }, f)


def store_features(
    outdir: Path, features: np.ndarray, patches_coords: np.ndarray, extractor_name: str
):
    with h5py.File(f"{outdir}.h5", "w") as f:
        f["coords"] = patches_coords[:, ::-1]  # store as (w, h) not (h, w) for backwards compatibility
        f["feats"] = features
        f["augmented"] = np.repeat([False, True], [features.shape[0], 0])
        assert len(f["feats"]) == len(f["augmented"])
        f.attrs["extractor"] = extractor_name


class SlideTileDataset(Dataset):
    def __init__(
        self, patches: np.array, transform=None, *, repetitions: int = 1
    ) -> None:
        self.tiles = patches
        self.tiles *= repetitions
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, i) -> torch.Tensor:
        image = self.tiles[i]
        if self.transform:
            image = self.transform(image)

        return image
