import hashlib
import json
import os
from pathlib import Path
from typing import Tuple, Union, List
import types 

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from torchvision.transforms import ToTensor, Normalize
from transformers import AutoImageProcessor, Dinov2Model

import uni
from conch.open_clip_custom import create_model_from_pretrained
from stamp.preprocessing.extractor.swin_transformer import swin_tiny_patch4_window7_224, ConvStem


__version__ = "1.2.0_19-08-2024"


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
        if device.type == "cuda":
            # allow for usage of TensorFloat32 as internal dtype for matmul on modern NVIDIA GPUs
            torch.set_float32_matmul_precision("high")
        self.dtype = next(self.model.parameters()).dtype

        self.mean, self.std = self._extract_mean_std(transform, self.dtype, self.device)

        # with torch.no_grad():
            # model = torch.jit.trace(model, torch.randn(1, 3, 224, 224, dtype=self.dtype, device=self.device))
        # model = torch.compile(model)

    @classmethod
    def init_ctranspath(cls, checkpoint_path: str, device: str) -> "FeatureExtractor":
        # loading the checkpoint weights
        digest = get_digest(checkpoint_path)
        assert digest == "7c998680060c8743551a412583fac689db43cec07053b72dfec6dcd810113539"

        model_name = f"xiyuewang-ctranspath-{digest[:8]}"
        ctranspath_weights = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)

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

    @classmethod
    def init_dinov2(cls, device: str, **kwargs) -> "FeatureExtractor":
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
        model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/dinov2/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/model.safetensors")
        digest = get_digest(model_path)
        model_name = f"facebook-dinov2-{digest[:8]}"

        # initializing the model and updating the weights

        class Dinov2(Dinov2Model):
            def forward(self, *args, **kwargs):
                out = super().forward(*args, **kwargs)
                return out.pooler_output
        
        model = Dinov2.from_pretrained(model_path.parent)

        processor = AutoImageProcessor.from_pretrained(model_path.parent)
        mean, std = processor.image_mean, processor.image_std
        size = (processor.crop_size["height"], processor.crop_size["width"])

        transform = transforms.Compose([
            transforms.ToImage(),  # convert to tensor, only needed for PIL images
            transforms.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
            transforms.Resize(size=size, antialias=True),
            transforms.ToDtype(torch.float32, scale=True),  # normalize expects float input
            transforms.Normalize(mean=mean, std=std),
        ])

        extractor = cls(model, model_name, transform, device)
        print("DINOv2 model successfully initialized...\n")

        return extractor

    @classmethod
    def init_conch(cls, device: str, **kwargs) -> "FeatureExtractor":
        """Extracts features from slide tiles. 
        Requirements: 
            Permission from authors via huggingface: https://huggingface.co/MahmoodLab/CONCH
            Huggingface account with valid login token
        On first model initialization, you will be prompted to enter your login token. The token is
        then stored in ./home/<user>/.cache/huggingface/token. Subsequent inits do not require you to re-enter the token. 

        Args:
            device: "cuda" or "cpu"
        """
        
        # loading the checkpoint weights
        model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/conch/models--MahmoodLab--conch/snapshots/f9ca9f877171a28ade80228fb195ac5d79003357/pytorch_model.bin")
        digest = get_digest(model_path)
        model_name = f"mahmood-conch-{digest[:8]}"

        # initializing the model and updating the weights
        model, transform = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=str(model_path))
        # CONCH implementations still uses transformsV1, which expects
        # either a PILImage or Tensor as input, not a numpy array like in the current code,
        # thus we have to move the ToTensor transform to the beginning 
        # Additionally, this filters out the "_convert_to_rgb" function
        transform.transforms = list(filter(lambda x: not isinstance(x, (ToTensor, types.FunctionType)), transform.transforms))
        transform.transforms.insert(0, ToTensor())
        print(model)
        print(transform)
        exit(0)

        extractor = cls(model, model_name, transform, device)
        print("UNI model successfully initialized...\n")

        return extractor

    def extract(
        self, tiles: Union[np.ndarray, List[np.ndarray]], cores: int = 2, batch_size: int = 64
    ) -> np.ndarray:
        """Extracts features from slide tiles.

        Args:
            tiles:  Array of shape (n_tiles, tile_h, tile_w, 3)
            cores:  Number of cores for dataloader
        """
        dataset = SlideTileDataset(tiles, self.transform)
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
            for tiles_batch in dataloader:
                tiles_batch = tiles_batch.to(dtype=self.dtype, device=self.device)
                features_batch = self.model(tiles_batch).half()
                features.append(features_batch)

        features = torch.concat(features, dim=0).cpu().numpy()
        return features
    
    def single_extract(self, tiles: np.ndarray) -> np.ndarray:
        """Extracts features from slide tiles.

        Args:
            tiles:  Array of shape (n_tiles, tile_h, tile_w, 3)
            cores:  Number of cores for dataloader
        """
        tiles = torch.from_numpy(tiles).to(dtype=self.dtype, device=self.device)

        tiles = tiles / 255.0
        tiles -= self.mean
        tiles /= self.std

        tiles = tiles.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        with torch.inference_mode():
            features = self.model(tiles).half().cpu().numpy()
        
        return features

    @staticmethod
    def _extract_mean_std(transform, dtype=None, device=None):
        for tf in transform.transforms:
            if isinstance(tf, (transforms.Normalize, Normalize)):
                mean = torch.tensor(tf.mean, dtype=dtype, device=device)
                std = torch.tensor(tf.std, dtype=dtype, device=device)
                break
        else:
            print("No `Normalize` transformation found!")
            mean = torch.tensor([0., 0., 0.], dtype=dtype, device=device)
            std = torch.tensor([1., 1., 1.], dtype=dtype, device=device)

        return mean, std


def store_metadata(
    outdir: Path,
    extractor_name: str,
    tile_size: Tuple[int, int],
    target_microns: int,
    normalized: bool,
):
    with open(outdir / "info.json", "w") as f:
        json.dump({
            "extractor": extractor_name,
            "augmented_repetitions": 0,
            "tiles_normalized": normalized,
            "microns": target_microns,
            "tile_size": tile_size,
        }, f)


def store_features(
    outdir: Path, features: np.ndarray, tiles_coords: np.ndarray, extractor_name: str
):
    with h5py.File(f"{outdir}.h5", "w") as f:
        f["coords"] = tiles_coords[:, ::-1]  # store as (w, h) not (h, w) for backwards compatibility
        f["feats"] = features
        f["augmented"] = np.repeat([False, True], [features.shape[0], 0])
        assert len(f["feats"]) == len(f["augmented"])
        f.attrs["extractor"] = extractor_name


class SlideTileDataset(Dataset):
    def __init__(
        self, tiles: Union[np.ndarray, List[np.ndarray]], transform=None, *, repetitions: int = 1
    ) -> None:
        self.tiles = tiles
        if isinstance(self.tiles, list):
            self._first_indices = np.cumsum([0] + [arr.shape[0] for arr in self.tiles[:-1]])
        # self.tiles *= repetitions
        self.transform = transform

    def __len__(self) -> int:
        if isinstance(self.tiles, list):
            return sum(map(lambda x: x.shape[0], self.tiles))
        return len(self.tiles)

    def __getitem__(self, i) -> torch.Tensor:
        if isinstance(self.tiles, np.ndarray):
            image = self.tiles[i]
        else:
            super_idx = np.argmax(self._first_indices > i) - 1
            sub_idx = i - self._first_indices[super_idx] 
            image = self.tiles[super_idx][sub_idx]
        if self.transform:
            image = self.transform(image)

        return image

