import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModel, PretrainedConfig
import timm
from tqdm import tqdm

from .data import get_augmentation
from ..extractor.feature_extractors import FeatureExtractor



class HistoClassifierConfig(PretrainedConfig):
    def __init__(
        self,
        categories: Optional[List[str]] = None,
        n_classes: Optional[int] = None,
        inp_height: Optional[int] = None,
        inp_width: Optional[int] = None,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        backbone: Optional[str] = None,
        hidden_dim: Optional[int] = None,
        is_ctranspath: bool = False,
        is_uni: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.categories = categories
        self.n_classes = n_classes
        self.inp_height = inp_height
        self.inp_width = inp_width
        self.mean = mean
        self.std = std
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.is_ctranspath = is_ctranspath
        self.is_uni = is_uni

    def to_dict(self):
        config_dict = super().to_dict()
        config_dict.update(
            {
                "categories": self.categories,
                "n_classes": self.n_classes,
                "inp_height": self.inp_height,
                "inp_width": self.inp_width,
                "mean": self.mean,
                "std": self.std,
                "backbone": self.backbone,
                "hidden_dim": self.hidden_dim,
                "is_ctranspath": self.is_ctranspath,
                "is_uni": self.is_uni,
            }
        )
        return config_dict


class HistoClassifier(nn.Module):
    def __init__(self, backbone: str, hidden_dim: int, n_classes: int, is_ctranspath: bool = False, is_uni: bool = False):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(hidden_dim, n_classes))
        self.config = None
        self.device = None
        self.dtype = next(self.parameters()).dtype
        self.is_ctranspath = is_ctranspath
        self.is_uni = is_uni

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if not (self.is_ctranspath or self.is_uni):
            out = out.pooler_output
        out = self.head(out)
        return out

    def predict(self, x) -> np.ndarray:
        cache_mode = self.training
        self.train(False)
        if unsqueeze := (x.dim() == 3):
            x = x.unsqueeze(0)

        with torch.inference_mode():
            pred = self.head(self.backbone(x) if (self.is_ctranspath or self.is_uni) else self.backbone(x).pooler_output)
            probs = torch.softmax(pred, dim=-1).cpu()
            if unsqueeze:
                probs = probs[0]
        probs, indices = torch.sort(probs, descending=True)
        probs, indices = probs.numpy(), indices.numpy()
        labels = np.take(self.config.categories, indices)

        out = np.empty(
            probs.shape,
            dtype=np.dtype(
                [
                    ("label", labels.dtype),
                    ("id", indices.dtype),
                    ("probability", probs.dtype),
                ]
            ),
        )
        out["probability"] = probs
        out["id"] = indices
        out["label"] = labels

        self.train(cache_mode)
        return out

    def predict_tiles(self, tiles: np.ndarray) -> np.ndarray:
        cache_mode = self.training
        self.train(False)
        if not hasattr(self, "mean"):
            self.mean = torch.tensor(self.config.mean, dtype=self.dtype, device=self.device)
        if not hasattr(self, "std"):
            self.std = torch.tensor(self.config.std, dtype=self.dtype, device=self.device)

        tiles = torch.from_numpy(tiles).to(dtype=self.dtype, device=self.device)

        tiles = tiles / 255.0
        tiles -= self.mean
        tiles /= self.std

        if unsqueeze := (tiles.dim() == 3):
            tiles = tiles.unsqueeze(0)
            
        tiles = tiles.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        with torch.inference_mode(), torch.autocast(self.device.type):
            pred =  self.backbone(tiles)
            if not (self.is_ctranspath or self.is_uni):
                pred = pred.pooler_output
            pred = self.head(pred)
            probs = torch.softmax(pred, dim=-1).half().cpu()

        if unsqueeze:
            probs = probs[0]
        
        probs, indices = torch.sort(probs, dim=-1, descending=True)
        probs, indices = probs.numpy(), indices.numpy()
        labels = np.take(self.config.categories, indices)

        out = np.empty(
            probs.shape,
            dtype=np.dtype(
                [
                    ("label", labels.dtype),
                    ("id", indices.dtype),
                    ("probability", probs.dtype),
                ]
            ),
        )
        out["probability"] = probs
        out["id"] = indices
        out["label"] = labels

        self.train(cache_mode)
        return out

    def predict_patches(
            self, patches: np.ndarray, cores: int = 8, batch_size: int = 64
    ) -> np.ndarray:
        img_size = (self.config.inp_height, self.config.inp_width)
        mean, std = self.config.mean, self.config.std
        transform = get_augmentation(img_size, mean, std, validation=True)

        dataset = SlideTileDataset(patches, transform)
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
            for patches_batch in dataloader:
                patches_batch = patches_batch.to(dtype=self.dtype, device=self.device)
                features_batch = self.predict(patches_batch)
                features.append(features_batch)

        features = np.concatenate(features, axis=0)
        return features

    @classmethod
    def from_backbone(cls, backbone_name: str, categories: List[str], device: str = "cpu"):
        is_ctranspath = "ctranspath" in backbone_name
        is_uni = "uni" in backbone_name
        if is_ctranspath:
            backbone = FeatureExtractor.init_ctranspath(checkpoint_path=backbone_name, device=device).model
            config = HistoClassifierConfig(
                categories = categories,
                n_classes = len(categories),
                inp_height = 224,
                inp_width = 224,
                mean= [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
                backbone = "ctranspath",
                hidden_dim = 768,
                is_ctranspath=True
            )
            config.update({
                "id2label": {i: cat for i, cat in enumerate(categories)},
                "label2id": {cat: i for i, cat in enumerate(categories)},
            })
        elif is_uni:
            backbone = FeatureExtractor.init_uni(asset_dir=backbone_name, device=device).model

            # freeze all except the last k layers
            k = 2
            for param in backbone.parameters():
                param.requires_grad = False
            for param in backbone.blocks[-k:].parameters():
                param.requires_grad = True
            for param in backbone.norm.parameters():
                param.requires_grad = True

            config = HistoClassifierConfig(
                categories = categories,
                n_classes = len(categories),
                inp_height = 224,
                inp_width = 224,
                mean= [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
                backbone = "uni",
                hidden_dim = 1024,
                is_uni=True
            )
            config.update({
                "id2label": {i: cat for i, cat in enumerate(categories)},
                "label2id": {cat: i for i, cat in enumerate(categories)},
            })
        else:
            config = AutoConfig.from_pretrained(backbone_name)
            if hasattr(config, "hidden_dim"):
                hidden_dim = config.hidden_dim
            elif hasattr(config, "hidden_size"):
                hidden_dim = config.hidden_size
            else:
                raise AttributeError
            config = HistoClassifierConfig(**config.to_dict())
            config.update(
                {
                    "hidden_dim": hidden_dim,
                    "n_classes": len(categories),
                    "id2label": {i: cat for i, cat in enumerate(categories)},
                    "label2id": {cat: i for i, cat in enumerate(categories)},
                }
            )

            backbone = AutoModel.from_pretrained(backbone_name)
        model = cls(backbone, config.hidden_dim, config.n_classes, is_ctranspath=is_ctranspath, is_uni=is_uni)
        model.config = config
        model.to(device)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, device: str = "cpu"):
        model_dir = Path(pretrained_model_path)
        assert model_dir.is_dir()

        config = HistoClassifierConfig.from_pretrained(model_dir)
        if config.is_ctranspath:
            backbone = create_ctranspath(device=device)
        if config.is_uni:
            backbone = create_uni(device=device)
        else:
            backbone = AutoModel.from_pretrained(config.backbone)
        model = cls(backbone, config.hidden_dim, config.n_classes, is_ctranspath=config.is_ctranspath, is_uni=config.is_uni)

        state_dict = torch.load(model_dir / "model.pt", map_location=torch.device("cpu"), weights_only=True)
        # state_dict_renamed = state_dict.copy()
        # for key in state_dict.keys():
        #     if "._orig_mod" in key:
        #         new_key = key.replace("._orig_mod", "")
        #         state_dict_renamed[new_key] = state_dict_renamed.pop(key)
        model.load_state_dict(state_dict)

        model.config = config
        model.to(device)
        return model

    def save_pretrained(self, path: str):
        path = Path(path)
        self.config.save_pretrained(path)
        torch.save(self.state_dict(), path / "model.pt")

    def to(self, device: str, dtype=None):
        self.device = torch.device(device)
        return super().to(self.device, dtype)


class SlideTileDataset(Dataset):
    def __init__(self, patches: np.array, transform=None) -> None:
        self.tiles = patches
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, i) -> torch.Tensor:
        image = self.tiles[i]
        if self.transform:
            image = self.transform(image)

        return image


def create_ctranspath(checkpoint_path: Optional[str] = None, device: str = "cpu"):
    # initializing the model
    model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
    model.head = nn.Identity()

    # loading the checkpoint weights and updating the weights
    if checkpoint_path is not None:  
        digest = get_digest(checkpoint_path)
        assert digest == "7c998680060c8743551a412583fac689db43cec07053b72dfec6dcd810113539"
        weights = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(weights["model"], strict=True)

        # initializing the model
        model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
        model.head = nn.Identity()
    model.to(device)
    return model


def create_uni(checkpoint_path: Optional[str] = None, device: str = "cpu"):
    uni_kwargs = {
            'model_name': 'vit_large_patch16_224',
            'img_size': 224, 
            'patch_size': 16, 
            'init_values': 1e-5, 
            'num_classes': 0, 
            'dynamic_img_size': True
        }
    model = timm.create_model(**uni_kwargs)
    model.to(device)
    return model
