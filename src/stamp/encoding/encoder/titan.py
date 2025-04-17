import math
import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from tqdm import tqdm

import stamp
from stamp.cache import get_processing_code_hash
from stamp.encoding.encoder import Encoder
from stamp.modeling.data import CoordsInfo, get_coords



class Mode(Enum):
    SLIDE = "slide"
    PATIENT = "patient"


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Attributes:
        root_dir: Directory containing HDF5 files of the slide features.
        output_dir: Directory to write slide/patient encoding HDF5 file to.
        mode: Operation mode; determines whether to encode per-slide or per-patient.
        slide_table_path: Path to slide-to-patient mapping table. Required when mode is PATIENT.
        margin: Pixel margin between concatenated slide coordinates.
    """
    root_dir: Path
    output_dir: Path
    mode: Mode = Mode.SLIDE
    slide_table_path: Optional[Path] = None
    margin: int = field(default=10, metadata={"unit": "px"})

    def __post_init__(self):
        # enforce that slide_table_path is present in PATIENT mode
        if self.mode == Mode.PATIENT:
            if self.slide_table_path is None:
                raise ValueError(
                    "EmbeddingConfigError: 'slide_table_path' must be provided when mode is PATIENT"
                )
            if not self.slide_table_path.exists():
                raise FileNotFoundError(
                    f"Slide table not found: {self.slide_table_path}"
                )
        # ensure directories exist or create them
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)


class EmbeddingDataset(Dataset):
    supported_extensions: set = {".h5"}

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.root_dir = config.root_dir
        self.output_dir = config.output_dir
        self.margin = config.margin
        self.mode = config.mode

        # load slide table in patient mode
        self.slide_table = None
        if self.mode == Mode.PATIENT:
            self.slide_table = self._load_slide_table()

        self.slide_paths = self._find_slides()
    
    def _load_slide_table(self) -> pd.DataFrame:
        ext = self.config.slide_table_path.suffix.lower()
        if ext == ".csv":
            slide_table = pd.read_csv(self.config.slide_table_path)
        elif ext in {".xls", ".xlsx"}:
            slide_table = pd.read_excel(self.config.slide_table_path)
        else:
            raise ValueError(f"Unsupported slide table format: {ext}")
        return slide_table

    def _find_slides(self) -> List[Tuple[str, List[Path]]]:
        if self.mode == Mode.SLIDE:
            return self._find_slides_for_slide_emb()
        else:
            return self._find_slides_for_patient_emb()

    def _find_slides_for_slide_emb(self) -> List[Tuple[str, List[Path]]]:
        return [
            (str(slide_path.relative_to(self.root_dir)), [slide_path])
            for ext in self.supported_extensions
            for slide_path in self.root_dir.glob(f"**/*{ext}")
        ]

    def _find_slides_for_patient_emb(self) -> List[Tuple[str, List[Path]]]:
        slide_list = self._find_slides_for_slide_emb()
        patient_paths: List[Tuple[str, List[Path]]] = []

        for patient_id, group in self.slide_table.groupby("PATIENT"):
            patient_slides: List[Path] = []
            for _, row in group.iterrows():
                filename = str(row["FILENAME"])
                for rel, [slide_path] in slide_list:
                    if slide_path.stem == filename:
                        patient_slides.append(slide_path)
                        break
                
            if patient_slides:
                patient_paths.append((str(patient_id), patient_slides))

        return patient_paths
    
    def __len__(self) -> int:
        return len(self.slide_paths)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        identifier, paths = self.slide_paths[idx]
        all_feats: List[torch.Tensor] = []
        all_coords: List[torch.Tensor] = []
        offset = torch.zeros(2, dtype=torch.int64)

        for p in paths:
            with h5py.File(p, "r") as f:
                feats = torch.tensor(f["feats"][:], dtype=torch.float32)
                # coords: CoordsInfo = get_coords(f)
                # hard coded for my prepro data :))))
                coords_um = torch.from_numpy(f["coords"][:])
            coords = CoordsInfo(coords_um, tile_size_um=256.0, tile_size_px=512)

            # convert to pixel coordinates and apply offset
            coords_px = (coords.coords_um / coords.mpp).to(torch.int64)
            coords_px += offset

            # update offset for next slide
            offset[0] = coords_px[:, 0].max() + self.margin * coords.tile_size_px

            all_feats.append(feats)
            all_coords.append(coords_px)

        feats_concat = torch.cat(all_feats, dim=0)
        coords_concat = torch.cat(all_coords, dim=0)
        return identifier, feats_concat, coords_concat


class Titan(Encoder):
    def __init__(self) -> None:
        model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
        super().__init__(model=model, identifier="mahmood-titan")
    
    def encode_slides(self, output_dir, feat_dir, device, **kwargs) -> None:
        """Encode slide from slide features."""
        config = EmbeddingConfig(Path(feat_dir), Path(output_dir), Mode.SLIDE) 
        self._encode(config, device, **kwargs)

    def encode_patients(self, output_dir, feat_dir, slide_table_path, device, **kwargs) -> None:
        """Encode patient from all their slide features."""
        config = EmbeddingConfig(
            Path(feat_dir), Path(output_dir), Mode.PATIENT, slide_table_path
        ) 
        self._encode(config, device, **kwargs)
    
    def _encode(self, config, device, **kwargs) -> None:
        # Ensure model weights and biases are on the same device as the input
        self.model.to(device).eval()

        output_name = f"{self.identifier}-{config.mode.name}" \
                      f"-{get_processing_code_hash(Path(__file__))[:8]}.h5"
        output_file = config.output_dir / output_name

        if output_file.exists():
            tqdm.write(f"Output file {output_file} already exists, skipping...")
            return
        
        # some autocast setup
        device = torch.device(device)
        is_cuda = device.type == "cuda"
        enable_autocast = torch.amp.autocast_mode.is_autocast_available(device.type)
        torch.set_float32_matmul_precision("high")

        emb_dataset = EmbeddingDataset(config)
        emb_dataloader = DataLoader(
            emb_dataset, batch_size=1, shuffle=True, num_workers=32, pin_memory=is_cuda
        )

        pbar = tqdm(emb_dataloader, leave=True)
        with h5py.File(output_file, "w") as h5_file:
            h5_file.attrs["encoder"] = self.identifier
            h5_file.attrs["stamp_version"] = stamp.__version__

            for [id], feats, coords_px in pbar:
                pbar.set_description_str(f"Encoding {id}")
                with torch.inference_mode(), torch.autocast(device.type, enabled=is_cuda):
                    slide_emb = self.model.encode_slide_from_patch_features(
                        feats.to(device), coords_px.to(device), 512
                    )
                slide_emb = slide_emb[0].to("cpu", torch.float32).numpy()
                h5_file.create_dataset(id, data=slide_emb)

        tqdm.write(f"Finished encoding, saved all {config.mode.name} embeddings to {output_file}")
