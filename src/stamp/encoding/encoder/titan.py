import math
import os
from pathlib import Path
from typing import Optional

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel

import stamp
from stamp.cache import get_processing_code_hash
from stamp.encoding.encoder import Encoder
from stamp.modeling.data import CoordsInfo, get_coords




        
class EmbeddingDataset(Dataset):
    supported_extensions: set = {".h5"}
    slide_paths: list

    def __init__(self, root_dir: Path, output_dir: Path, slide_table_path: Optional[Path] = None):
        self.root_dir = root_dir
        self.output_dir = output_dir

        if slide_table_path is not None and slide_table_path.exists():
            if slide_table_path.suffix == ".csv":
                self.slide_table = pd.read_csv(slide_table_path)
            elif slide_table_path.suffix == ".xlsx":
                self.slide_table = pd.read_excel(slide_table_path)
            else:
                self.slide_table = None
        else:
            self.slide_table = None
        
        self.mode = "slide" if self.slide_table is None else "patient"
        self._find_slides()
        
    def _find_slides(self):
        if self.mode == "slide":
            self.slide_paths = self._find_slides_for_slide_emb()
        elif self.mode == "patient":
            self.slide_paths = self._find_slides_for_patient_emb()
        else:
            self.slide_paths = []
    
    def _find_slides_for_slide_emb(self) -> list:
        return [
            (slide_path.relative_to(self.root_dir), [slide_path])
            for extension in self.supported_extensions
            for slide_path in self.root_dir.glob(f"**/*{extension}")
            if not (self.output_dir / slide_path.relative_to(self.root_dir).with_suffix(".h5")).exists()
        ]

    def _find_slides_for_patient_emb(self) -> list:
        slide_paths = self._find_slides_for_slide_emb()
        patient_groups = self.slide_table.groupby("PATIENT")
        patient_paths = []

        for patient_id, group in patient_groups:
            patient_slides = []
            for _, row in group.iterrows():
                slide_filename = str(row["FILENAME"])
                for (_, [slide_path]) in slide_paths:
                    if slide_path.stem == slide_filename:
                        patient_slides.append(self.root_dir / slide_path)
                        break
            
            patient_paths.append((patient_id, patient_slides))

        return patient_paths
        
    def __len__(self):
        return len(self.slide_paths)
    
    def __getitem__(self, idx):
        id, h5_paths = self.slide_paths[idx]

        all_feats = []
        all_coords = []
        offset = torch.zeros((2,), dtype=torch.int64)

        for h5_path in h5_paths:
            with h5py.File(h5_path, "r") as f:
                feats = torch.tensor(f["feats"][:], dtype=torch.float32)
                # coords: CoordsInfo = get_coords(f)
                # hard coded for my prepro data :))))
                coords_um = torch.from_numpy(f["coords"][:])
                coords = CoordsInfo(coords_um, tile_size_um = 256.0, tile_size_px = 512)

            # Convert coordinates from microns to pixels
            coords_px = coords.coords_um / coords.mpp  # Convert to pixels
            coords_px = coords_px.to(torch.int64) # Convert to integer
            coords_px += offset
            offset[0] = coords_px[:, 0].max() + 10*coords.tile_size_px 

            all_feats.append(feats)
            all_coords.append(coords_px)

        all_feats = torch.cat(all_feats, dim=0)
        all_coords = torch.cat(all_coords, dim=0)
        
        return str(id), all_feats, all_coords


class Titan(Encoder):
    def __init__(self) -> None:
        model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
        super().__init__(model=model, identifier="mahmood-titan")
    
    def encode_slides(self, output_dir, feat_dir, device, **kwargs) -> None:
        self._encode(output_dir, feat_dir, device)

    def encode_patients(
        self, output_dir, feat_dir, slide_table_path, device, **kwargs
    ) -> None:
        """Encode patients from slide features."""
        self._encode(output_dir, feat_dir, device, slide_table_path)
    
    def _encode(self, output_dir, feat_dir, device, slide_table_path=None, **kwargs) -> None:
        output_dir = Path(output_dir)
        feat_dir = Path(feat_dir)

        # Ensure model weights and biases are on the same device as the input
        self.model.to(device).eval()

        mode = "slide" if slide_table_path is None else "patient"
        output_name = f"{self.identifier}-{mode}-{get_processing_code_hash(Path(__file__))[:8]}.h5"
        output_file = output_dir / output_name

        if output_file.exists():
            tqdm.write(f"Output file {output_file} already exists, skipping")
            return
        
        # some autocast setup
        device = torch.device(device)
        is_cuda = device.type == "cuda"
        enable_autocast = torch.amp.autocast_mode.is_autocast_available(device.type)
        torch.set_float32_matmul_precision("high")

        emb_dataset = EmbeddingDataset(feat_dir, output_dir, slide_table_path)
        emb_dataloader = DataLoader(emb_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=is_cuda)

        with h5py.File(output_file, "w") as h5_file:
            h5_file.attrs["encoder"] = self.identifier
            h5_file.attrs["stamp_version"] = stamp.__version__

            for [id], feats, coords_px in tqdm(emb_dataloader, leave=True):
                print(id, feats.shape, feats.dtype, coords_px.shape, coords_px.dtype)
                with torch.inference_mode(), torch.autocast(device.type, enabled=is_cuda):
                    slide_emb = self.model.encode_slide_from_patch_features(
                        feats.to(device), coords_px.to(device), 512
                    )
                slide_emb = slide_emb[0].to("cpu", torch.float32).numpy()
                
                h5_file.create_dataset(id, data=slide_emb)

        tqdm.write(f"Finished encoding, saved all {mode} embeddings to {output_file}")
