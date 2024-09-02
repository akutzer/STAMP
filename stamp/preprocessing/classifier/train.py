import argparse
from pathlib import Path
from datetime import datetime
from functools import partial

import torch
from torch import nn
from fastai.vision.all import (
    Learner, DataLoader, DataLoaders, RocAuc, RocAucBinary, F1Score, AccumMetric, 
    Precision, Recall, accuracy, SaveModelCallback, CSVLogger, EarlyStoppingCallback,
    OptimWrapper
)
from transformers import AutoImageProcessor
import matplotlib.pyplot as plt

from stamp.preprocessing.classifier.data import get_augmentation, HistoCRCDataset, plot_grid
from stamp.preprocessing.classifier.model import HistoClassifier
from stamp.preprocessing.classifier.utils import validate, validate_binarized



def train(
    backbone: str,
    train_dir: str,
    valid_dir: str,
    save_dir: str = "models/",
    batch_size: int = 64,
    n_epoch: int = 20,
    binary: bool = False,
    ignore_categories: list = [],
    cores: int = 8,
    plot: bool = True
) -> nn.Module:
    run_name = get_run_name(backbone, binary)
    model_save_path = Path(save_dir) / Path(run_name)
    model_save_path.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    if use_gpu: # use tensor floats
        torch.set_float32_matmul_precision("high")

    # generate augmentation
    if "ctranspath" in backbone or "uni" in backbone:
        img_size = (224, 224)
        mean, std= [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        image_processor = AutoImageProcessor.from_pretrained(backbone)
        img_size = (image_processor.size["height"], image_processor.size["width"])
        mean, std = image_processor.image_mean, image_processor.image_std
    train_aug = get_augmentation(img_size, mean, std)
    valid_aug = get_augmentation(img_size, mean, std, validation=True)

    # initialize datasets and dataloaders
    train_ds = HistoCRCDataset(train_dir, augmentation=train_aug, reduce_to_binary=binary, ignore_categories=ignore_categories, truncate=True)
    train_ds.describe()
    valid_ds = HistoCRCDataset(valid_dir, augmentation=valid_aug, reduce_to_binary=binary, ignore_categories=ignore_categories)
    valid_ds.describe()

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cores,
        pin_memory=device.type == "cuda",
        drop_last=len(train_ds) > batch_size,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cores,
        pin_memory=device.type == "cuda",
    )

    # initialize model
    model = HistoClassifier.from_backbone(backbone, train_ds.categories, device=device)
    model.config.update({
        "categories": train_ds.categories,
        "n_categories": train_ds.n_categories,
        "inp_height": img_size[0],
        "inp_width": img_size[1],
        "mean": mean,
        "std": std,
        "backbone": backbone,
    })
    print(model.head)
    print("Trainable parameters: ",
        sum(param.numel() for param in model.parameters() if param.requires_grad),
    )

    # initialize fastai learner
    weight = train_ds.inv_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0)
    dls = DataLoaders(train_dl, valid_dl, device=device)
    learn = Learner(
        dls,
        model,
        loss_func=criterion,
        opt_func = partial(OptimWrapper, opt=torch.optim.AdamW),
        metrics=[
            AccumMetric(accuracy, flatten=False),
            Precision(average="macro"),
            Recall(average="macro"),
            F1Score(average="macro"),
            RocAucBinary(average="macro") if binary else RocAuc(average="macro"),
        ],
        path=model_save_path
    )

    cbs = [
        SaveModelCallback(monitor='valid_loss'),
        EarlyStoppingCallback(monitor="valid_loss", min_delta=0., patience=4),
        CSVLogger(),
    ]
    learn.fit_one_cycle(n_epoch=n_epoch, reset_opt=True, lr_max=1e-4, wd=1e-2, cbs=cbs)

    # Plot training and validation losses as well as learning rate schedule
    if plot:
        path_plots = model_save_path / "plots"
        path_plots.mkdir(parents=True, exist_ok=True)

        learn.recorder.plot_loss()
        plt.savefig(path_plots / 'losses_plot.png')
        plt.close()

        learn.recorder.plot_sched()
        plt.savefig(path_plots / 'lr_scheduler.png')
        plt.close()

    # save best checkpoint
    learn.model.save_pretrained(model_save_path)
    (model_save_path / "models" / "model.pth").unlink()
    (model_save_path / "models").rmdir()

    # store performance of best checkpoint on validation dataset in file
    report = validate(model, valid_dl)
    print(report)
    with open(model_save_path / "report.txt", mode="w") as f:
        f.write(report)
    if not binary:
        report = validate_binarized(model, valid_dl)
        print(report)
        with open(model_save_path / "report.txt", mode="a") as f:
            f.write(5 * "\n" + "Binarized:\n")
            f.write(report)

    return model


def get_run_name(backbone: str, is_binary: bool) -> str:
    time = datetime.now().isoformat(timespec="milliseconds")

    backbone_name = Path(backbone).stem
    run_name = f"{backbone_name}_binary={str(is_binary)}_{time}"
    return run_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tissue classifier.")
    parser.add_argument("--backbone", required=True, help="The backbone model to use for training. Must be either `ctranspath`, `uni` or a path/hubname to a huggingface model.")
    parser.add_argument("--train_dir", required=True, help="Directory containing training data. Directory structure must be in a CRC-100K like form.")
    parser.add_argument("--valid_dir", required=True, help="Directory containing validation data. Directory structure must be in a CRC-100K like form.")
    parser.add_argument("--save_dir", default="models/", help="Directory to save the trained model.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--n_epoch", type=int, default=20, help="Maximal number of epochs.")
    parser.add_argument("--binary", action="store_true", help="Merges the categories `TUM` and `STR` into a single class, while all other categories become `NORM`. This should only be used for CRC-100k.")
    parser.add_argument("--ignore_categories", nargs='*', default=[], help="List of categories to ignore.")
    parser.add_argument("--cores", type=int, default=8, help="Number of CPU cores to use during dataloading.")

    args = parser.parse_args()

    train(
        backbone=args.backbone,
        train_dir=args.train_dir,
        valid_dir=args.valid_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        n_epoch=args.n_epoch,
        binary=args.binary,
        ignore_categories=args.ignore_categories,
        cores=args.cores
    )


if __name__ == "__main__":
    main()
