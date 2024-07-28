from typing import Any, Iterable, Optional, Sequence, Tuple, TypeVar
from pathlib import Path
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from fastai.vision.all import (
    Learner, DataLoader, DataLoaders, RocAuc,
    SaveModelCallback, CSVLogger, EarlyStoppingCallback,
    MixedPrecision, AMPMode, OptimWrapper, Metric
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_censored


from .data import make_dataset, SKLearnEncoder
from .TransMIL import TransMIL
from .AttnMIL import AttnMIL

from .metric import *


__all__ = ['train', 'deploy']


T = TypeVar('T')


class L1Regularizer(nn.Module):
    """https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch"""
    def __init__(self, module, weight_decay):
        super().__init__()
        self.module = module
        self.weight_decay = weight_decay

        # Backward hook is registered on the specified module
        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    # Not dependent on backprop incoming values, placeholder
    def _weight_decay_hook(self, *_):
        for param in self.module.parameters():
            # If there is no gradient or it was zeroed out
            if param.grad is None or torch.all(param.grad == 0.):
                # Apply regularization on it
                param.grad = self.regularize(param)

    def regularize(self, parameter):
        # L1 regularization formula
        return self.weight_decay * torch.sign(parameter.data)

    def forward(self, *args, **kwargs):
        # Simply forward and args and kwargs to module
        return self.module(*args, **kwargs)


def train(
    *,
    bags: Sequence[Iterable[Path]],
    targets: Tuple[SKLearnEncoder, np.ndarray],
    add_features: Iterable[Tuple[SKLearnEncoder, Sequence[Any]]] = [],
    valid_idxs: np.ndarray,
    n_epoch: int = 100,
    patience: int = 10,
    path: Optional[Path] = None,
    batch_size: int = 100,
    cores: int = 8,
    plot: bool = True,
    method: str = "cox",
    aggregation: str = "trans_mil"
) -> Learner:
    """Train a MLP on image features.

    Args:
        bags:  H5s containing bags of tiles.
        targets:  An (encoder, targets) pair.
        add_features:  An (encoder, targets) pair for each additional input.
        valid_idxs:  Indices of the datasets to use for validation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        # allow for usage of TensorFloat32 as internal dtype for matmul on modern NVIDIA GPUs
        torch.set_float32_matmul_precision("high")

    target_enc, targs = targets
    train_ds = make_dataset(
        bags=bags[~valid_idxs],
        targets=(target_enc, targs[~valid_idxs]),
        add_features=[
            (enc, vals[~valid_idxs])
            for enc, vals in add_features],
        bag_size=1024)

    valid_ds = make_dataset(
        bags=bags[valid_idxs],
        targets=(target_enc, targs[valid_idxs]),
        add_features=[
            (enc, vals[valid_idxs])
            for enc, vals in add_features],
        bag_size=None)


    event_time, event = targs[valid_idxs][:, 0], targs[valid_idxs][:, 1].astype(np.bool_)
    comparable = (event_time[:, None] < event_time) & (event[:, None])
    c = comparable.sum()
    n, k = targs.shape[0], event.sum()
    print(f"Fraction of comparable pairs given maximal number of comparable pairs when {k} patients had an events:")
    print(f"{c / (- .5*k * (k - 2*n + 1)):.4f}")

    print(f"Fraction of comparable pairs given maximal number of comparable pairs when all patients are comparable pairs:")
    print(f"{c / (n * (n - 1)):.4f}")

    # build dataloaders
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=cores,
        drop_last=len(train_ds) > batch_size,
        device=device, pin_memory=device.type == "cuda"
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=1, shuffle=False, num_workers=cores,
        device=device, pin_memory=device.type == "cuda"
    )
    batch = train_dl.one_batch()
    feature_dim = batch[0].shape[-1]

    # for binary classification num_classes=2
    if aggregation == "trans_mil":
        model = TransMIL(
            num_classes=len(target_enc.categories_), input_dim=feature_dim,
            dim=512, depth=2, heads=8, mlp_dim=512, dropout=.1
        )
    elif aggregation == "attn_mil":
        model = AttnMIL(
            n_out=len(target_enc.categories_), n_feats=feature_dim
        )
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}. Must be either `trans_mil` or `attn_mil`")
    # model = L1Regularizer(model, weight_decay=1e-3)
    
    model.to(device)
    print(f"Model: {model}", end=" ")
    print(f"[Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}]")

    dls = DataLoaders(train_dl, valid_dl, device=device)

    if method == "cox":
        criterion = CoxLossBreslow()
        metrics = [
            SurvivalMetric(criterion.cox_loss_breslow),
            SurvivalMetric(ConcordanceIndex("cox")),            
        ]
        monitor = criterion.cox_loss_breslow.__name__
    elif method == "logistic-hazard":
        criterion = LogisticHazardLoss()
        metrics = [
            SurvivalMetric(ConcordanceIndex("mrl", intervals=target_enc.categories_)),
            SurvivalMetric(ConcordanceIndex("isurv", intervals=target_enc.categories_))
        ]
        monitor = "valid_loss"
    elif method == "mrl":
        criterion = MRLLoss(target_enc.categories_)
        metrics = [
            SurvivalMetric(criterion.negative_concordance_index_mrl),
            SurvivalMetric(ConcordanceIndex("mrl", intervals=target_enc.categories_)),
            SurvivalMetric(ConcordanceIndex("isurv", intervals=target_enc.categories_))
        ]
        monitor = criterion.negative_concordance_index_mrl.__name__
    else:
        raise ValueError
    
    learn = Learner(
        dls,
        model,
        loss_func=criterion,
        opt_func = partial(OptimWrapper, opt=torch.optim.AdamW),
        metrics=metrics,
        path=path,
    )

    cbs = [
        SaveModelCallback(monitor=monitor, fname=f'best_valid'),
        EarlyStoppingCallback(monitor=monitor, patience=patience),
        CSVLogger()
    ]
    # learn.fit_one_cycle(n_epoch=n_epoch, reset_opt=True, lr_max=1e-4, wd=1e-3, cbs=cbs, pct_start=.05)
    learn.fit(n_epoch=n_epoch, reset_opt=True, lr=5e-5, wd=1e-3, cbs=cbs)
    
    # Plot training and validation losses as well as learning rate schedule
    if plot:
        path_plots = path / "plots"
        path_plots.mkdir(parents=True, exist_ok=True)

        learn.recorder.plot_loss()
        plt.savefig(path_plots / 'losses_plot.png')
        plt.close()

    return learn


def deploy(
    test_df: pd.DataFrame, learn: Learner, *,
    target_label: Optional[str] = None,
    cat_labels: Optional[Sequence[str]] = None, cont_labels: Optional[Sequence[str]] = None,
    device: torch.device = torch.device('cpu')
) -> pd.DataFrame:
    assert test_df.PATIENT.nunique() == len(test_df), 'duplicate patients!'

    if target_label is None: target_label = learn.target_label
    if cat_labels is None: cat_labels = learn.cat_labels
    if cont_labels is None: cont_labels = learn.cont_labels

    method = learn.method
    target_enc = learn.dls.dataset._datasets[-1].encode
    categories = target_enc.categories_
    add_features = []
    if cat_labels:
        cat_enc = learn.dls.dataset._datasets[-2]._datasets[0].encode
        add_features.append((cat_enc, test_df[cat_labels].values))
    if cont_labels:
        cont_enc = learn.dls.dataset._datasets[-2]._datasets[1].encode
        add_features.append((cont_enc, test_df[cont_labels].values))

    test_ds = make_dataset(
        bags=test_df.slide_path.values,
        targets=(target_enc, test_df[target_label].values),
        add_features=add_features,
        bag_size=None)

    test_dl = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=8,
        device=device, pin_memory=device.type == "cuda")

    #removed softmax in forward, but add here to get 0-1 probabilities
    patient_preds, patient_targs = learn.get_preds(dl=test_dl, act=lambda x: x)

    # make into DF w/ ground truth
    patient_preds_df = pd.DataFrame.from_dict({
        'PATIENT': test_df.PATIENT.values,
        **{f'{label}': test_df[label].values
            for label in target_label},
    })

    if method == "cox":
        patient_preds_df["relative_log_risk"] = patient_preds.flatten()
        patient_preds_df["relative_risk"] = np.exp(patient_preds).flatten()
    elif method == "logistic-hazard":
        intervals = target_enc.categories_
        hazard = torch.sigmoid(patient_preds).cpu()
        assert patient_preds.shape[-1] == len(intervals)
        for i, t in enumerate(intervals):
            patient_preds_df[f"{t}"] = hazard[:, i]
        patient_preds_df["mean_residual_lifetime"] = calc_mrl(patient_preds, intervals)
        patient_preds_df["integrated_survival"] = calc_isurv(patient_preds, intervals)
        
    patient_preds_df = patient_preds_df.sort_values(by='PATIENT')
    return patient_preds_df
