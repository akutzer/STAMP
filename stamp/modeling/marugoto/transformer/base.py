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


def cox_loss(event_time, event, estimate, reduce="mean"):
    # TODO: harden for the case if there are no uncensored patients
    B = estimate.shape[0]

    # determine all R patients, which are not censored
    uncensored_time = event_time[event]  # shape: (R,)

    # mask for filtering out all patient that already had the event
    mask = (uncensored_time[:, None] <= event_time)  # shape: (R, B)

    # calculate the negative log partial likelihood
    partial_like = - (estimate[event] - torch.log(torch.sum(mask * torch.exp(estimate), axis=-1)))

    if reduce == "mean":
        loss = torch.mean(partial_like)
    elif reduce == "sum":
        loss = torch.sum(partial_like)

    return loss


def cox_loss_fn(y_pred, y_true, reduce="mean"):
    event_time, event = y_true[:, 0], y_true[:, 1].type(torch.bool)
    estimate = y_pred[:, 0]
    return cox_loss(event_time, event, estimate, reduce)


def concordance_index(event_time, event, estimate):
    # event_time, event = y_true[:, 0], y_true[:, 1].type(torch.bool)
    # estimate = y_pred[:, 0]
    B = estimate.shape[0]

    comparable = (
        # both of them experienced an event (at different times)
        (event_time[:, None] < event_time) & (event[:, None] & event)
        |  # or
        # the one with a shorter observed survival time experienced an event,
        # in which case the event-free patient “outlived” the other
        ((event_time[:, None] < event_time) & (event[:, None] & (~event)))
    )  
    
    idx = torch.where(comparable)
    # extract the relative risk scores (in log space) for comparable patients
    risk1 = estimate[idx[0]]
    risk2 = estimate[idx[1]]
    # patient 1, who experienced an event earlier than patient 2, should have
    # a higher predicted relative risk score (in log space)
    ci = torch.mean((risk1 > risk2).float())

    # ci2 = concordance_index_censored(event, event_time, estimate)

    return ci
    

class SurvivalMetric(Metric):
    "Stores predictions and targets on CPU in accumulate to perform final calculations with `func`."
    def __init__(self, func, **kwargs):
        self.func = func
        self.func_kwargs = kwargs
        self._name = self.func.__name__

    def reset(self):
        "Clear all stores values"
        self.event_times, self.events, self.estimates = [], [], []

    def accumulate(self, learn):
        "Store targs and preds from `learn`, using activation function and argmax as appropriate"
        self.accum_values(learn.pred.detach(), learn.y.detach())

    def accum_values(self, preds, targs):
        "Store targs and preds"
        self.event_times.append(targs[:, 0].cpu())
        self.events.append(targs[:, 1].cpu())
        self.estimates.append(preds[:, 0].cpu())

    def __call__(self, preds, targs):
        "Calculate metric on one batch of data"
        self.reset()
        self.accum_values(preds.detach(), targs.detach())
        return self.value

    @property
    def value(self):
        "Value of the metric using accumulated preds and targs"
        if len(self.estimates) == 0: return

        self.event_times = torch.cat(self.event_times)
        self.events = torch.cat(self.events).type(torch.bool)
        self.estimates = torch.cat(self.estimates)

        return self.func(self.event_times, self.events, self.estimates, **self.func_kwargs)

    @property
    def name(self):  return self._name

    @name.setter
    def name(self, value): self._name = value


def train(
    *,
    bags: Sequence[Iterable[Path]],
    targets: Tuple[SKLearnEncoder, np.ndarray],
    add_features: Iterable[Tuple[SKLearnEncoder, Sequence[Any]]] = [],
    valid_idxs: np.ndarray,
    n_epoch: int = 32,
    patience: int = 12,
    path: Optional[Path] = None,
    batch_size: int = 128,
    cores: int = 8,
    plot: bool = True
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
    model = TransMIL(
        num_classes=len(target_enc.categories_[0]), input_dim=feature_dim,
        dim=512, depth=2, heads=8, mlp_dim=512, dropout=.0
    )
    model = L1Regularizer(model, weight_decay=1e-3)
    # TODO:
    # maybe increase mlp_dim? Not necessary 4*dim, but maybe a bit?
    # maybe add at least some dropout?
    
    # model = torch.compile(model)
    model.to(device)
    print(f"Model: {model}", end=" ")
    print(f"[Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}]")

    # # weigh inversely to class occurrences
    # counts = pd.Series(targets[~valid_idxs]).value_counts()
    # weight = counts.sum() / counts
    # weight /= weight.sum()
    # # reorder according to vocab
    # weight = torch.tensor(
    #     list(map(weight.get, target_enc.categories_[0])), dtype=torch.float32, device=device)
    loss_func = cox_loss_fn

    dls = DataLoaders(train_dl, valid_dl, device=device)

    learn = Learner(
        dls,
        model,
        loss_func=loss_func,
        opt_func = partial(OptimWrapper, opt=torch.optim.AdamW),
        metrics=[
            SurvivalMetric(cox_loss),
            SurvivalMetric(concordance_index),
        ],
        path=path,
    )#.to_bf16()

    cbs = [
        SaveModelCallback(monitor='cox_loss', fname=f'best_valid'),
        EarlyStoppingCallback(monitor='cox_loss', patience=patience),
        CSVLogger(),
        # MixedPrecision(amp_mode=AMPMode.BF16)
    ]
    learn.fit_one_cycle(n_epoch=n_epoch, reset_opt=True, lr_max=1e-4, wd=1e-2, cbs=cbs)
    
    # Plot training and validation losses as well as learning rate schedule
    if plot:
        path_plots = path / "plots"
        path_plots.mkdir(parents=True, exist_ok=True)

        learn.recorder.plot_loss()
        plt.savefig(path_plots / 'losses_plot.png')
        plt.close()

        learn.recorder.plot_sched()
        plt.savefig(path_plots / 'lr_scheduler.png')
        plt.close()

    return learn


def deploy(
    test_df: pd.DataFrame, learn: Learner, *,
    target_label: Optional[str] = None,
    cat_labels: Optional[Sequence[str]] = None, cont_labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    assert test_df.PATIENT.nunique() == len(test_df), 'duplicate patients!'
    #assert (len(add_label)
    #        == (n := len(learn.dls.train.dataset._datasets[-2]._datasets))), \
    #    f'not enough additional feature labels: expected {n}, got {len(add_label)}'
    if target_label is None: target_label = learn.target_label
    if cat_labels is None: cat_labels = learn.cat_labels
    if cont_labels is None: cont_labels = learn.cont_labels

    target_enc = learn.dls.dataset._datasets[-1].encode
    categories = target_enc.categories_[0]
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
        test_ds, batch_size=1, shuffle=False, num_workers=1)

    #removed softmax in forward, but add here to get 0-1 probabilities
    patient_preds, patient_targs = learn.get_preds(dl=test_dl, act=nn.Softmax(dim=1))

    # make into DF w/ ground truth
    patient_preds_df = pd.DataFrame.from_dict({
        'PATIENT': test_df.PATIENT.values,
        target_label: test_df[target_label].values,
        **{f'{target_label}_{cat}': patient_preds[:, i]
            for i, cat in enumerate(categories)}})

    # calculate loss
    patient_preds = patient_preds_df[[
        f'{target_label}_{cat}' for cat in categories]].values
    patient_targs = target_enc.transform(
        patient_preds_df[target_label].values.reshape(-1, 1))
    patient_preds_df['loss'] = F.cross_entropy(
        torch.tensor(patient_preds), torch.tensor(patient_targs),
        reduction='none')

    patient_preds_df['pred'] = categories[patient_preds.argmax(1)]

    # reorder dataframe and sort by loss (best predictions first)
    patient_preds_df = patient_preds_df[[
        'PATIENT',
        target_label,
        'pred',
        *(f'{target_label}_{cat}' for cat in categories),
        'loss']]
    patient_preds_df = patient_preds_df.sort_values(by='loss')

    return patient_preds_df
