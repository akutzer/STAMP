import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import trapezoid, simpson
from fastai.vision.all import Metric



class CoxLossBreslow(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, estimate, target):
        event_time, event = target[:, 0], target[:, 1].bool()
        loss = self.cox_loss_breslow(event_time, event, estimate)
        return loss
    
    def cox_loss_breslow(self, event_time, event, estimate):
        if estimate.ndim > 1:
            estimate = estimate.squeeze(dim=-1)

        # determine all R patients, which had an event occur
        uncensored_time = event_time[event]  # shape: (R,)

        # mask for finding all patient that haven't had the event yet
        mask = (uncensored_time[:, None] <= event_time)  # shape: (R, B)

        # calculate the negative log partial likelihood
        partial_like = - (estimate[event] - torch.log(torch.sum(mask * torch.exp(estimate), axis=-1)))

        # use sum in case there a not patients with an event, 
        # reulting in the loss being 0 and not NaN (which happens when using torch.mean)
        if self.reduction == "sum" or partial_like.numel() == 0:
            partial_like = torch.sum(partial_like)
        elif self.reduction == "mean":
            partial_like = torch.mean(partial_like)

        return partial_like


class LogisticPDFLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, estimate, target):
        event_time, event = target[:, 0], target[:, 1].bool()
        loss = self.logistic_pdf_loss(event_time, event, estimate)
        return loss
    
    def logistic_pdf_loss(self, event_time, event, estimate):
        B = estimate.shape[0]
        event = event.float()
        event_time = event_time.int()

        pdf = torch.softmax(estimate, dim=-1)
        cdf = torch.cumsum(pdf, dim=-1)

        likelihood = - (
            event * torch.log(pdf[torch.arange(B), event_time])
            + (1 - event) * torch.log(1 - cdf[torch.arange(B), event_time])
        )

        if self.reduction == "mean":
            loss = torch.mean(likelihood)
        elif self.reduction == "sum":
            loss = torch.sum(likelihood)

        return loss


class LogisticHazardLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, estimate, target):
        event_time, event = target[:, 0], target[:, 1].bool()
        loss = self.logistic_hazard_loss(event_time, event, estimate)
        return loss
    
    def logistic_hazard_loss(self, event_time, event, estimate):
        B, L = estimate.shape
        event = event.float()
        event_time = event_time.int()

        hazard = torch.sigmoid(estimate)
        log_survival = torch.cumsum(torch.log(1 - F.pad(hazard, (1, 0))), dim=-1)

        likelihood = - (
            event * torch.log(hazard[torch.arange(B), event_time])
            + (1 - event) * torch.log(1 - hazard[torch.arange(B), event_time])
            + log_survival[torch.arange(B), event_time]
        )

        # alternative naive implementation, which highlights the similarity
        # to binary cross-entropy
        # mask[i, j] = True if patient i had an event and it happend at time t_j
        # mask = event[:, None].bool() & (event_time[:, None] == torch.arange(L, device=estimate.device))
        # likelihood = mask * torch.log(hazard) + (~mask) * torch.log(1 - hazard)
        # likelihood = -torch.cumsum(likelihood, dim=-1)[torch.arange(B), event_time]

        if self.reduction == "mean":
            likelihood = torch.mean(likelihood)
        elif self.reduction == "sum":
            likelihood = torch.sum(likelihood)

        return likelihood


class MRLLoss(nn.Module):
    def __init__(self, intervals, reduction="mean"):
        super().__init__()
        self.intervals = intervals
        self.reduction = reduction
    
    def forward(self, estimate, target):
        event_time, event = target[:, 0], target[:, 1].bool()
        loss = self.negative_concordance_index_mrl(event_time, event, estimate)
        return loss
    
    def negative_concordance_index_mrl(self, event_time, event, estimate):
        hazard = torch.sigmoid(estimate)
        # print(hazard.requires_grad)
        survival = F.pad(torch.cumprod(1 - hazard, dim=-1), (1, 0), value=1)
        # print(survival.dtype)
        # print(survival.requires_grad)  
        pdf = hazard * survival[..., :-1]
        # print(pdf.requires_grad)  

        # calculate mean residual lifetime, aka expected lifetime
        intervals = torch.tensor(self.intervals, device=pdf.device, dtype=pdf.dtype)
        mrl = torch.sum(intervals * pdf, axis=-1)
        # print(mrl.requires_grad, mrl.dtype) 

        # the patient with a shorter observed survival time experienced an event,
        # and was “outlived” by the second patient (with might not even had an event
        comparable = (event_time[:, None] < event_time) & (event[:, None])
        
        idx = torch.where(comparable)
        mrl1 = mrl[idx[0]]
        mrl2 = mrl[idx[1]]
        # print(mrl1.requires_grad, mrl1.dtype) 
        # print(mrl2.requires_grad, mrl2.dtype) 
        # patient 1, who experienced an event earlier than patient 2, should have
        # a smaller mean residual lifetime
        # ci = torch.mean((mrl1 < mrl2).float())
        diff = torch.mean(mrl1 - mrl2)
        # print((mrl1 - mrl2).shape)
        # print(diff)
        return diff


class ConcordanceIndex():
    def __init__(self, mode, intervals=None):
        assert mode in ["cox", "mrl", "isurv"]
        self.mode = mode
        self.intervals = intervals        
        self.__name__ = f"concordance_index_{mode}"
    
    def __call__(self, event_time, event, estimate):
        if self.mode == "cox":
            # use the relative_risk scores (in log space) for CI
            score = estimate
        elif self.mode == "mrl":
            # use mean residual lifetime for CI
            score = calc_mrl(estimate, self.intervals)
        elif self.mode == "isurv":
            # use integrated survival for CI
            score = calc_isurv(estimate, self.intervals)
        
        ci = self.calc_ci(event_time, event, score)
        return ci
        
    def calc_ci(self, event_time, event, score):
        # the patient with a shorter observed survival time experienced an event,
        # and was “outlived” by the second patient (with might not even had an event
        comparable = (event_time[:, None] < event_time) & (event[:, None])
        
        idx = torch.where(comparable)
        score1 = score[idx[0]]
        score2 = score[idx[1]]
    
        if self.mode in ["cox"]:
            # patient 1, who experienced an event earlier than patient 2, should have
            # a higher predicted relative_risk score (in log space)
            ci = torch.mean((score1 > score2).float())
        elif self.mode in ["mrl", "isurv"]:
            # patient 1, who experienced an event earlier than patient 2, should have
            # a smaller mean residual lifetime or integrated survival
            ci = torch.mean((score1 < score2).float())
        else:
            raise ValueError
            
        return ci


def calc_isurv(estimate, intervals):
    hazard = torch.sigmoid(estimate)
    survival = torch.cumprod(1 - hazard, dim=-1)

    times = torch.tensor(intervals, device=estimate.device)
    integrated_survival = trapezoid(y=survival.cpu(), x=times.cpu())
    integrated_survival = torch.from_numpy(integrated_survival)

    return integrated_survival

def calc_mrl(estimate, intervals):
    hazard = torch.sigmoid(estimate)
    survival = F.pad(torch.cumprod(1 - hazard, dim=-1), (1, 0), value=1)    
    pdf = hazard * survival[..., :-1]

    # calculate mean residual lifetime, aka expected lifetime
    times = torch.tensor(intervals, device=estimate.device)
    mrl = torch.sum(times * pdf, axis=-1)

    return mrl
    

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
        self.estimates.append(preds.cpu())

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
