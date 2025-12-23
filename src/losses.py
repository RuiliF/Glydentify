import torch
import torch.nn.functional as F
import torch.nn as nn

@torch.no_grad()
def class_alpha_from_counts(counts, smooth=1.0, cap=(0.25, 4.0)):
    c = torch.as_tensor(counts, dtype=torch.float32)
    C = c.numel()
    N = c.sum().clamp_min(1.0)
    w = N / (C * (c + smooth))                  # inverse frequency w_c
    w = torch.clamp(w, min=cap[0], max=cap[1])  # clamp extremes
    w = w / w.mean().clamp_min(1e-8)            # mean ~ 1
    return w                                     # use as α_c

class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        if alpha is not None:
            self.alpha = torch.as_tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = torch.ones(1, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        # CE term (stable)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # (N,C)
        # p_t = exp(-CE) is numerically stable (since CE = -log p_t)
        pt = torch.exp(-ce)

        self.alpha = self.alpha.to(logits.device)
        loss = self.alpha * (1 - pt).pow(self.gamma) * ce  # focal modulation
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        print(f"Using AsymmetricLoss: gamma_pos={gamma_pos}, gamma_neg={gamma_neg}, clip={clip}")
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()

class AsymmetricFocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha=None, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, reduction="mean"):
        super().__init__()
        if alpha is not None:
            self.alpha = torch.as_tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = torch.ones(1, dtype=torch.float32)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction
    
    def forward(self, logits, targets):
        targets = targets.to(logits.device)
        # Probabilities
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Optional clipping of negative probabilities
        if self.clip is not None and self.clip > 0:
            xs_neg = torch.clamp(xs_neg + self.clip, max=1.0)

        # Focal modulation (asymmetric)
        mod_pos = (1 - xs_pos).pow(self.gamma_pos)
        mod_neg = (xs_pos).pow(self.gamma_neg)

        self.alpha = self.alpha.to(logits.device)

        # BCE parts
        loss_pos = -targets * torch.log(xs_pos.clamp_min(1e-8)) * mod_pos * self.alpha
        loss_neg = -(1 - targets) * torch.log(xs_neg.clamp_min(1e-8)) * mod_neg * self.alpha

        loss = loss_pos + loss_neg
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class WeightedBCELoss(nn.Module):
    def __init__(self, alpha=None, pos_weight=None, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (B, C), raw model outputs
        targets: (B, C), 0/1 float tensor
        """

        if self.pos_weight is not None:
            self.pos_weight = self.pos_weight.to(logits.device)
        # element‑wise BCE with per‑class pos_weight
        loss_raw = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none"        # (B, C) tensor
        )
        
        # weight each class
        if self.alpha is not None:
            self.alpha = self.alpha.to(logits.device)
            loss_weighted = loss_raw * self.alpha.unsqueeze(0)  # broadcast to (B, C)
        else:
            loss_weighted = loss_raw
        # average over batch and classes
        return loss_weighted.mean()

def get_criterion(criterion_name, alpha=None, pos_weight=None, gamma=2.0, gamma_pos=0.0, clip=0.05):
    if criterion_name == "bce":
        return nn.BCEWithLogitsLoss()
    elif criterion_name == "weighted_bce":
        return WeightedBCELoss(alpha=alpha, pos_weight=pos_weight)
    elif criterion_name == "focal":
        return FocalBCEWithLogitsLoss(alpha=alpha, gamma=gamma)
    elif criterion_name == "asl":
        return AsymmetricLoss(gamma_neg=gamma, gamma_pos=gamma_pos, clip=clip)
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")
