# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/08_losses.ipynb (unless otherwise specified).

__all__ = ['smooth_soft_labels', 'soft_label_cross_entropy_with_logits', 'soft_label_cross_entropy']


# Cell
import torch
import torch.nn.functional as F

from htools import add_docstring


# Cell
def smooth_soft_labels(labels, alpha=.1):
    """Add uniform probability to a tensor of soft labels or
    one-hot-encoded hard labels. Classes with zero probability
    will therefore end up with alpha/k probability, where k is
    the total number of classes. Note that some implementations
    may use alpha/(k-1), so if you notice a difference in output
    that could be the source.

    Parameters
    ----------
    labels: torch.tensor
        Tensor of labels where each row is a sample and each column
        is a class. These can be one hot encodings or a vector of soft
        probabilities. Shape (batch_size, num_classes).
    alpha: float
        A positive value which will be used to assign nonzero
        probabilities to the classes that are currently zeros. A larger
        alpha corresponds to a higher degree of smoothing (useful when
        accounting for noisier labels, trying to provide a stronger
        regularizing effect, or encouraging less confident predictions).
    """
    if alpha < 0:
        raise InvalidArgumentError('Alpha must be non-negative.')

    # Avoid unnecessary computation.
    if not alpha:
        return labels
    length = labels.shape[-1]
    nonzeros = (labels > 0).sum(-1).unsqueeze(-1).float()
    return torch.clamp_min(labels - alpha/nonzeros, 0) + alpha/length


# Cell
def soft_label_cross_entropy_with_logits(y_pred, y_true, alpha=0.0,
                                         reduction='mean'):
    """Compute cross entropy with soft labels. PyTorch's built in
    multiclass cross entropy functions require us to pass in integer
    indices, which doesn't allow for soft labels which are shaped like
    a one hot encoding. FastAI's label smoothing loss uniformly divides
    uncertainty over all classes, which again does not allow us to pass
    in our own soft labels.

    Parameters
    ----------
    y_pred: torch.FloatTensor
        Logits output by the model.
        Shape (bs, num_classes).
    y_true: torch.FloatTensor
        Soft labels, where values are between 0 and 1.
        Shape (bs, num_classes).
    alpha: float
        Label smoothing hyperparameter: a positive value which will be used to
        assign nonzero probabilities to the classes that are currently zeros.
        A larger alpha corresponds to a higher degree of smoothing (useful when
        accounting for noisier labels, trying to provide a stronger
        regularizing effect, or encouraging less confident predictions).
    reduction: str
        One of ('mean', 'sum', 'none'). This determines how to reduce
        the output of the function, similar to most PyTorch
        loss functions.

    Returns
    -------
    torch.FloatTensor: If reduction is 'none', this will have shape
        (bs, ). If 'mean' or 'sum', this will be be a tensor with a
        single value (no shape).
    """
    res = (-smooth_soft_labels(y_true, alpha)
           * F.log_softmax(y_pred, dim=-1)).sum(-1)
    if reduction == 'none': return res
    return getattr(res, reduction)(0)


# Cell
@add_docstring(soft_label_cross_entropy_with_logits)
def soft_label_cross_entropy(y_pred, y_true, alpha=0.0, reduction='mean'):
    """Same as `soft_label_cross_entropy_with_logits` but operates on
    softmax output instead of logits. The version with logits is
    recommended for numerical stability. Below is the docstring for the logits
    version. The only difference in this version is that y_pred will not be
    logits.
    """
    res = -smooth_soft_labels(y_true, alpha) * torch.log(y_pred)
    res = torch.where(torch.isnan(res) | torch.isinf(res),
                      torch.zeros_like(res), res).sum(-1)
    if reduction == 'none': return res
    return getattr(res, reduction)(0)