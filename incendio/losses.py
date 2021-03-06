# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/08_losses.ipynb (unless otherwise specified).

__all__ = ['smooth_soft_labels', 'soft_label_cross_entropy_with_logits', 'soft_label_cross_entropy',
           'PairwiseLossReduction', 'reduce', 'contrastive_loss', 'ContrastiveLoss1d', 'ContrastiveLoss2d']


# Cell
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from htools import add_docstring, valuecheck, identity
from .layers import SmoothLogSoftmax


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


# Cell
class PairwiseLossReduction(nn.Module):
    """Basically lets us use L2 or L1 distance as a loss function with the
    standard reductions. If we don't want to reduce, we could use the built-in
    torch function, but that will usually output a tensor rather than a
    scalar.
    """

    @valuecheck
    def __init__(self, reduce:('sum', 'mean', 'none')='mean', **kwargs):
        super().__init__()
        self.distance = nn.PairwiseDistance(**kwargs)
        self.reduce = identity if reduce == 'none' else getattr(torch, reduce)

    def forward(self, y_proba, y_true):
        return self.reduce(self.distance(y_proba, y_true))


# Cell
def reduce(x, reduction='mean'):
    """This is a very common line in my custom loss functions so I'm providing
    a convenience function for it. I think this function call is also more
    intuitive than the code behind it.

    Parameters
    ----------
    x: torch.Tensor
        The object to reduce.
    reduction: str
        Should be one of ('mean', 'sum', 'none'), though technically some
        other operations (e.g. 'std') are supported.

    Returns
    -------
    torch.Tensor: Scalar if using 'mean' or 'sum', otherwise the same as
    input `x`.

    Examples
    --------
    def squared_error(x, reduction='mean'):
        return reduce(x.pow(2), reduction)
    """
    return x if reduction == 'none' else getattr(torch, reduction)(x)


# Cell
def contrastive_loss(x1, x2, y, m=1.25, p=2, reduction='mean'):
    """Functional form of the contrastive loss as described by Hadsell,
    Chopra, and LeCun in
    "Dimensionality Reduction by Learning an Invariant Mapping":
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    Similar examples always contribute to the loss while negatives only do
    if they are sufficiently similar.

    A couple words of caution: this uses the convention that y=1 represents a
    positive label (similar examples) despite the paper doing the opposite.
    Also, the value of m (margin) might benefit from some tuning as I'm not
    very confident in the default value.

    Parameters
    ----------
    x1: torch.Tensor
        Shape (bs, n_features).
    x2: torch.Tensor
        Shape (bs, n_features).
    y: torch.Tensor
        Labels. Unlike the paper, we use the convention that a label of 1
        means images are similar. This is consistent with all our existing
        datasets and just feels more intuitive.
    m: float
        Margin that prevents dissimilar pairs from affecting the loss unless
        they are sufficiently far apart. I believe the reasonable range of
        values depends on the size of the feature dimension. The default is
        based on a figure in the paper linked above but I'm not sure how
        much stock to put in that.
    p: int
        The p that determines the p-norm used to calculate the initial
        distance measure between x1 and x2. The default of 2 therefore uses
        euclidean distance.
    reduction: str
        One of ('sum', 'mean', 'none'). Standard pytorch loss reduction. Keep
        in mind 'none' will probably not allow backpropagation since it
        returns a rank 2 tensor.

    Returns
    -------
    torch.Tensor: Scalar measuring the contrastive loss. If no reduction is
    applied, this will instead be a tensor of shape (bs,).
    """
    dw = F.pairwise_distance(x1, x2, p, keepdim=True)
    # Loss_similar + Loss_different
    res = y*dw.pow(p).div(2) + (1-y)*torch.clamp_min(m-dw, 0).pow(p).div(2)
    return reduce(res, reduction)


# Cell
class ContrastiveLoss1d(nn.Module):

    @add_docstring(contrastive_loss)
    def __init__(self, m=1.25, p=2, reduction='mean'):
        """OOP version of contrastive loss. The docs for the functional
        version are below:
        """
        super().__init__()
        self.m = m
        self.p = p
        self.reduction = reduction
        self.loss = partial(contrastive_loss, m=m, p=p, reduction=reduction)

    def forward(self, x1, x2, y_true):
        """
        Parameters
        ----------
        x1: torch.Tensor
            Shape (bs, feature_dim).
        x2: torch.Tensor
            Shape (bs, feature_dim).
        y_true: torch.Tensor
            Shape (bs, 1). 1's indicate the inputs are "similar", 0's indicate
            they are dissimilar.

        Returns
        -------
        torch.Tensor: scalar if `reduction` is not 'none', otherwise tensor
        has same shape as `y_true`.
        """
        assert y_true.ndim == 2, "y_true must be rank 2."
        return self.loss(x1, x2, y_true)


# Cell
class ContrastiveLoss2d(nn.Module):

    @add_docstring(contrastive_loss)
    def __init__(self, m=1.25, p=2, reduction='mean'):
        """OOP version of contrastive loss. We're using the name "2d" somewhat
        differently here: we use this module to compare 1 image to n different
        variants. Picture, for instance, a task where a single example
        contains n+1 images and we want to find which of the final n images
        are similar to the first image. Concretely, this would be a
        multi-label (not multi-class) classification problem with OHE labels.

        The docs for the functional
        version are below:
        """
        super().__init__()
        self.m = m
        self.p = p
        self.loss = partial(contrastive_loss, m=m, p=p, reduction='none')

        if reduction == 'none':
            self.reduction = identity
        elif reduction == 'row':
            self.reduction = partial(torch.sum, dim=-1)
        else:
            self.reduction = getattr(torch, reduction)

    def forward(self, x1, x2, y_true):
        """
        Parameters
        ----------
        x1: torch.Tensor
            Shape (bs, feats).
        x2: torch.Tensor
            Shape (bs, n_item, n_feats).
        y_true:
            Shape (bs, n_item).

        Returns
        ---------
        torch.Tensor: scalar if reduction is 'mean' or 'sum', same shape as y
        """
        # if reduction is 'none', or shape (bs,) if reduction is 'row'.
        bs, n, dim = x2.shape
        res = self.loss(x1.repeat_interleave(n, dim=0),
                        x2.view(-1, dim),
                        y_true.view(-1, 1))
        return self.reduction(res.view(bs, -1))