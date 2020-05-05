# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/01_utils.ipynb (unless otherwise specified).

__all__ = ['DEVICE', 'hasarg', 'quick_stats', 'concat', 'weighted_avg', 'identity']


# Cell
import inspect
import torch


# Cell
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Cell
def hasarg(func, arg):
    """Checks if a function has a given argument.
    Works with args and kwargs as well if you exclude the
    stars. See example below.

    Parameters
    ----------
    func: function
    arg: str
        Name of argument to look for.

    Returns
    -------
    bool

    Example
    -------
    def foo(a, b=6, *args):
        return

    >>> hasarg(foo, 'b')
    True

    >>> hasarg(foo, 'args')
    True

    >>> hasarg(foo, 'c')
    False
    """
    return arg in inspect.signature(func).parameters


# Cell
def quick_stats(x, digits=3):
    """Quick wrapper to get mean and standard deviation of a tensor.

    Parameters
    ----------
    x: torch.Tensor
    digits: int
        Number of digits to round mean and standard deviation to.

    Returns
    -------
    tuple[float]
    """
    return round(x.mean().item(), digits), round(x.std().item(), digits)


# Cell
def concat(*args, dim=-1):
    """Wrapper to torch.cat which accepts tensors as non-keyword
    arguments rather than requiring them to be wrapped in a list.
    This can be useful if we've built some generalized functionality
    where parameters must be passed in a consistent manner.

    Parameters
    ----------
    args: torch.tensor
        Multiple tensors to concatenate.
    dim: int
        Dimension to concatenate on (last dimension by default).

    Returns
    -------
    torch.tensor
    """
    return torch.cat(args, dim=dim)


# Cell
def weighted_avg(*args, weights):
    """Compute a weighted average of multiple tensors.

    Parameters
    ----------
    args: torch.tensor
        Multiple tensors with the same dtype and shape that you want to average.
    weights: list
        Ints or floats to weight each input tensor. The length of this list must
        match the number of tensors passed in: the first weight will be multiplied
        by the first tensor, the second weight by the second tensor, etc. If your
        weights don't sum to 1, they will be normalized automatically.

    Returns
    -------
    torch.tensor: Same dtype and shape as each of the input tensors.
    """
    weights = torch.tensor(weights)
    total = weights.sum().float()
    if total != 1: weights = weights / total
    res = torch.stack(args)
    weights_shape = [-1 if i == 0 else 1 for i, _ in enumerate(range(res.ndim))]
    return torch.mean(res * weights.view(*weights_shape), axis=0)


# Cell
def identity(x):
    """Temporarily copied from htools.

    Returns the input argument. Sometimes it is convenient to have this if
    we sometimes apply a function to an item: rather than defining a None
    variable, sometimes setting it to a function, then checking if it's None
    every time we're about to call it, we can set the default as identity and
    safely call it without checking.
    Parameters
    ----------
    x: any
    Returns
    -------
    x: Unchanged input.
    """
    return x