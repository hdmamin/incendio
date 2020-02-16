#AUTOGENERATED! DO NOT EDIT! File to edit: dev/01_utils.ipynb (unless otherwise specified).

__all__ = ['hasarg', 'quick_stats']

#Cell
import inspect

#Cell
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

#Cell
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