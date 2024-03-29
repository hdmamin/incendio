# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/07_data.ipynb (unless otherwise specified).

__all__ = ['probabilistic_hash_item', 'probabilistic_hash_tensor', 'plot_images', 'RandomTransform', 'RandomPipeline',
           'dataloader_subset', 'BotoUploader', 'plot_images']


# Cell
import boto3
from collections import deque
from functools import partial
import mmh3
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import warnings

from htools import auto_repr, save, load, func_name, ifnone, tolist, \
    BasicPipeline


# Cell
def probabilistic_hash_item(x, n_buckets, mode=int, n_hashes=3):
    """Slightly hacky way to probabilistically hash an integer by
    first converting it to a string.

    Parameters
    ----------
    x: int
        The integer or string to hash.
    n_buckets: int
        The number of buckets that items will be mapped to. Typically
        this would occur outside the hashing function, but since
        the intended use case is so narrow here it makes sense to me
        to include it here.
    mode: type
        The type of input you want to hash. This is user-provided to prevent
        accidents where we pass in a different item than intended and hash
        the wrong thing. One of (int, str). When using this inside a
        BloomEmbedding layer, this must be `int` because there are no
        string tensors. When used inside a dataset or as a one-time
        pre-processing step, you can choose either as long as you
        pass in the appropriate inputs.
    n_hashes: int
        The number of times to hash x, each time with a different seed.

    Returns
    -------
    list[int]: A list of integers with length `n_hashes`, where each integer
        is in [0, n_buckets).
    """
    # Check type to ensure we don't accidentally hash Tensor(5) instead of 5.
    assert isinstance(x, mode), f'Input `x` must have type {mode}.'
    return [mmh3.hash(str(x), i, signed=False) % n_buckets
            for i in range(n_hashes)]


# Cell
def probabilistic_hash_tensor(x_r2, n_buckets, n_hashes=3, pad_idx=0):
    """Hash a rank 2 LongTensor.

    Parameters
    ----------
    x_r2: torch.LongTensor
        Rank 2 tensor of integers. Shape: (bs, seq_len)
    n_buckets: int
        Number of buckets to hash items into (i.e. the number of
        rows in the embedding matrix). Typically a moderately large
        prime number, like 251 or 997.
    n_hashes: int
        Number of hashes to take for each input index. This determines
        the number of rows of the embedding matrix that will be summed
        to get the representation for each word. Typically 2-5.
    pad_idx: int or None
        If you want to pad sequences with vectors of zeros, pass in an
        integer (same as the `padding_idx` argument to nn.Embedding).
        If None, no padding index will be used. The sequences must be
        padded before passing them into this function.

    Returns
    -------
    torch.LongTensor: Tensor of indices where each row corresponds
        to one of the input indices. Shape: (bs, seq_len, n_hashes)
    """
    return torch.tensor(
        [[probabilistic_hash_item(x.item(), n_buckets, int, n_hashes)
          if x != pad_idx else [pad_idx]*n_hashes for x in row]
         for row in x_r2]
    )


# Cell
def plot_images(images, titles=None, nrows=None, figsize=None,
                tight_layout=True, title_colors=None):
    nrows = nrows or int(np.ceil(np.sqrt(len(images))))
    titles = ifnone(titles, [None] * len(images))
    colors = ifnone(title_colors, ['black'] * len(images))
    figsize = figsize or (nrows*2, nrows*2)
    fig, ax = plt.subplots(nrows, nrows, figsize=figsize)
    for axi, img, title, color in zip(ax.flatten(), images, titles, colors):
        axi.imshow(img.permute(1, 2, 0))
        axi.set_title(title, color=color)
        axi.set_axis_off()
    if tight_layout: plt.tight_layout()
    plt.show()


# Cell
class RandomTransform:
    """Wrap a function to create a data transform that occurs with some
    probability p.
    """

    def __init__(self, func, p=.5):
        """
        Parameters
        ----------
        func: function
            Transforms an input tensor (x) in some way. X will be the first
            argument passed in but the function can accept additional
            arguments.
        p: float
            Between 0 and 1, determines the likelihood that func will be
            applied to any input x.
        """
        self.func = func
        self.p = p

    def __call__(self, x, *args, **kwargs):
        """Additional args and kwargs are forwarded to self.func."""
        if np.random.uniform() < self.p: x = self.func(x, *args, **kwargs)
        return x

    def __repr__(self):
        return f'{type(self).__name__}({func_name(self.func)}, p={self.p})'


# Cell
class RandomPipeline(BasicPipeline):
    """Create a pipeline of callables that are applied in sequence, each with
    some random probability p (this can be the same or different for each
    step). This is useful for on-the-fly data augmentation (think in the
    __getitem__ method of a torch Dataset).
    """

    def __init__(self, *transforms, p=.5):
        """
        Parameters
        ----------
        transforms: callable
            Functions or callable classes that accept a single argument (use
            functools.partial if necessary). They will be applied in the order
            you pass them in.
        p: float or Iterable[float]
            Probability that each transform will be applied. If a single
            float, each transform will have the same probability. If a list,
            its length msut match the number of transforms passed in: p[0]
            will be assigned to transforms[0], p[1] to transforms[1], and so
            on.
        """
        p = tolist(p, transforms, error_message='p must be a float or a list '
                   'with one float for each transform.')
        if any(n <= 0 or n > 1 for n in p):
            raise ValueError('p must be in range (0, 1]. I.E. you can choose '
                             'to always apply a transform, but if you never '
                             'want to apply it there\'s no need to include '
                             'it in the pipeline.')

        super().__init__(*[RandomTransform(t, p_)
                           for t, p_ in zip(transforms, p)])

    @classmethod
    def from_dict(cls, t2p):
        """
        Parameters
        ----------
        t2p: dict[callable, float]
            Maps transform to its corresponding probability.

        Examples
        --------
        transforms = {times_3: .33,
                      to_string: 1.0,
                      dashed_join: .67,
                      to_upper: .95}
        pipeline = RandomPipeline.from_dict(transforms)
        """
        return cls(*t2p.keys(), p=t2p.values())


# Cell
def dataloader_subset(dl, batches=1):
    """Like torch's dataset Subset but for dataloaders. This lets us easily
    do things like fit on a single batch (or small number of batches) by
    essentially copying an existing dataloader's options but using a smaller
    dataset. A few limitations: custom samplers and batch_samplers are not
    supported for reasons explained the error message within this function.

    Parameters
    ----------
    dl: torch.utils.data.DataLoader
    batches: int
        Number of batches to use in the new dataloader.

    Returns
    -------
    DataLoader: Should be identical to the input dataloader except it operates
    one a subset of that dataset.

    Examples
    --------
    train_dataset = MyDataset()
    train_dataloader = DataLoader(train_dataset, shuffle=True, drop_last=)
    """
    ds_sub = torch.utils.data.Subset(dl.dataset,
                                     range(dl.batch_size * batches))

    # Check with type rather than isinstance since subclasses of these types
    # should raise an error.
    custom_sampler = False
    if type(dl.sampler) == torch.utils.data.sampler.SequentialSampler:
        shuffle = False
    elif type(dl.sampler) == torch.utils.data.sampler.RandomSampler:
        shuffle = True
    else:
        custom_sampler = True

    if custom_sampler or (type(dl.batch_sampler)
                          != torch.utils.data.sampler.BatchSampler):
        raise RuntimeError(
            'dataloader_subset does not support custom samplers or '
            'batch_samplers. Sampler constructors can require non-standard '
            'args so we can\'t reliably create a new instance automatically, '
            'and they often require a dataset as an argument so we can\'t '
            'simply copy the existing instance - that uses the full dataset '
            'whereas we want a subset.'
        )

    # Exclude shuffle since this isn't stored as a dataloader attribute.
    # It can be inferred from samplers if the user specified a shuffle value
    # rather than passing in non-default sampler objects.
    defaults = {'batch_size', 'num_workers', 'collate_fn', 'pin_memory',
                'drop_last', 'timeout', 'worker_init_fn', 'prefetch_factor',
                'persistent_workers'}
    kwargs = {attr: getattr(dl, attr) for attr in defaults}
    return DataLoader(ds_sub, shuffle=shuffle, **kwargs)


# Cell
class BotoUploader:
    """Uploads files to S3. Built as a public alternative to Accio. Note to
    self: the interfaces are not identical so be careful to know which you're
    using.
    """

    def __init__(self, bucket, verbose=True):
        """
        Parameters
        ----------
        bucket: str
            Name of s3 bucket to upload to. For a single project, I generally
            stick to a single bucket so we can usually keep this fixed. We can
            always change the attribute later if necessary.
        verbose: bool
            If True, print message when downloading each file.
        """
        self.s3 = boto3.resource('s3')
        self.bucket = bucket
        self.verbose = verbose

    def upload_file(self, path, s3_dir='', retain_tree=True):
        """Upload a single local file. By default, its path in S3 will be the
        same as its local path. Usually, this means your S3 bucket will have a
        single directory called "data" which corresponds exactly to your local
        "data" directory.

        Parameters
        ----------
        path: str or Path
            Local path to the file to upload. This must be relative to the
            project root directory (e.g. "data/models/v1/model.gz").
        s3_dir: str
            If provided, this will pre prepended to each path: {s3_dir}/{path}.
            Otherwise, s3 paths will be the same as local paths.
        retain_tree: bool
            If True, the local file structure will be retained. Otherwise,
            only the base name is kept. All four combinations of retain_tree
            (True/False) and s3_dir (empty/non-empty) are supported.
        """
        path = str(path)
        s3_path = self._convert_local_path(path, s3_dir, retain_tree)
        if self.verbose: print(f'Uploading {path} -> {s3_path}.')
        self.s3.meta.client.upload_file(path, self.bucket, s3_path)

    def upload_files(self, paths, s3_dir='', retain_tree=True):
        """Upload multiple files. Only a ~2x speedup over brute force, but I
        tried threads and that provided no speedup 🤷‍♂️.

        Parameters
        ----------
        paths: Iterable[str or Path]
            Sequence of file paths to upload.
        s3_dir: str
            If provided, this will pre prepended to each path: {s3_dir}/{path}.
            Otherwise, s3 paths will be the same as local paths.
        retain_tree: bool
            If True, the local file structure will be retained. Otherwise,
            only the base name is kept. All four combinations of retain_tree
            (True/False) and s3_dir (empty/non-empty) are supported.
        """
        func = partial(self.upload_file, s3_dir=s3_dir,
                       retain_tree=retain_tree)
        with Pool() as p:
            p.map(func, paths)

    def upload_folder(self, dirname, s3_dir, retain_tree=True, recurse=True,
                      keep_fn=None):
        """Upload all files in a directory.

        Parameters
        ----------
        dirname: str or Path
            Directory to upload.
        s3_dir: str
            If provided, this will pre prepended to each path: {s3_dir}/{path}.
            Otherwise, s3 paths will be the same as local paths.
        retain_tree: bool
            If True, the local file structure will be retained. Otherwise,
            only the base name is kept. When uploading recursively, we require
            that the file tree is retained.
        recurse: bool
            If True, upload all files in subdirectories as well.
        keep_fn: None or callable
            If provided, this should be a function that accepts a filename as
            input and returns a boolean specifying whether to include it in the
            upload or not. For example:

            lambda x: os.path.splitext(x)[-1] != '.pkl'

            keeps all files except those with an '.pkl' extension.
        """
        if recurse and not retain_tree:
            raise ValueError('retain_tree must be True when uploading '
                             'recursively.')
        pat = os.path.join(str(dirname), '**' if recurse else '*')
        # glob's recursive option only has an effect when using '**'.
        paths = (o for o in glob(pat, recursive=True) if os.path.isfile(o))
        if keep_fn: paths = filter(keep_fn, paths)
        self.upload_files(paths, s3_dir, retain_tree)

    def _convert_local_path(self, path, s3_dir='', retain_tree=True):
        """Convert local path to s3 path. See public methods for parameter
        documentation.
        """
        path = path if retain_tree else os.path.basename(path)
        return os.path.join(s3_dir, path)

    def __getstate__(self):
        return {'bucket': self.bucket,
                'verbose': self.verbose}

    def __setstate__(self, data):
        self.bucket = data['bucket']
        self.verbose = data['verbose']
        self.s3 = boto3.resource('s3')

    def __eq__(self, other):
        return type(other) == type(self) \
            and self.bucket == other.bucket \
            and self.s3 == other.s3 \
            and self.verbose == other.verbose


# Cell
def plot_images(images, titles=None, nrows=None, figsize=None,
                tight_layout=True, title_colors=None):
    """Plot a grid of images.

    Parameters
    ----------
    images: Iterable
        List of tensors/arrays to plot.
    titles: Iterable[str] or None
        Title for each subplot. Must be same length and order as `images`.
    nrows: int or None
        If provided, this manually sets the number of rows in the grid.
    figsize: tuple[int] or None
        Determines size of plot. By default, we double the number of rows and
        columns, respectively, to get dimensions.
    tight_layout: bool
        Often helps matplotlib formatting when we have many images.
    title_colors: Iterable[str] or None
        If provided, this should have the same length as `titles` and
        `images`. It will be used to determine the color of each title (see
        PredictionExaminer in incendio.core for an example).
    """
    nrows = nrows or int(np.ceil(np.sqrt(len(images))))
    titles = ifnone(titles, [None] * len(images))
    colors = ifnone(title_colors, ['black'] * len(images))
    figsize = figsize or (nrows*2, nrows*2)
    fig, ax = plt.subplots(nrows, nrows, figsize=figsize)
    for axi, img, title, color in zip(ax.flatten(), images, titles, colors):
        axi.imshow(np.transpose(img, (1, 2, 0)))
        axi.set_title(title, color=color)
        axi.set_axis_off()
    if tight_layout: plt.tight_layout()
    plt.show()