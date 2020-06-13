# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/04_layers.ipynb (unless otherwise specified).

__all__ = ['GRelu', 'JRelu', 'Mish', 'mish', 'ConvBlock', 'ResBlock', 'ReflectionPaddedConv2d', 'Dropin',
           'LinearSkipBlock', 'LinearResBlock', 'LinearDenseBlock', 'WeightedLinearResBlock', 'trunc_normal_',
           'PaddedEmbedding', 'BloomEmbedding', 'AxialEncoding', 'MultiAxialEncoding']


# Cell
from functools import partial
import numpy as np
from operator import add, truediv, sub
import torch
import torch.nn as nn
import torch.nn.functional as F

from htools import add_docstring
from .data import probabilistic_hash_tensor
from .utils import concat, weighted_avg, identity


# Cell
class GRelu(nn.Module):
    """Generic ReLU."""

    def __init__(self, leak=0.0, max=float('inf'), sub=0.0):
        super().__init__()
        self.leak = leak
        self.max = max
        self.sub = sub

    def forward(self, x):
        """Check which operations are necessary to save computation."""
        x = F.leaky_relu(x, self.leak) if self.leak else F.relu(x)
        if self.sub:
            x -= self.sub
        if self.max:
            x = torch.clamp(x, max=self.max)
        return x

    def __repr__(self):
        return f'GReLU(leak={self.leak}, max={self.max}, sub={self.sub})'


# Cell
JRelu = GRelu(leak=.1, sub=.4, max=6.0)


# Cell
class Mish(nn.Module):
    """OOP form of mish activation.

    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    https://arxiv.org/pdf/1908.08681v1.pdf
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# Cell
def mish(x):
    """Functional form of mish activation.

    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    https://arxiv.org/pdf/1908.08681v1.pdf

    Parameters
    ----------
    x: torch.Tensor[float]
        Input tensor.
    Returns
    -------
    torch.Tensor[float]: Tensor of same shape as input x.
    """
    return x * torch.tanh(F.softplus(x))


# Cell
class ConvBlock(nn.Module):
    """Create a convolutional block optionally followed by a batch norm layer.
    """

    def __init__(self, c_in, c_out, kernel_size=3, norm=True, activation=JRelu,
                 **kwargs):
        """
        Parameters
        -----------
        c_in: int
            # of input channels.
        c_out: int
            # of output channels.
        kernel_size: int
            Size of kernel in conv2d layer. An integer argument will be used
            as both the height and width.
        norm: bool
            If True, include a batch norm layer after the conv layer. If False,
            no norm layer will be used. Note that batch norm has learnable
            affine parameters which remove the need for a bias in the preceding
            conv layer. When batch norm is not used, however, the conv layer
            will include a bias term.
        activation: nn.Module
            Activation function to use at the end of the convolutional block.
            (In some cases such as our ResBlock implementation, we pass in None
            so that an extra addition can be performed before the final
            activation.) Do not use the functional form here as it will be
            added to a sequential object.
        kwargs: any
            Additional keyword args are passed to Conv2d. Useful kwargs include
            stride, and padding (see pytorch docs for nn.Conv2d).
        """
        super().__init__()
        self.norm = norm
        layers = [nn.Conv2d(c_in, c_out, kernel_size, bias=not norm, **kwargs)]
        if norm:
            layers.append(nn.BatchNorm2d(c_out))
        if activation is not None:
            layers.append(activation)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# Cell
class ResBlock(nn.Module):

    def __init__(self, c_in, activation=JRelu, f=3, stride=1, pad=1,
                 skip_size=2, norm=True):
        """Residual block using 2D convolutional layers. Note that f,
        stride, and pad must be selected such that the height and width of
        the input remain the same.

        Parameters
        -----------
        c_in: int
            # of input channels.
        activation: callable
            Activation function to use.
        f: int
            Size of filter (f x f) used in convolution. Default 3.
        stride: int
            # of pixels the filter moves between each convolution. Default 1.
        pad: int
            Pixel padding around the input. Default 1.
        skip_size: int
            Number of conv blocks inside the skip connection (default 2).
            ResNet paper notes that skipping a single layer did not show
            noticeable improvements.
        norm: bool
            Specifies whether to include a batch norm layer after each conv
            layer.
        """
        super().__init__()
        self.skip_size = skip_size
        self.layers = nn.ModuleList([
            ConvBlock(c_in, c_in, norm=norm, activation=None, kernel_size=f,
                      stride=stride, padding=pad)
            for i in range(skip_size)
        ])
        self.activation = activation

    def forward(self, x):
        x_out = x
        for i, layer in enumerate(self.layers):
            x_out = layer(x_out)

            # Final activation must be applied after addition.
            if i != self.skip_size - 1:
                x_out = self.activation(x_out)

        return self.activation(x + x_out)


# Cell
@add_docstring(nn.Conv2d)
class ReflectionPaddedConv2d(nn.Module):
    """Conv2d only allows padding_mode of `zeros` or `circular`. This
    layer is a quick way for us to use reflection padding.
    """

    def __init__(self, in_channels, out_channels, padding=1,
                 kernel_size=3, **kwargs):
        """Do not specify a padding mode.
        """
        super().__init__()
        if 'padding_mode' in kwargs:
            raise InvalidArgumentError('Remove `padding_mode` from arguments.')
        self.reflect = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=0)

    def forward(self, x):
        x = self.reflect(x)
        return self.conv(x)


# Cell
class Dropin(nn.Module):
    """Additive dropout. This injects small amounts of noise into a model
    in the form of randomly generated floats from a zero-centered
    gaussian distribution (variance can be adjusted). This does nothing
    in eval mode. Unlike Dropout, this does not scale weights during
    training since it does not bias them in any direction.
    """

    def __init__(self, scale=.5):
        """
        Parameters
        ----------
        scale: float
            Used to scale the magnitude of the random noise. Keep in mind
            that the scalar term is square rooted, so the relationship
            will not be linear. Relatively large values (e.g. 1.0) will have
            a stronger regularizing effect, while small values (e.g. 0.1)
            will have a slight regularizing effect. There is no max value
            enforced, so it's up to the user to select a reasonable value.
        """
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if not self.training:
            return x

        # Storing noise allows us to run diagnostics.
        self.noise = torch.randn_like(x) * np.sqrt(self.scale / x.shape[-1])
        return x + self.noise


# Cell
class LinearSkipBlock(nn.Module):
    """This lets us easily create residual block equivalents with linear
    layers.
    """

    def __init__(self, x_dim, layer_dims, op, activation=mish):
        """
        Parameters
        ----------
        x_dim: int
            Size of input tensor.
        layer_dims: Iterable[int]
            Size of each layer. The length of this list will be the skip size
            (2 is probably a reasonable starting point).
        op: function
            This will be called on the input x and the processed x in the
            forward method. This is a concatenation for dense blocks and an
            addition for residual blocks, but any operation is possible.
        activation: callable
            Activation function or callable class. This will be applied after
            each layer. The final activation is applied after the `op` function.
        """
        super().__init__()
        self.skip_size = len(layer_dims)
        self.activation = activation
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out
                                     in zip([x_dim]+list(layer_dims), layer_dims)])
        self.op = op

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers, 1):
            out = layer(out)
            if i < self.skip_size: out = self.activation(out)
        return self.activation(self.op(x, out))


# Cell
class LinearResBlock(LinearSkipBlock):
    """Equivalent of ResNet block with linear layers."""

    def __init__(self, x_dim, hidden_dims, activation=mish):
        if hidden_dims[-1] != x_dim:
            raise InvalidArgumentError(
                'Last hidden dimension must match input dimension.'
            )
        super().__init__(x_dim, hidden_dims, add, activation)


# Cell
class LinearDenseBlock(LinearSkipBlock):
    """Equivalent of DenseNet block with linear layers."""

    def __init__(self, x_dim, hidden_dims, activation=mish):
        super().__init__(x_dim, hidden_dims, concat, activation)


# Cell
class WeightedLinearResBlock(LinearSkipBlock):
    """Like a LinearResBlock but takes a weighted average of the input and output
    rather than adding them. Addition gives them equal weight and we may want to
    weight the output more heavily.
    """

    def __init__(self, x_dim, hidden_dims, weights=(.25, .75), activation=mish):
        super().__init__(x_dim, hidden_dims,
                         partial(weighted_avg, weights=list(weights)), activation)


# Cell
def trunc_normal_(x, mean=0.0, std=1.0):
    """Ported from fastai to remove dependency:

    Truncated normal initialization.
    From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    """
    return x.normal_().fmod_(2).mul_(std).add_(mean)


# Cell
@add_docstring(nn.Embedding)
def PaddedEmbedding(num_embeddings, embedding_dim, padding_idx=None, **kwargs):
    """Patched version of Fastai `embedding` that allows us to specify a row of
    zeros for a padding token.
    """
    emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx, **kwargs)
    with torch.no_grad():
        trunc_normal_(emb.weight, std=.01)
        if padding_idx is not None:
            torch.zero_(emb.weight[0])
    return emb


# Cell
class BloomEmbedding(nn.Module):
    """Bloom Embedding layer for memory-efficient word representations.
    Each word is encoded by a combination of rows of the embedding
    matrix. The number of rows can therefore be far lower than the number
    of words in our vocabulary while still providing unique representations.
    The reduction in rows allows us to use memory in other ways: a larger
    embedding dimension, more or larger layers after the embedding,
    larger batch sizes, etc.

    Note that if hashing is done in the Dataset, we could use a simple
    nn.EmbeddingBag to achieve the same thing. Many users have reported
    poor performance with this layer though (especially on CPU, but in some
    cases on GPU) so I stick with the standard Embedding. We also bake in
    the truncated normal intialization provided by fastai, with a slight tweak
    to allow a row for padding.
    """

    def __init__(self, n_emb=251, emb_dim=100, n_hashes=4, padding_idx=0,
                 pre_hashed=False):
        """
        Parameters
        ----------
        n_emb: int
            Number of rows to create in the embedding matrix. A prime
            number is recommended. Lower numbers will be more
            memory-efficient but increase the chances of collisions.
        emb_dim: int
            Size of each embedding. If emb_dim=100, each word will
            be represented by a 100-dimensional vector.
        n_hashes: int
            This determines the number of hashes that will be taken
            for each word index, and as a result, the number of rows
            that will be summed to create each unique representation.
            The higher the number, the lower the chances of a collision.
        padding_idx: int or None
            If an integer is provided, this will set aside the corresponding
            row in the embedding matrix as a vector of zeros. If None, no
            padding vector will be allocated.
        pre_hashed: bool
            Pass in True if the input tensor will already be hashed by the time
            it enters this layer (you may prefer pre-compute the hashes in the
            Dataset to save computation time during training). In this
            scenario, the layer is a simple embedding bag with mode "sum".
            Pass in False if the inputs will be word indices that have not yet
            been hashed. In this case, hashing will be done inside the
            `forward` call.

        Suggested values for a vocab size of ~30,000:

        | n_emb | n_hashes | unique combos |
        |-------|----------|---------------|
        | 127   | 5        | 29,998        |
        | 251   | 4        | 29,996        |
        | 997   | 3        | 29,997        |
        | 5,003 | 2        | 29,969        |
        """
        super().__init__()
        self.n_emb = n_emb
        self.emb = PaddedEmbedding(n_emb, emb_dim, padding_idx=padding_idx)
        self.n_hashes = n_hashes
        self.pad_idx = padding_idx
        self.pre_hashed = pre_hashed
        self.process_fn = identity if pre_hashed else \
            partial(probabilistic_hash_tensor, n_buckets=n_emb,
                    n_hashes=n_hashes, pad_idx=pad_idx)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.LongTensor
            Input tensor of word indices (bs x seq_len) if pre_hashed is False.
            Hashed indices (bs x seq_len x n_hashes) if pre_hashed is False.

        Returns
        -------
        torch.FloatTensor: Words encoded with combination of embeddings.
            (bs x seq_len x emb_dim)
        """
        # If not pre-hashed: (bs, seq_len) -> hash -> (bs, seq_len, n_hashes)
        hashed = self.process_fn(x)
        # (bs, seq_len, n_hashes, emb_dim) -> sum -> (bs, seq_len, emb_dim)
        return self.emb(hashed).sum(-2)


# Cell
class AxialEncoding(nn.Module):
    """Axial encodings. These are intended to encode position in a sequence
    (e.g. index in a sentence). It's possible we could adapt these for use as
    word embeddings but this would likely require some experimentation (for
    example, words would likely need to be sorted in a thoughtful manner
    (e.g. pre-trained embeddings compressed to 1D?) since adjacent inputs will
    share half of their encodings).
    """

    def __init__(self, vocab_dim, emb_dim, pad_idx=None):
        """
        Parameters
        ----------
        vocab_dim: int
            Number of words in vocab (or max sequence length if being used for
            positional encodings).
        emb_dim: int
            Size of embedding vectors (often numbers like 50, 100, 300).
        pad_idx: int or None
            If necessary, pass in an integer to represent padding. Otherwise
            no rows are reserved for padding.
        """
        super().__init__()
        if emb_dim % 2 != 0:
            raise ValueError('emb_dim must be an even number.')

        self.v = self._decompose_mult(vocab_dim)
        self.e = self._decompose_add(emb_dim)
        self.emb = nn.ModuleList(PaddedEmbedding(self.v, self.e, pad_idx)
                                 for _ in range(2))

    def _decompose_mult(self, dim):
        return int(np.ceil(np.sqrt(dim)))

    def _decompose_add(self, dim):
        return int(np.ceil(dim / 2))

    def forward(self, idx):
        return torch.cat([self.emb[0](idx%self.v), self.emb[1](idx//self.v)],
                         dim=-1)


# Cell
class MultiAxialEncoding(nn.Module):
    """Adapted axial encodings to allow for more than 2 embedding matrices.
    These are intended to encode position in a sequence (e.g. index in a
    sentence) but might work as word embeddings. This version may be better
    suited for that use case because using more blocks results in fewer shared
    numbers in the output vectors of adjacent inputs.

    Some experimentation is still required for this use case (for
    example, words would likely need to be sorted in a thoughtful manner
    (e.g. pre-trained embeddings compressed to 1D?) since adjacent inputs will
    share half of their encodings).
    """

    def __init__(self, vocab_dim, emb_dim, n_blocks=2, pre_hashed=False,
                 pad_idx=None):
        super().__init__()
        # Must set n_blocks before computing v or e.
        self.n_blocks = n_blocks
        self.v = self._decompose_mult(vocab_dim)
        self.e = self._decompose_add(emb_dim)
        self.pre_hashed = pre_hashed
        # Must set emb blocks before defining process_fn.
        self.emb = nn.ModuleList(PaddedEmbedding(self.v, self.e, pad_idx)
                          for _ in range(n_blocks))
        self.process_fn = identity if pre_hashed else \
            partial(probabilistic_hash_tensor, n_buckets=self.v,
                    n_hashes=len(self.emb), pad_idx=pad_idx)
        self.emb_dim = self.e * self.n_blocks

    def _decompose_mult(self, dim):
        return int(np.ceil(dim ** (1 / self.n_blocks)))

    def _decompose_add(self, dim):
        return int(np.ceil(dim // self.n_blocks))

    def forward(self, idx):
        # Hashed shape: (bs, seq_len, n_hashes)
        xhash = self.process_fn(idx)
        # Each embedding takes in a tensor of shape (bs, seq_len).
        res_blocks = [e(hashed.squeeze()) for e, hashed in
                      zip(self.emb, torch.chunk(xhash, xhash.shape[0], -1))]
        return torch.cat(res_blocks, dim=-1)