# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/00_core.ipynb (unless otherwise specified).

__all__ = ['BaseModel', 'adam', 'handle_interrupt', 'Trainer']


# Cell
from collections import defaultdict
from collections.abc import Iterable
from functools import partial, wraps
import matplotlib.pyplot as plt
import os
import shutil
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm
import warnings

from htools import load, save, LoggerMixin, valuecheck, hasarg
from .callbacks import BasicConfig, StatsHandler, MetricPrinter
from .metrics import batch_size
from .optimizers import variable_lr_optimizer, update_optimizer
from .utils import quick_stats, DEVICE, identity


# Cell
class BaseModel(nn.Module):

    def unfreeze(self, n_layers=None, n_groups=None):
        """Pass in either the number of layers or number of groups to
        unfreeze. Unfreezing always starts at the end of the network and moves
        backward (e.g. n_layers=1 will unfreeze the last 1 layer, or n_groups=2
        will unfreeze the last 2 groups.) Remember than weights and biases are
        treated as separate layers.

        Parameters
        ----------
        n_layers: int or None
            Number of layers to unfreeze.
        n_groups: int or None
            Number of layer groups to unfreeze. For this to work, the model
            must define an attribute `groups` containing the layer groups.
            Each group can be a layer, a nn.Sequential object, or
            nn.Module.
        """
        if n_groups is not None:
            self._unfreeze_by_group(n_groups)
            return

        length = len(self)
        for i, p in enumerate(self.parameters()):
            p.requires_grad = i >= length - n_layers

    def freeze(self):
        """Freeze whole network. Mostly used for testing."""
        self.unfreeze(n_layers=0)

    def _unfreeze_by_group(self, n_groups):
        """Helper for unfreeze() method.

        Parameters
        ----------
        n_groups: int
            Number of groups to unfreeze, starting at the end of the network.
        """
        length = len(self.groups)
        for i, group in enumerate(self.groups):
            setting = i >= length - n_groups
            for p in group.parameters():
                p.requires_grad = setting

    def __len__(self):
        """Number of parameter matrices in model (basically number of layers,
        except that biases are counted separately).
        """
        return sum(1 for p in self.parameters())

    def dims(self):
        """Get shape of each layer's weights."""
        return [tuple(p.shape) for p in self.parameters()]

    def trainable(self):
        """Check which layers are trainable."""
        return [(tuple(p.shape), p.requires_grad) for p in self.parameters()]

    def weight_stats(self):
        """Check mean and standard deviation of each layer's weights."""
        return [quick_stats(p.data, 3) for p in self.parameters()]

    def plot_weights(self):
        """Plot histograms of each layer's weights."""
        n_layers = len(self.dims())
        fig, ax = plt.subplots(n_layers, figsize=(8, n_layers * 1.25))
        if not isinstance(ax, Iterable): ax = [ax]
        for i, p in enumerate(self.parameters()):
            ax[i].hist(p.data.flatten())
            ax[i].set_title(
                f'Shape: {tuple(p.shape)} Stats: {quick_stats(p.data)}'
            )
        plt.tight_layout()
        plt.show()


# Cell
adam = partial(torch.optim.Adam, eps=1e-3)


# Cell
def handle_interrupt(meth):
    """Decorator for Trainer.fit() method that allows the user to
    interrupt training with Ctrl+c while still running the
    `on_train_end` method for each of its callbacks. Without this,
    cutting training short would lose that functionality, so we
    couldn't do things like uploading to S3.

    Arguments
    ---------
    meth: callable
        The method to decorate.

    Returns
    -------
    callable: The wrapped method.
    """
    @wraps(meth)
    def wrapper(*args, **kwargs):
        instance = args[0]
        try:
            meth(*args, **kwargs)
        except KeyboardInterrupt:
            instance.logger.info(f'Stop training due to KeyboardInterrupt.')
            instance._stop_training = True
            # Dummy values used for epoch and stats to indicate that
            # training was interrupted. `fit()` method returns None.
            _ = instance.decide_stop('on_train_end', -1, {}, {})
    return wrapper


# Cell
class Trainer(LoggerMixin):

    @valuecheck
    def __init__(self, net, ds_train, ds_val, dl_train, dl_val,
                 criterion, mode:('binary', 'multiclass', 'regression'),
                 out_dir, optim=None, optim_type=Adam, eps=1e-3, last_act=None,
                 threshold=0.5, metrics=None, callbacks=None, device=DEVICE):
        """An object to handle model training. This makes it easy for us to
        model weights, optimizer state, datasets and dataloaders all
        at once.

        Parameters
        ----------
        net: BaseModel (inherits from nn.Module)
            A pytorch model. The BaseModel implementation from this library
            should be used, since Trainer relies on its `unfreeze` method.
        ds_train: torch.utils.data.Dataset
            Training dataset.
        ds_val: torch.utils.data.Dataset
            Validation dataset.
        dl_train: torch.utils.data.DataLoader
            Training dataloader. Lazily retrieves items from train dataset.
        dl_val: torch.utils.data.DataLoader
            Validation dataloader. Lazily retrieves items from val dataset.
        criterion: callable
            Typically a PyTorch loss function, but you could define your
            own as long as it accepts the same arguments in the same order.
            This can be a function (e.g. F.cross_entropy) or a callable
            object (e.g. nn.CrossEntropyLoss(). Notice that this is the
            object, not the class.)
        mode: str
            Specifies the problem type. Multi-label classification is
            considered 'binary' as well since each example receives a binary
            prediction for each class.
        out_dir: str
            The path to an output directory where logs, model weights, and
            more will be stored. If it doesn't already exist, it will be
            created.
        optim: torch.optim
            Optional: an optimizer object
        optim_type: torch.optim callable
            Callable optimizer. The default is Adam. Notice that this is the
            class, not the object.
        eps: float
            The value of epsilon that will be passed to our optimizer.
            We use a larger value than PyTorch's default, which empirically
            can cause exploding gradients.
        last_act: callable or None
            Last activation function to be applied outside the model.
            For example, for a binary classification problem, if we choose
            to use binary_cross_entropy_with_logits loss but want to compute
            some metric using soft predictions, we would pass in torch.sigmoid
            for `last_act`. For a multi-class problem using F.cross_entropy
            loss, we would need to pass in F.softmax to compute predicted
            probabilities.  Remember this is ONLY necessary if all of the
            following conditions are met:
            1. It is a classification problem.
            2. We have excluded the final activation from our model for
            numerical stability reasons. (I.E. the loss function has the
            the final activation built into it.)
            3. We wish to compute 1 or more metrics based on soft predictions,
            such as AUC-ROC.
        threshold: float or None
            For a classification problem, pass in the decision threshold to
            use when converting soft predictions to hard predictions. For a
            regression problem, pass in None.
        metrics: list
            A list of callable metrics. These will be computed on both the
            train and validation sets during training. To maintain
            compatibility with scikit-learn metrics, they should accept
            two arguments: y_true, followed by either y_score (for soft
            predictions) or y_pred (for hard predictions). The name and
            order of these arguments matters. If other arguments are
            required, pass in a partial with those values specified.
        callbacks: list[TorchCallback]
            List of callbacks. These will be evaluated during model training
            and can be used to track stats, adjust learning rates, clip
            gradients, etc.
        device: torch.device
            Trainer will place the model and current batch of data on this
            device during training. The default value uses a GPU if one is
            available, otherwise falls back to a CPU.

        Reference
        ---------
        Classification Loss Function (k = number of classes)

        Loss                               y shape  yhat shape  dtype
        --------------------------------------------------------------
        binary_cross_entropy_with_logits   (bs, 1)  (bs, 1)     float
        "" (multilabel case)               (bs, k)  (bs, k)     float
        cross_entropy                      (bs,)    (bs, k)     long
        """
        if last_act is None and mode != 'regression':
            warnings.warn(
                'Last activation is None for a classification problem. This '
                'means your network must include a sigmoid or softmax at the '
                'end if you wish to compute any metrics using soft '
                'predictions.'
            )

        if optim:
            optim_type = type(optim)
            warnings.warn('Inferring optim_type from optim argument.')

        self.net = net
        self.ds_train, self.ds_val = ds_train, ds_val
        self.dl_train, self.dl_val = dl_train, dl_val
        # Optim created in fit() method. Must be after net is on the GPU.
        self.optim_type = optim_type
        self.optim = optim
        self.eps = eps
        self.criterion = criterion
        self.mode = mode
        self.device = DEVICE
        self.last_act = last_act or identity
        self.thresh = threshold
        self._stop_training = False
        # For now, only print logs. During training, a file will be created.
        self.logger = self.get_logger()

        # Storage options.
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # Dict makes it easier to adjust callbacks after creating model.
        self.callbacks = {}
        self.add_callbacks(*[BasicConfig(), StatsHandler(), MetricPrinter()]
                           + (callbacks or []))
        self.metrics = [batch_size] + (metrics or [])

    def save(self, fname):
        """Save model and optimizer state dicts for later use. This
        includes the model, optimizer. Datasets and data loaders are
        excluded since:
        a. It seems that they can't be pickled in some cases (e.g. on Ubuntu).
        b. They have no learnable parameters to track during training.

        Parameters
        ----------
        fname: str
            File name to save to (not a full path - the trainer already has
            an `out_dir` attribute which will be used). The extension must
            be .pkl or .zip, and will determine whether the trainer is
            compressed.

        Returns
        -------
        None
        """
        data = {'model': self.net.state_dict()}
        try:
            data['optim'] = self.optim.state_dict()
        except AttributeError:
            self.logger.warning('No optimizer. Only saving model state dict.')
        torch.save(data, os.path.join(self.out_dir, fname))

    def load(self, fname=None, old_path=None):
        """This lets a trainer load previously saved model and optimizer
        weights. This is an in-place operation, so nothing is returned.

        Parameters
        ----------
        fname: str
            Name of file where Trainer object is stored. Must end in either
            .zip or .pkl. Do not include the full path. This automatically
            checks the output directory.
        old_path: str
            Full path to file where a previous Trainer object is stored.
            This allows us to load model and optimizer weights from a
            different round of training and store the results in a new
            directory, potentially using different hyperparameters or
            datasets.

        Returns
        -------
        None

        Examples
        --------
        trainer = Trainer(...)
        trainer.fit(...)
        trainer.save('v1')
        trainer = trainer.load('v1')
        """
        path = old_path or os.path.join(self.out_dir, fname)
        self.logger.info(f'Loading weights from {path}.')
        data = torch.load(path, map_location=self.device)
        self.net.load_state_dict(data['model'])

        # Create optimizer to load state dict. LR will be updated later.
        if not self.optim:
            self.net.to(self.device)
            self.optim = variable_lr_optimizer(self.net,
                                               optimizer=self.optim_type)
        try:
            self.optim.load_state_dict(data['optim'])
        except (AttributeError, KeyError) as e:
            self.logger.warning('Could not load optimizer. '
                                ' Loading model weights only.\n' + repr(e))

    def add_callbacks(self, *callbacks):
        """Attach additional callbacks to Trainer. Note that callback order
        will be determined by their `order` attribute, not insertion
        order.

        Parameters
        ----------
        callbacks: TorchCallback
            One or more callbacks to add.

        Returns
        -------
        None
        """
        self.callbacks.update({type(cb).__name__: cb for cb in callbacks})
        self.callbacks = dict(sorted(self.callbacks.items(),
                                     key=lambda x: x[1].order))

    def add_metrics(self, *metrics):
        """Add additional metrics to track. See the `metrics` parameter in
        the __init__ docstring for more details.

        Parameters
        ----------
        metrics: callable

        Returns
        -------
        None
        """
        self.metrics.extend(metrics)

    def set_callback_attr(self, cb_name, attr, val):
        """Convenience method to change an attribute of an existing
        callback.

        Parameters
        ----------
        cb_name: str
            Name of callback to update.
        attr: str
            Name of attribute to update.
        val: any
            Value of attribute to set.

        Returns
        -------
        None

        Examples
        --------
        # This reduces the frequency with with we update the batch stats
        # in the progress bar.
        trainer.set_callback_attr('MetricPrinter', 'batch_freq', 10)
        """
        setattr(self.callbacks[cb_name], attr, val)

    @handle_interrupt
    def fit(self, epochs, lrs=3e-3, lr_mult=1.0, **kwargs):
        """Train the model.

        Parameters
        ----------
        epochs: int
            Number of epochs to train for.
        lrs: float or Iterable(float)
            Pass in one or more learning rates. If lr_mult < 1, these
            will be the max LR(s). If the number of values matches the number
            of layer groups in the model, they will be matched accordingly,
            with the first layer is assigned the first LR. If 1 LR is passed
            in and lr_mult < 1, the multiplier will be used to create an
            appropriate number of LRs. Example: for a network with 3 groups,
            lrs=3e-3 and lr_mult=0.1 will produce LRs of [3e-5, 3e-4, 3e-3].
        lr_mult: float
            Multiplier used to compute additional learning rates if needed.
            See `update_optimizer()` for details.
        kwargs: any
            Pass in clean=True to remove existing files in out_dir.
        """
        stats = defaultdict(list)
        sum_i = 0
        _ = self.decide_stop('on_train_begin', epochs, lrs, lr_mult, **kwargs)
        for e in range(epochs):
            _ = self.decide_stop('on_epoch_begin', e, stats, None)
            for i, batch in enumerate(self.pbar):
                sum_i += 1
                *xb, yb = map(lambda x: x.to(self.device), batch)
                self.optim.zero_grad()
                _ = self.decide_stop('on_batch_begin', i, sum_i, stats)

                # Forward and backward passes.
                y_score = self.net(*xb)
                loss = self.criterion(y_score, yb)
                loss.backward()
                self.optim.step()

                # Separate because callbacks are only applied during training.
                self._update_stats(stats, loss, yb, y_score)
                if self.decide_stop('on_batch_end', i, sum_i, stats): break

            # If on_batch_end callback halts training, else block is skipped.
            else:
                val_stats = self.validate()
                if self.decide_stop('on_epoch_end', e, stats, val_stats): break
                continue
            break
        _ = self.decide_stop('on_train_end', e, stats, val_stats)

    def validate(self, dl_val=None):
        """Evaluate the model on a validation set.

        Parameters
        ----------
        dl_val: torch.utils.data.DataLoader
            Accepting an optional dataloader allows the user to pass in
            different loaders after training for evaluation. If None is
            passed in, self.dl_val is used.
        """
        dl_val = self.dl_val or dl_val
        val_stats = defaultdict(list)
        self.net.eval()
        with torch.no_grad():
            for batch in tqdm(dl_val):
                *xb, yb = map(lambda x: x.to(self.device), batch)
                y_score = self.net(*xb)
                loss = self.criterion(y_score, yb)
                self._update_stats(val_stats, loss, yb, y_score)
        return val_stats

    def predict(self, *xb, logits=True):
        """Make predictions on a batch of data. This automatically does things
        like putting the data and model on the same device, putting the model
        in eval mode, and ensuring that gradients are not computed (reduces
        time and memory usage).

        Parameters
        ----------
        xb: torch.tensors
            Inputs to the model. This will often just be one x tensor, but
            sometimes other inputs are required as well (e.g. attention masks).

        Returns
        -------
        torch.tensor: Model predictions.
        """
        xb = map(lambda x: xb.to(self.device), xb)
        self.net.to(self.device)
        self.net.eval()
        with torch.no_grad():
            res = self.net(*xb)
        if not logits: res = self.last_act(res)
        return res

    def _update_stats(self, stats, loss, yb, y_score):
        """Update stats in place.

        Parameters
        ----------
        stats: defaultdict[str, list]
        loss: torch.Tensor
            Tensor containing single value (mini-batch loss).
        yb: torch.Tensor
            Mini-batch of labels.
        y_score: torch.Tensor
            Mini-batch of raw predictions. In the case of
            classification, these may still need to be passed
            through a sigmoid or softmax.

        Returns
        -------
        None
        """
        yb, y_score = yb.detach().cpu(), y_score.detach().cpu()
        # Final activation often excluded from network architecture.
        y_score = self.last_act(y_score)

        # Convert soft predictions to hard predictions.
        if self.mode == 'binary':
            # In multi-label case, this will have shape (bs, k).
            y_pred = (y_score > self.thresh).float()
        elif self.mode == 'multiclass':
            y_pred = y_score.argmax(-1)
        elif self.mode == 'regression':
            y_pred = y_score

        stats['loss'].append(loss.detach().cpu().numpy().item())
        for m in self.metrics:
            yhat = y_pred if hasarg(m, 'y_pred') else y_score
            stats[m.__name__.replace('_score', '')].append(m(yb, yhat))

    def decide_stop(self, attr, *args, **kwargs):
        """Evaluates each of the trainer's callbacks. If any callback
        encounters a condition that signals that training should halt,
        it will set the attribute trainer._stop_training to True.
        This method returns that value. By design, all callbacks will
        be called before stopping training.

        Parameters
        ----------
        attr: str
            Determines which method to call for each callback.
            One of ('on_train_begin', 'on_train_end', 'on_batch_begin',
            'on_batch_end', 'on_epoch_begin', 'on_epoch_end').
        args, kwargs: any
            Additional arguments to pass to the callbacks.

        Returns
        -------
        bool: If True, halt training.
        """
        self._stop_training = False
        # Pass model object as first argument to callbacks.
        for cb in self.callbacks.values():
            getattr(cb, attr)(self, *args, **kwargs)
        return self._stop_training

    def unfreeze(self, n_layers=None, n_groups=None, msg_pre=''):
        """Pass in either the number of layers or number of groups to
        unfreeze. Unfreezing always starts at the end of the network and moves
        backward (e.g. n_layers=1 will unfreeze the last 1 layer, or n_groups=2
        will unfreeze the last 2 groups.) Remember than weights and biases are
        treated as separate layers.

        Parameters
        ----------
        n_layers: int or None
            Number of layers to unfreeze.
        n_groups: int or None
            Number of layer groups to unfreeze. For this to work, the model
            must define an attribute `groups` containing the layer groups.
            Each group can be a layer, a nn.Sequential object, or
            nn.Module.
        msg_pre: str
            Optional: add a prefix to the logged message. For example,
            this can be used to record the epoch that unfreezing occurred
            during.
        """
        mode = 'layers' if n_layers is not None else 'groups'
        msg_pre += f'Unfreezing last {n_layers or n_groups} {mode}.'
        self.logger.info(msg_pre)
        self.net.unfreeze(n_layers, n_groups)

    def freeze(self):
        """Freeze whole network. Mostly used for testing."""
        self.logger.info('Freezing whole network.')
        self.net.unfreeze(n_layers=0)

    def cleanup(self, sentinel=None, confirmed=False):
        """Delete output directory. An empty directory with the same name
        will be created in its place.

        Parameters
        ----------
        sentinel: None
            Placeholder to force user to pass confirmed as keyword arg.
        confirmed: bool
            Placeholder variable. This is just intended to force the user
            to confirm their desire to delete files before doing it. If
            True, the directory will be deleted. (Technically, any truthy
            value will work.)

        Returns
        -------
        None
        """
        if not confirmed:
            self.logger.info('Missing confirmation, cleanup skipped.')
            return
        self.logger.info('Removing files from output directory.')
        shutil.rmtree(self.out_dir)
        os.makedirs(self.out_dir)

    def __repr__(self):
        r = (f'Trainer(criterion={repr(self.criterion.__name__)}, '
             f'out_dir={repr(self.out_dir)})'
             f'\n\nDatasets: {len(self.ds_train)} train rows, '
             f'{len(self.ds_val)} val rows'
             f'\n\nOptimizer: {repr(self.optim)}'
             f'\n\n{repr(self.net)})')
        return r