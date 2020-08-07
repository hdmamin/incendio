# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/02_callbacks.ipynb (unless otherwise specified).

__all__ = ['TorchCallback', 'BasicConfig', 'StatsHandler', 'MetricPrinter', 'BatchMetricPrinter', 'EarlyStopper',
           'PerformanceThreshold', 'ModelCheckpoint', 'MetricHistory', 'S3Uploader', 'BotoS3Uploader', 'EC2Closer',
           'ModelUnfreezer', 'SchedulerMixin', 'CosineLRScheduler', 'AdaptiveSawtoothScheduler']


# Cell
import boto3
from collections.abc import Iterable
import matplotlib.pyplot as plt
import numpy as np
from operator import lt, gt, add, sub
import os
import pandas as pd
import requests
from tabulate import tabulate
from tqdm.auto import tqdm
import warnings

try:
    from accio.s3tool import S3tool
except ImportError:
    warnings.warn('Accio not available.')
from htools import auto_repr, valuecheck, save, delegate
from .data import BotoUploader
from .optimizers import variable_lr_optimizer, update_optimizer
from .utils import DEVICE


# Cell
@auto_repr
class TorchCallback:

    def on_train_begin(self, trainer, epochs, lrs, lr_mult, **kwargs):
        pass

    def on_epoch_begin(self, trainer, epoch, val_stats):
        pass

    def on_batch_begin(self, trainer, i, sum_i):
        pass

    def after_zero_grad(self, trainer, i, sum_i, xb, yb):
        pass

    def after_forward(self, trainer, i, sum_i):
        pass

    def after_loss(self, trainer, i, sum_i):
        pass

    def after_backward(self, trainer, i, sum_i):
        pass

    def after_step(self, trainer, i, sum_i):
        pass

    def on_batch_end(self, trainer, i, sum_i):
        pass

    def on_epoch_end(self, trainer, epoch, val_stats):
        pass

    def on_train_end(self, trainer, epoch, val_stats):
        pass


# Cell
class BasicConfig(TorchCallback):
    """Handles basic model tasks like putting the model on the GPU
    and switching between train and eval modes.
    """

    def __init__(self, order=0):
        self.order = order

    def on_train_begin(self, trainer, epochs, lrs, lr_mult, **kwargs):
        trainer.net.to(DEVICE)
        if not trainer.optim:
            trainer.optim = variable_lr_optimizer(
                trainer.net, lrs, lr_mult, trainer.optim_type, trainer.eps
            )
        else:
            update_optimizer(trainer.optim, lrs, lr_mult=lr_mult)
        trainer.logger.info(trainer.optim)
        if kwargs.get('clean') is True: trainer.cleanup(confirmed=True)

    def on_epoch_begin(self, trainer, *args, **kwargs):
        trainer.net.train()

    def on_train_end(self, trainer, *args, **kwargs):
        trainer.logger.info('Training complete. Model in eval mode.')
        trainer.net.eval()


# Cell
class StatsHandler(TorchCallback):
    """This updates metrics at the end of each epoch to account for
    potentially varying batch sizes.
    """

    def __init__(self, order=5):
        self.order = order

    def on_epoch_begin(self, trainer, epoch, val_stats):
        """Resets stats at the start of each epoch."""
        trainer.stats.clear()

    def on_epoch_end(self, trainer, epoch, val_stats):
        """Computes (possibly weighted) averages of mini-batch stats
        at the end of each epoch.
        """
        for group in (trainer.stats, val_stats):
            for k, v in group.items():
                if k == 'batch_size': continue
                group[k] = np.average(v, weights=group['batch_size'])
            group.pop('batch_size')


# Cell
class MetricPrinter(TorchCallback):
    """Prints metrics at the end of each epoch. This is one of the
    default callbacks provided in BaseModel - it does not need to
    be passed in explicitly.
    """

    def __init__(self, pbar_metric='loss', batch_freq=1, order=10):
        """Order must be higher than StatsHandler, otherwise
        metrics will be printed before they're aggregated.

        Parameters
        ----------
        pbar_metric: str
            Metric to update in the tqdm progress bar. This will be
            the value computed on the most recent mini batch in the
            training set.
        batch_freq: int
            How often to update the tqdm progress bar with mini batch
            stats. The default of 1 means we'll update it every mini
            batch, which can be too fast to read if mini batches
            are processed quickly (e.g. if batch size is small and the
            forward pass is fast).
        order: int
        """
        self.pbar_metric = pbar_metric
        self.order = order
        self.batch_freq = batch_freq

    def on_train_begin(self, trainer, *args, **kwargs):
        trainer.logger = trainer.get_logger(
            os.path.join(trainer.out_dir, 'train.log'),
            fmt='\n%(asctime)s\n %(message)s'
        )

    def on_epoch_begin(self, trainer, epoch, val_stats):
        """Create progress bar."""
        trainer.pbar = tqdm(trainer.dl_train)

    def on_epoch_end(self, trainer, epoch, val_stats):
        """Print stats and close progress bar."""
        data = [[k, v, val_stats[k]] for k, v in trainer.stats.items()]
        table = tabulate(data, headers=['Metric', 'Train', 'Validation'],
                         tablefmt='github', floatfmt='.4f')
        trainer.logger.info(
            f'\n{"="*5}\n\nEpoch {epoch}\n\n{table}\n\n{"="*5}'
        )
        trainer.pbar.close()

    def on_batch_end(self, trainer, i, sum_i):
        """Update progress bar with batch stats."""
        if sum_i % self.batch_freq != 0:
            return
        kwargs = {self.pbar_metric:
                  format(trainer.stats[self.pbar_metric][-1], '.4f')}
        trainer.pbar.set_postfix(**kwargs)


# Cell
class BatchMetricPrinter(TorchCallback):
    """Prints mini batch metrics to help us see if a model is
    learning early in training (helpful for debugging). We
    remove the callback after the specified number of prints
    so that it isn't called unnecessarily throughout the whole
    training process.
    """

    def __init__(self, batch_freq, n_prints=float('inf'), order=10):
        """Order must be higher than StatsHandler, otherwise
        metrics will be printed before they're aggregated.
        """
        self.order = order
        self.batch_freq = batch_freq
        self.n_prints = n_prints
        self.curr_prints = 0

    def on_batch_end(self, trainer, i, sum_i):
        if sum_i % batch_freq:
            self.curr_prints += 1
            metric_str = "\n".join(
                f'{k}={round(v[-1], 4)}' for k, v in trainer.stats.items()
            )
            trainer.logger.info(f'Batch {sum_i}\n: {metric_str}')
        if self.curr_prints >= self.n_prints:
            trainer.callbacks.pop(type(self).__name__)


# Cell
class EarlyStopper(TorchCallback):

    @valuecheck
    def __init__(self, metric, goal:('max', 'min'), min_improvement=0.0,
                 patience=3, order=15):
        """
        Parameters
        ----------
        metric: str
            Quantity to monitor. This will always be computed on the
            validation set.
        goal: str
            Indicates what we want to do to the metric in question.
            Either 'min' or 'max'. E.g. metric 'loss' should have goal 'min'
            while metric 'precision' should have goal 'max'.
        min_improvement: float
            Amount of change needed to qualify as improvement. For example,
            min_improvement of 0.0 means any improvement is sufficient. With
            a min_improvent of 0.2, we will stop training even if the
            quantity improves by, for example, 0.1.
        patience: int
            Number of acceptable epochs without improvement. E.g. patience=0
            means the metric must improve every epoch for training to continue.
        """
        # Will use op like: self.op(new_val, current_best)
        if goal == 'min':
            self.init_metric = self.best_metric = float('inf')
            self.op = lt
            self.op_best = sub
        elif goal == 'max':
            self.init_metric = self.best_metric = float('-inf')
            self.op = gt
            self.op_best = add

        self.order = order
        self.metric = metric
        self.min_improvement = min_improvement
        self.patience = patience
        self.since_improvement = 0

    def on_train_begin(self, trainer, *args, **kwargs):
        """Resets tracked variables at start of training."""
        self.best_metric = self.init_metric
        self.since_improvement = 0

    def on_epoch_end(self, trainer, epoch, val_stats):
        # Error handling.
        new_val = val_stats.get(self.metric)
        if new_val is None:
            trainer.logger.info(f'EarlyStopper could not find {self.metric}. '
                                f'Callback behavior may not be enforced.')
            return

        # Expected behavior.
        if self.op(new_val, self.op_best(self.best_metric, self.min_improvement)):
            self.best_metric = new_val
            self.since_improvement = 0
        else:
            self.since_improvement += 1
            if self.since_improvement > self.patience:
                trainer.logger.info(
                    f'EarlyStopper halting training: validation {self.metric} '
                    f'has not improved enough in {self.since_improvement} epochs.'
                )
                trainer._stop_training = True


# Cell
class PerformanceThreshold(TorchCallback):

    @valuecheck
    def __init__(self, metric, goal:('min', 'max'), threshold, skip_epochs=0,
                 split:('train', 'val')='val', order=15):
        self.order = order
        self.metric = metric
        self.threshold = threshold
        self.skip_epochs = skip_epochs
        self.split = split
        self.op = gt if goal == 'min' else lt

    def on_epoch_end(self, trainer, epoch, val_stats):
        if epoch < self.skip_epochs:
            return

        # Error handling.
        data = val_stats if self.split == 'val' else trainer.stats
        new_val = data.get(self.metric)
        if new_val is None:
            trainer.logger.info(f'{self.metric.title()} not found in metrics. '
                                 'PerformanceThreshold may not be enforced.')
            return

        # Expected behavior.
        if self.op(new_val, self.threshold):
            trainer.logger.info(
                f'PerformanceThreshold halting training: {self.metric} '
                f'of {new_val:.4f} did not meet threshold.'
            )
            trainer._stop_training = True


# Cell
class ModelCheckpoint(TorchCallback):

    @valuecheck
    def __init__(self, metric='loss', goal:('max', 'min')='min', order=25):
        # Will use op like: self.op(new_val, current_best)
        if goal == 'min':
            self.init_metric = self.best_metric = float('inf')
            self.op = lt
            self.op_best = sub
        elif goal == 'max':
            self.init_metric = self.best_metric = float('-inf')
            self.op = gt
            self.op_best = add

        self.order = order
        self.metric = metric
        self.metric_path = None

    def on_train_begin(self, trainer, *args, **kwargs):
        self.best_metric = self.init_metric
        self.metric_path = os.path.join(trainer.out_dir,
                                        'best_val_metrics.json')

    def on_epoch_end(self, trainer, epoch, val_stats):
        new_val = val_stats.get(self.metric)
        # Error handling.
        if new_val is None:
            trainer.logger.info(f'{self.metric} not found in metrics.'
                                 'ModelCheckpoint may not save models.')
            return

        # Expected behavior.
        if self.op(new_val, self.best_metric):
            trainer.logger.info(
                f'Saving model. {self.metric.title()} improved from '
                f'{self.best_metric:.4f} to {new_val:.4f}.'
            )
            trainer.save(f'trainer.pkl')
            save({k: round(v, 5) for k, v in val_stats.items()},
                 self.metric_path)
            self.best_metric = new_val


# Cell
class MetricHistory(TorchCallback):
    """Separate from StatsHandler in case we don't want to log outputs."""

    def __init__(self, fname='history.csv', plot_fname='history.png',
                 order=90):
        self.train_hist = []
        self.val_hist = []
        self.fname = fname
        self.plot_fname = plot_fname
        self.order = order

    def on_train_begin(self, trainer, *args, **kwargs):
        self.train_hist.clear()
        self.val_hist.clear()

    def on_epoch_end(self, trainer, epoch, val_stats):
        self.train_hist.append(trainer.stats.copy())
        self.val_hist.append(val_stats.copy())

    def on_train_end(self, trainer, epoch, val_stats):
        self.df = pd.concat([
            pd.DataFrame(self.train_hist),
            pd.DataFrame(self.val_hist)\
              .rename(lambda x: f'val_{x}', axis='columns')
        ], axis=1)
        self.df.round(5).to_csv(
            os.path.join(trainer.out_dir, self.fname), index=False
        )
        self.plot(os.path.join(trainer.out_dir, self.plot_fname)
                  if self.plot_fname else None)

    def plot(self, path=None):
        df_cols = self.df.shape[1]
        n_rows = max(1, df_cols // 4)
        n_cols = df_cols // (n_rows * 2)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
        if isinstance(ax, Iterable):
            ax = ax.flatten()
        else:
            # If no extra metrics are specified, we only have 1 plot with loss.
            ax = [ax]
        for i, axi in zip(range(0, n_cols, 2), ax):
            col = self.df.columns[i]
            axi.plot(self.df[col], label='train')
            axi.plot(self.df[f'val_{col}'], label='val')
            axi.set_title(col.title())
            axi.set_xlabel('Epoch')
            axi.set_ylabel('Score')
            axi.legend()
        plt.tight_layout()
        if path:
            plt.savefig(path)
        else:
            plt.show()


# Cell
class S3Uploader(TorchCallback):
    """Upload model and logs to S3 when training finishes."""

    def __init__(self, bucket, prefix, order=95):
        """See Accio documentation for parameter details."""
        self.bucket = bucket
        self.prefix = prefix
        self.order = order

    def on_train_end(self, trainer, *args, **kwargs):
        """Upload files to s3 once training completes."""
        paths = [f.path for f in os.scandir(trainer.out_dir)
                 if f.is_file() and not f.name.startswith('.')]
        s3 = S3tool()
        try:
            s3.upload_files(paths, self.bucket, self.prefix)
        except Exception as e:
            trainer.logger.error(e)


# Cell
class BotoS3Uploader(TorchCallback):
    """Upload model and logs to S3 when training finishes. This version of the
    callback does not rely on any GoGuardian packages.
    """

    def __init__(self, bucket, s3_dir='', retain_tree=True, recurse=True,
                 keep_fn=None, order=95):
        """
        Parameters
        ----------
        bucket: str
            Name of s3 bucket to upload to. For a single project, I generally
            stick to a single bucket so we can usually keep this fixed. We can
            always change the attribute later if necessary.
        s3_dir: str
            If provided, this will pre prepended to each path: {s3_dir}/{path}.
            Otherwise, s3 paths will be the same as local paths.
        retain_tree: bool
            If True, the local file structure will be retained. Otherwise,
            only the base name is kept. All four combinations of retain_tree
            (True/False) and s3_dir (empty/non-empty) are supported.
        recurse: bool
            If True, upload all files in subdirectories as well.
        keep_fn: None or callable
            If provided, this should be a function that accepts a filename as
            input and returns a boolean specifying whether to include it in the
            upload or not.
        """
        self.bucket = bucket
        self.s3_dir = s3_dir
        self.retain_tree = retain_tree
        self.recurse = recurse
        self.keep_fn = keep_fn
        self.order = order

    def on_train_end(self, trainer, *args, **kwargs):
        """Upload files to s3 once training completes."""
        paths = [f.path for f in os.scandir(trainer.out_dir)
                 if f.is_file() and not f.name.startswith('.')]
        s3 = BotoUploader(self.bucket, self.verbose)
        try:
            self.s3.upload_folder(trainer.out_dir, self.s3_dir,
                                  self.retain_tree, self.recurse, self.keep_fn)
        except Exception as e:
            trainer.logger.error(e)


# Cell
class EC2Closer(TorchCallback):

    def __init__(self, timeout=5, order=100):
        self.timeout = timeout
        self.order = order

    def on_train_end(self, trainer, *args, **kwargs):
        try:
            r = requests.get(url, timeout=self.timeout).json()
        except requests.ReadTimeout as e:
            trainer.logger.info('Request timed out. Failed to '
                                'shutdown instance.')
            return

        id_, region = r['instanceId'], r['region']
        ec2 = boto3.client('ec2', region_name=region)
        ec2.stop_instances(InstanceIds=[id_], DryRun=debug)


# Cell
class ModelUnfreezer(TorchCallback):
    """Gradually unfreeze a model during training.
    """

    @valuecheck
    def __init__(self, i2n, unfreeze_type:('groups', 'layers')='groups',
                 mode:('batch', 'epoch')='epoch', order=25):
        """
        Parameters
        ----------
        i2n: dict
            Maps index of batch/epoch to the number of layers or groups
            to unfreeze at that point in time. Batches and epochs are
            both zero-indexed. Note that batch refers to the global
            batch number (e.g. if there are 100 batches per epoch, the
            first batch of the second epoch is batch #101.)
        unfreeze_type: str
            Specifies whether to unfreeze groups or layers.
        mode: str
            Specifies whether the indices in `i2n` refer to batches or
            epochs.
        order: int
            Determine place in the callback queue. Smaller numbers are
            executed earlier.

        Examples
        --------
        This will create a callback that unfreezes the last 2 layer
        groups at epoch 2, the last 3 groups at epoch 10, and the
        last 4 groups at epoch 25.

        ModelUnfreezer(
            i2n={2: 2, 10: 3, 25: 4},
            unfreeze_type='groups',
            mode='epoch'
        )
        """
        self.order = order
        self.i2kwargs = {i: {f'n_{unfreeze_type}': n}
                         for i, n in i2n.items()}
        self.mode = mode

    def on_batch_begin(self, trainer, i, sum_i):
        if self.mode != 'batch':
            return
        kwargs = self.i2kwargs.get(sum_i, None)
        if kwargs:
            trainer.unfreeze(**kwargs, msg_pre=f'Global batch {sum_i}: ')

    def on_epoch_begin(self, trainer, epoch, val_stats):
        if self.mode != 'epoch':
            return
        kwargs = self.i2kwargs.get(epoch, None)
        if kwargs:
            trainer.unfreeze(**kwargs, msg_pre=f'Epoch {epoch}: ')


# Cell
class SchedulerMixin(TorchCallback):

    verbose = False

    def on_train_end(self, trainer, *args, **kwargs):
        self.plot_lrs(os.path.join(trainer.out_dir, 'lrs.png'))

    def update_lr(self, trainer, n):
        try:
            lr = self.lrs[n]
        except IndexError as e:
            return

        update_optimizer(trainer.optim, lr, lr_mult=self.lr_mult)
        if self.verbose:
            trainer.logger.info(f'Set learning rate to {lr:.4f}.')

    def plot_lrs(self, path=None):
        """Display learning rate by iteration.

        Note: If the plot is not as smooth as expected, this likely
        means that there are very few iterations per epoch
        (i.e. the batch size is very large, at least in relative terms).
        """
        plt.plot(self.lrs)
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        if path:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()


# Cell
class CosineLRScheduler(SchedulerMixin):
    """Learning rate scheduler that makes updates each batch.
    """

    def __init__(self, warm=0.3, restarts=False, cycle_len=5, cycle_decay=0.0,
                 min_lr=None, verbose=False, order=10):
        """
        Parameters
        ----------
        warm: float
            Percent of training run (or cycle length) devoted to the increasing
            portion of the schedule. Default 0.3.
        restarts: bool
            Specifies whether to use restarts, i.e. use a cyclical LR.
            True: Version of cosine annealing with restarts. In one
                  cycle, LR starts high and gradually decreases.
                  At the start of the next cycle, it is
                  immediately increased again.
            False: Version of cosine annealing where LR increases
                   for first 30% of training, then decreases for
                   remaining 70%.
        cycle_len: int
            Number of epochs contained in a single cycle. Only used
            when scheduler uses restarts.
        cycle_decay: float
            Scalar to decay the learning rate at the end of each cycle.
            This is only used with restarts, since the regular cosine
            annealing already decays the LR over time.
            E.g. 1.0 will use no decay.
            0.9 means that cycle 2 LRs = cycle 1 LRs * 0.9,
            cycle 3 LRs = cycle 1 LRs * .81,
            etc.
        min_lr: float
            Minimum learning rate. If None is specified, it will be set
            to max_lr / 10.
        """
        super().__init__()
        self.warm = warm
        self.cycle_len = cycle_len
        self.cycle_decay = cycle_decay
        self.restarts = restarts
        self.verbose = verbose
        self.min_lr = min_lr
        self.order = order

        # Set in `on_train_begin()`.
        self.lrs = None             # Iterable[float]
        self.batches_per_e = None   # int
        self.batches = None         # int
        self.max_lr = None          # float
        self.lr_mult = None         # float

    def on_train_begin(self, trainer, epochs, lrs, lr_mult, **kwargs):
        """Wrapper to schedule learning rates depending on chosen method.

        Parameters
        ----------
        restarts: bool
            If True, use schedule with restarts. If False, use regular
            cosine annealing that spans whole duration of training.

        Returns
        -------
        np.array: LR for each iteration (i.e. output[i] is the LR to use
            at iteration i).
        """
        self.batches_per_e = len(trainer.dl_train)
        self.batches = epochs * self.batches_per_e
        self.max_lr = max(lrs) if isinstance(lrs, Iterable) else lrs
        self.lr_mult = lr_mult
        if not self.min_lr: self.min_lr = self.max_lr / 10

        if self.restarts and self.batches < self.cycle_len:
            warnings.warn('Training will be less than 1 full cycle.')

        if self.restarts:
            self.lrs = self._cosine_restarts_schedule()
        else:
            self.lrs = self._cosine_schedule()

    def on_batch_begin(self, trainer, i, sum_i):
        self.update_lr(trainer, sum_i)

    @staticmethod
    def _cosine_anneal(batches, lr1, lr2):
        """Helper function for _cosine_schedule().

        Parameters
        ----------
        batches: int
            Number of batches in segment.
        lr1: float
            Learning rate at start of segment.
        lr2: float
            Learning rate at end of segment.

        Returns
        -------
        np.array
        """
        i = np.arange(batches)
        return lr2 + (lr1 - lr2)*(1 + np.cos(np.pi * i/batches))/2

    def _cosine_schedule(self):
        """Cosine annealing scheduler. Computes learning rates for each
        iteration.

        Returns
        -------
        np.array
        """
        seg1 = self._cosine_anneal(int(self.warm * self.batches),
                                   self.min_lr, self.max_lr)
        seg2 = self._cosine_anneal(int(np.ceil((1 - self.warm) * self.batches)),
                                   self.max_lr, self.min_lr)
        return np.concatenate((seg1, seg2))

    def _cosine_restarts_schedule(self):
        """Cosine annealing with restarts."""
        cycles = int(np.ceil(self.batches / (self.cycle_len * self.batches_per_e)))
        cycle_batches = self.cycle_len * self.batches_per_e
        lrs = [self._cosine_anneal(cycle_batches, self.max_lr, self.min_lr)
               / (1 + self.cycle_decay * i) for i in range(cycles)]
        return np.concatenate(lrs)


# Cell
class AdaptiveSawtoothScheduler(SchedulerMixin):
    """Learning rate scheduler inspired by the sawtooth pattern often
    used to manage TCP flow
    (ex: https://witestlab.poly.edu/blog/tcp-congestion-control-basics/).
    This uses a strategy called "additive increase, multiplicative decrease".
    Basically, while the training loss is generally decreasing, we
    gradually increase the learning rate. When things show signs of getting
    worse, we dramatically decrease the LR and begin slowly climbing again.
    The result looks something like a cyclical policy with restarts,
    except that in this case the cycle lengths are dependent on training
    rather than pre-defined. SGD w/ restarts typically also uses a sharp
    increase and a gradual decrease, while this is closer to the opposite.

    Unlike the standard AIMD algorithm, we decay the amount added if the
    batch loss increases, even if we're still within the patience window.
    """

    def __init__(self, add=1e-4, scale=0.6, patience=5, order=10):
        """Note: further experimentation is required to determine
        sensible defaults for these hyperparameters.

        Parameters
        ----------

        """
        self.add = add
        self.scale = scale
        self.patience = patience
        self.order = order

        # These are reset in `on_train_begin`, but types remain the same.
        self.lrs = []
        self.since_improve = 0
        self.recent_best = float('inf')
        self.lr_mult = 1.0

    def on_train_begin(self, trainer, epochs, lrs, lr_mult, **kwargs):
        """Wrapper to schedule learning rates depending on chosen method.

        Parameters
        ----------
        restarts: bool
            If True, use schedule with restarts. If False, use regular
            cosine annealing that spans whole duration of training.

        Returns
        -------
        np.array: LR for each iteration (i.e. output[i] is the LR to use
            at iteration i).
        """
        self.lrs.clear()
        self.since_improve = 0
        self.recent_best = float('inf')
        self.lr_mult = lr_mult

    def on_batch_begin(self, trainer, i, sum_i):
        """Update LR at the start of every batch."""
        try:
            loss = trainer.stats.get('loss')[-1]
        except:
            return

        lr = max(p['lr'] for p in trainer.optim.param_groups)
        if loss <= self.recent_best:
            self.recent_best = loss
            self.since_improve = 0
            lr += self.add
        elif loss > self.recent_best and self.since_improve < self.patience:
            self.since_improve += 1
            lr += self.add / (self.since_improve+1)
        else:
            self.since_improve += 1
            lr *= self.scale
        update_optimizer(trainer.optim, lr, lr_mult=self.lr_mult)
        self.lrs.append(lr)