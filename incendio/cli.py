"""Command line interface to augment text datasets. This is probably preferable
to computing augmented versions on the fly during training (as we often do in
computer vision applications) since these ML-based augmentations can be
memory-intensive.

Examples
--------
incendio augment data/raw.csv --dest data/augmented.csv \
    --transforms mask \
    --n 5 \
    --text_col text

incendio augment data/raw_files --dest data/augmented_files \
    --transforms [mask, paraphrase] \
    --stack_transforms True \
    --id_cols [user_id, data_source, label] \
"""
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

from htools.cli import fire
from htools.core import tolist, save, load
from htools.meta import immutify_defaults
from incendio.nlp import FillMaskTransform, GenerativeTransform, \
    ParaphraseTransform


TRANSFORMS = {
    'fillmask': FillMaskTransform,
    'paraphrase': ParaphraseTransform,
    'generative': GenerativeTransform
}


@immutify_defaults
def generate(source, dest, transform, n=5, text_col='text', id_cols=(),
             nrows=None, tfm_kwargs={}, call_kwargs={}):
    # Use list args like --id_cols '[col1, col2]'
    # Use dict args like --call_kwargs '{"drop_pct": .2, "min_keep": 4}'
    source, dest = Path(source), Path(dest)

    # For simplicity, we stick to one transform at a time.
    transform = TRANSFORMS[transform](n=n, **tfm_kwargs)
    df = pd.read_csv(source, usecols=[text_col] + list(id_cols), nrows=nrows)
    res = transform(df[text_col].tolist(), **call_kwargs)
    print(res)
    # df_res = transform(df[text_col].tolist())
    # df_res.ends().pprint()
    # df_res.to_csv(dest, index=False)


if __name__ == '__main__':
    fire.Fire()


