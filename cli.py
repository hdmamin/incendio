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
from incendio.nlp import FillMaskTransform, GenerativeTransform, \
    ParaphraseTransform


TRANSFORMS = {
    'fillmask': FillMaskTransform,
    'paraphrase': ParaphraseTransform,
    'generative': GenerativeTransform
}


def generate(source, dest, transforms, n=5, text_col='text', id_cols=(),
             stack_transforms=False, nrows=None):
    # Note: use list args like --id_cols '[col1, col2]'
    source, dest = Path(source), Path(dest)

    # TODO: starting with simple case of 1 transform. Worry about adding
    # flexibility later.
    # tfms = [TRANSFORMS[t] for t in tolist(transforms)]
    transform = TRANSFORMS[transforms](n=n)
    df = pd.read_csv(source, usecols=[text_col] + list(id_cols))
    res = transform(df[text_col].tolist())
    print(res)
    # df_res = transform(df[text_col].tolist())
    # df_res.ends().pprint()
    # df_res.to_csv(dest, index=False)


if __name__ == '__main__':
    fire.Fire()


