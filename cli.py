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
from htools.cli import fire
from incendio.nlp import FillMaskTransform, GenerativeTransform, \
    ParaphraseTransform


def generate(source, dest, transforms, n=5, text_col='text', id_cols=(),
             stack_transforms=False):
    pass


if __name__ == '__main__':
    fire.Fire(generate)


