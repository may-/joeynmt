# coding: utf-8
"""
Collection of helper functions for audio processing
"""

from pathlib import Path
import io
import logging

from collections import defaultdict
import numpy as np
from textgrid import TextGrid

from joeynmt.constants import UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN


logger = logging.getLogger(__name__)


class SpeechInstance:
    def __init__(self, path, n_frames, id):
        """Speech Instance

        :param path: (str) Feature file path in the format of
            "<zip path>:<byte offset>:<byte length>".
        :param n_frames: (int) number of frames
        """
        self.path = path
        self.n_frames = n_frames
        self.id = id

    def __len__(self):
        return self.n_frames

# from fairseq
def _is_npy_data(data: bytes) -> bool:
    return data[0] == 147 and data[1] == 78

# from fairseq
def _get_features_from_zip(path: Path, byte_offset: int, byte_size: int):
    with path.open("rb") as f:
        f.seek(byte_offset)
        data = f.read(byte_size)
    byte_features = io.BytesIO(data)
    if _is_npy_data(data):
        features = np.load(byte_features)
    else:
        raise ValueError(f'Unknown file format for "{path}"')
    return features

# from fairseq
def get_features(path: str):
    """Get speech features from ZIP file
       accessed via byte offset and length

    :return: (np.ndarray) speech features in shape of (num_frames, num_freq)
    """
    _path, *extra = path.split(":")
    _path = Path(_path)
    if not _path.is_file():
        raise FileNotFoundError(f"File not found: {_path}")

    if len(extra) == 0 and _path.suffix == ".npy":
        features = np.load(_path)
    elif len(extra) == 2 and _path.suffix == ".zip":
        extra = [int(i) for i in extra]
        features = _get_features_from_zip(_path, extra[0], extra[1])
    else:
        raise ValueError(f"Invalid path: {path}")
    return features


def pad_features(batch, embed_size=80, max_len=None):
    """
    Pad continuous feature representation in batch.
    called in batch construction (not in data loading)

    :param batch: SpeechInstance
    :param embed_size: (int) number of frequencies
    :param max_len: (int) max input length in the given batch
    :returns:
      - features np.ndarray, (batch_size, src_len, embed_size)
      - lengths np.ndarray, (batch_size)
    """
    if max_len is None:
        max_len = max([len(b) for b in batch])
    batch_size = len(batch)

    # encoder input has shape of (batch_size, src_len, embed_size)
    # (see encoder.forward())
    features = np.zeros((batch_size, max_len, embed_size))
    lengths = np.zeros(batch_size, dtype=int)

    for i, b in enumerate(batch):
        # look up zip file
        f = get_features(b.path)
        assert abs(f.shape[0] - b.n_frames) <= 1, (f.shape[0] - b.n_frames, b.id)
        length = min(f.shape[0], b.n_frames)
        lengths[i] = length
        features[i, :length, :] += f[:length, :]

    # validation
    assert len(lengths) == features.shape[0]
    assert lengths.max() == features.shape[1]
    assert embed_size == features.shape[2]

    return features, lengths


def get_textgrid(textgrid_path, trg, frame_shift=100, return_phn=False):
    if not textgrid_path.is_file():
        raise FileNotFoundError(f"TextGrid not found: {textgrid_path}")
    textgrid = TextGrid.fromFile(textgrid_path)

    alignments = defaultdict(list)
    for tg in textgrid:
        for i in tg:
            alignments[tg.name].append([int(float(i.minTime) * frame_shift),
                                        int(float(i.maxTime) * frame_shift), i.mark])

    assert set(alignments.keys()) == set(['phones', 'words']), \
        f"TextGrid {textgrid_path} doesn't contain alignments information."

    # append original surface form (because montreal forces aligner outputs <unk> for OOV)
    words = [(j, a[2]) for j, a in enumerate(alignments['words']) if len(a[2]) > 0]
    if len(words) == len(trg.split()):
        for (j, mfa), orig in zip(words, trg.split()):
            assert (len(orig) > 0 and
                    (mfa == orig or mfa == '<unk>' or mfa == orig.strip("'"))), (j, mfa, orig)
            alignments['words'][j].append(orig)

    ####
    # alignments = {
    # 'words': [[0, 8, ''],                         # empty surface for silent token
    #           [8, 30, 'dey', 'dey'],
    #           [30, 42, 'all', 'all'],
    #           [42, 68, 'made', 'made'],
    #           [68, 76, 'der', 'der'],
    #           [76, 137, '<unk>', "bre'kfus"],     # original surface for <unk>
    #           [137, 167, 'offen', 'offen'],
    #           [167, 200, '<unk>', "roas'"],       # original surface for <unk>
    #           [200, 209, 'in', "in'"],            # apostrophe can be removed in montreal forced aligner, so put it back here
    #           [209, 247, 'years', 'years'],
    # ...
    ###
    assert " ".join([a[-1] for a in alignments['words']
                     if len(a[-1]) > 0]) == trg, (alignments['words'], trg)

    if not return_phn:
        return alignments['words']
    else:
        # word to phoneme alignments
        dic = defaultdict(list)
        i = 0
        for j, wrd in enumerate(alignments['words']):
            for phn in alignments['phones'][i:]:
                if wrd[0] <= phn[0] and phn[1] <= wrd[1]:
                    dic[j].append(i)
                    i += 1
                else:
                    break
        wrd2phn = {}
        for k, (j, _) in enumerate(words):
            wrd2phn[k] = ' '.join([alignments['phones'][phn][-1] for phn in dic[j]])

        return alignments, wrd2phn
