# coding: utf-8
"""
Collection of helper functions for audio processing
"""

import os
import os.path
import io

from collections import defaultdict
import numpy as np


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
    def _is_npy_data(self, data: bytes) -> bool:
        return data[0] == 147 and data[1] == 78

    # from fairseq
    def _get_features_from_zip(self, path, byte_offset, byte_size):
        assert path.endswith(".zip")
        with open(path, "rb") as f:
            f.seek(byte_offset)
            data = f.read(byte_size)
        byte_features = io.BytesIO(data)
        if self._is_npy_data(data):
            features = np.load(byte_features)
        else:
            raise ValueError(f'Unknown file format for "{path}"')
        return features

    # from fairseq
    def get_features(self):
        """Get speech features from ZIP file
           accessed via byte offset and length

        :return: (np.ndarray) speech features in shape of (num_frames, num_freq)
        """
        _path, *extra = self.path.split(":")
        if not os.path.exists(_path):
            raise FileNotFoundError(f"File not found: {_path}")

        if len(extra) == 0:
            ext = os.path.splitext(os.path.basename(_path))[1]
            if ext == ".npy":
                features = np.load(_path)
            else:
                raise ValueError(f"Invalid file type: {_path}")
        elif len(extra) == 2:
            extra = [int(i) for i in extra]
            features = self._get_features_from_zip(
                _path, extra[0], extra[1]
            )
        else:
            raise ValueError(f"Invalid path: {self.path}")
        assert abs(features.shape[0] - self.n_frames) <= 1, \
            (features.shape[0] - self.n_frames, self.id)
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
        f = b.get_features()
        length = min(f.shape[0], b.n_frames)
        lengths[i] = length
        features[i, :length, :] += f[:length, :]

    # validation
    assert len(lengths) == features.shape[0]
    assert lengths.max() == features.shape[1]
    assert embed_size == features.shape[2]

    return features, lengths
