# coding: utf-8
"""
Data module
"""
import sys
from pathlib import Path
from itertools import zip_longest
from typing import Optional, List, Tuple, Callable, Union
import logging

import numpy as np

import torch
from torch.utils.data import Dataset, Sampler, SequentialSampler, \
    RandomSampler, BatchSampler, DataLoader

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.helpers import log_data_info
from joeynmt.vocabulary import build_vocab, Vocabulary
from joeynmt.batch import Batch

logger = logging.getLogger(__name__)


def load_data(data_cfg: dict, datasets: list = None)\
        -> (Dataset, Dataset, Optional[Dataset], Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :param datasets: list of dataset names to load
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
    """
    if datasets is None:
        datasets = ["train", "dev", "test"]

    # load data from files
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    train_path = data_cfg.get("train", None)
    dev_path = data_cfg.get("dev", None)
    test_path = data_cfg.get("test", None)

    if train_path is None and dev_path is None and test_path is None:
        raise ValueError('Please specify at least one data source path.')

    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_len = data_cfg["max_sent_length"]

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    train_data, train_src, train_trg = None, None, None
    if "train" in datasets and train_path is not None:
        logger.info("Loading training data...")
        train_src, train_trg = _read_data_file(
            Path(train_path), (src_lang, trg_lang), tok_fun, lowercase, max_len)

        random_train_subset = data_cfg.get("random_train_subset", -1)
        if random_train_subset > -1:
            # select this many training examples randomly and discard the rest
            keep_index = np.random.permutation(np.arange(len(train_src)))
            keep_index = keep_index[:random_train_subset]
            keep_index.sort()
            train_src = [train_src[i] for i in keep_index]
            train_trg = [train_trg[i] for i in keep_index]

        train_data = TranslationDataset(train_src, train_trg)

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    src_vocab_file = data_cfg.get("src_vocab", None)
    trg_vocab_file = data_cfg.get("trg_vocab", None)
    src_vocab_file = None if src_vocab_file is None else Path(src_vocab_file)
    trg_vocab_file = None if trg_vocab_file is None else Path(trg_vocab_file)

    assert (train_src is not None) or (src_vocab_file is not None)
    assert (train_trg is not None) or (trg_vocab_file is not None)

    logger.info("Building vocabulary...")
    src_vocab = build_vocab(min_freq=src_min_freq, max_size=src_max_size,
                            tokens=train_src, vocab_file=src_vocab_file)
    trg_vocab = build_vocab(min_freq=trg_min_freq, max_size=trg_max_size,
                            tokens=train_trg, vocab_file=trg_vocab_file)
    assert src_vocab.stoi[UNK_TOKEN] == trg_vocab.stoi[UNK_TOKEN]
    assert src_vocab.stoi[PAD_TOKEN] == trg_vocab.stoi[PAD_TOKEN]
    assert src_vocab.stoi[BOS_TOKEN] == trg_vocab.stoi[BOS_TOKEN]
    assert src_vocab.stoi[EOS_TOKEN] == trg_vocab.stoi[EOS_TOKEN]

    dev_data = None
    if "dev" in datasets and dev_path is not None:
        logger.info("Loading dev data...")
        dev_src, dev_trg = _read_data_file(
            Path(dev_path), (src_lang, trg_lang), tok_fun, lowercase)
        dev_data = TranslationDataset(dev_src, dev_trg)

    test_data = None
    if "test" in datasets and test_path is not None:
        logger.info("Loading test data...")
        # check if target exists
        if not Path(test_path).with_suffix(f'.{trg_lang}').is_file():
            # no target is given -> create dataset from src only
            trg_lang = None
        test_src, test_trg = _read_data_file(
            Path(test_path), (src_lang, trg_lang), tok_fun, lowercase)
        test_data = TranslationDataset(test_src, test_trg)

    logger.info("Data loaded.")
    log_data_info(train_data, dev_data, test_data, src_vocab, trg_vocab)
    return train_data, dev_data, test_data, src_vocab, trg_vocab


def make_data_iter(dataset: Dataset,
                   src_vocab: Vocabulary,
                   trg_vocab: Vocabulary,
                   batch_size: int,
                   batch_type: str = "sentence",
                   batch_class: Batch = Batch,
                   seed: int = 42,
                   shuffle: bool = False,
                   num_workers: int = 0) -> DataLoader:
    """
    Returns a torch DataLoader for a torch Dataset. (no bucketing)

    :param dataset: torch dataset containing src and optionally trg
    :param src_vocab:
    :param trg_vocab:
    :param batch_class: joeynmt batch class
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param seed: random seed for shuffling
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :param num_workers: number of cpus for multiprocessing
    :return: torch DataLoader
    """
    # sampler
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)

    # batch generator
    if batch_type == "sentence":
        batch_sampler = BatchSampler(sampler,
                                     batch_size=batch_size,
                                     drop_last=False)
    elif batch_type == "token":
        batch_sampler = TokenBatchSampler(sampler,
                                          batch_size=batch_size,
                                          drop_last=False)

    pad_index = src_vocab.stoi[PAD_TOKEN]

    def collate_fn(batch) -> Batch:
        src_list, trg_list = zip(*batch)
        src, src_length = src_vocab.sentences_to_ids(src_list)
        trg, trg_length = None, None
        if any(map(None.__ne__, trg_list)):
            trg, trg_length = trg_vocab.sentences_to_ids(trg_list)

        return batch_class(torch.LongTensor(src),
                           torch.LongTensor(src_length),
                           torch.LongTensor(trg) if trg else None,
                           torch.LongTensor(trg_length) if trg_length else None,
                           pad_index)

    data_iter = DataLoader(dataset, batch_sampler=batch_sampler,
                           collate_fn=collate_fn, num_workers=num_workers)

    return data_iter


def _read_data_file(path: Path, exts: Tuple[str, Union[str, None]],
                    tokenize: Callable, lowercase: bool, max_len: int = -1) \
        -> Tuple[List[List[str]], List[List[str]]]:
    """
    Read data files
    :param path: data file path
    :param exts: pair of file extensions
    :param tokenize: tokenize function
    :param lowercase: whether to lowercase or not
    :param max_len: maximum length (longer instances will be filtered out)
    :return: pair of tokenized sentence lists
    """
    s_lang, t_lang = exts
    src_doc = path.with_suffix(f'.{s_lang}').read_text().strip().split('\n')
    trg_doc = []
    if t_lang:
        trg_doc = path.with_suffix(f'.{t_lang}').read_text().strip().split('\n')
        assert len(src_doc) == len(trg_doc)

    src_tok, trg_tok = [], []
    invalid = 0
    for i, (s, t) in enumerate(zip_longest(src_doc, trg_doc)):
        src = tokenize(s.strip().lower() if lowercase else s.strip())
        if t: # parallel dataset
            trg = tokenize(t.strip().lower() if lowercase else t.strip())
            if (max_len < 0) or (len(src) <= max_len and len(trg) <= max_len):
                src_tok.append(src)
                trg_tok.append(trg)
            else:
                invalid += 1
                logger.debug('Instance in line %d is too long. '
                             'Exclude:\t%s\t%s', i, s, t)
        else: # mono-lingual dataset
            if max_len < 0 or len(src) <= max_len:
                src_tok.append(src)
            else:
                invalid += 1
                logger.debug('Instance in line %d is too long. '
                             'Exclude:\t%s', i, s)
    if invalid > 0:
        logger.debug("\t%d instances were filtered out.", invalid)
    return src_tok, trg_tok


class TranslationDataset(Dataset):
    """
    TranslationDataset which stores raw sentence pairs (tokenized)
    """
    def __init__(self, src, trg=None):
        if isinstance(trg, list) and len(trg) == 0:
            trg = None
        if trg is not None:
            assert len(src) == len(trg)
        self.src = src
        self.trg = trg

    def token_batch_size_fn(self, idx) -> int:
        """
        count num of tokens
        (used for shaping minibatch based on token count)
        :param idx:
        :return: length
        """
        length = len(self.src[idx]) + 2 # +2 because of EOS_TOKEN and BOS_TOKEN
        if self.trg:
            length = max(length, len(self.trg[idx]) + 2)
        return length

    def __getitem__(self, idx: int) -> Tuple[List[str], List[str]]:
        """
        raise a raw instance
        :param idx: index
        :return: pair of tokenized sentences
        """
        src = self.src[idx]
        trg = self.trg[idx] if self.trg else None
        return src, trg

    def __len__(self):
        return len(self.src)

    def __repr__(self):
        return "%s(len(src)=%s, len(trg)=%s)" % (self.__class__.__name__,
            len(self.src), len(self.trg) if self.trg else 0)


class TokenBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices
    based on num of tokens (incl. padding).
    * no bucketing implemented

    :param sampler: Base sampler. Can be any iterable object
    :param batch_size: Size of mini-batch.
    :param drop_last: If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        batch = []
        max_len = 0
        for idx in self.sampler:
            batch.append(idx)
            n_tokens = self.sampler.data_source.token_batch_size_fn(idx)
            if max_len < n_tokens:
                max_len = n_tokens
            if max_len * len(batch) >= self.batch_size:
                yield batch
                batch = []
                max_len = 0
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        raise NotImplementedError
