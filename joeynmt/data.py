# coding: utf-8
"""
Data module
"""
from __future__ import annotations
import sys
from pathlib import Path
from functools import partial
from typing import Optional, List, Tuple, Union, Callable
import logging
import numpy as np

import torch
from torch.utils.data import Dataset, Sampler, SequentialSampler, \
    RandomSampler, BatchSampler, DataLoader

from joeynmt.constants import PAD_ID
from joeynmt.helpers import log_data_info
from joeynmt.vocabulary import build_vocab, Vocabulary
from joeynmt.batch import Batch

logger = logging.getLogger(__name__)

# multiprocessing
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError as e:
    logger.debug("torch.multiprocessing.set_start_method('spawn') faild. %s", e)

# pandas (for multiprocessing)
try:
    import pandas as pd
except ImportError as no_pd:
    logger.debug('pandas package not found. %s', no_pd)


def load_data(data_cfg: dict, datasets: list = None, num_workers: int = 0) \
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
        ("data" part of configuration file)
    :param datasets: list of dataset names to load
    :param num_workers:
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: test dataset if given, otherwise None
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

    train_data, train_src, train_trg = None, None, None
    if "train" in datasets and train_path is not None:
        logger.info("Loading training data...")
        train_src, train_trg = read_data_file(path=Path(train_path),
                                              exts=(src_lang, trg_lang),
                                              level=level,
                                              lowercase=lowercase,
                                              max_len=max_len,
                                              num_workers=num_workers)

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
    assert src_vocab.pad_index == trg_vocab.pad_index
    assert src_vocab.bos_index == trg_vocab.bos_index
    assert src_vocab.eos_index == trg_vocab.eos_index

    if train_data is not None:
        train_data.src_padding = src_vocab.sentences_to_ids
        train_data.trg_padding = trg_vocab.sentences_to_ids

    dev_data = None
    if "dev" in datasets and dev_path is not None:
        logger.info("Loading dev data...")
        dev_src, dev_trg = read_data_file(path=Path(dev_path),
                                          exts=(src_lang, trg_lang),
                                          level=level,
                                          lowercase=lowercase,
                                          num_workers=num_workers)
        dev_data = TranslationDataset(dev_src, dev_trg,
                                      src_padding=src_vocab.sentences_to_ids,
                                      trg_padding=trg_vocab.sentences_to_ids)

    test_data = None
    if "test" in datasets and test_path is not None:
        logger.info("Loading test data...")
        # check if target exists
        if not Path(f'{test_path}.{trg_lang}').is_file():
            # no target is given -> create dataset from src only
            trg_lang = None
        test_src, test_trg = read_data_file(path=Path(test_path),
                                            exts=(src_lang, trg_lang),
                                            level=level,
                                            lowercase=lowercase,
                                            num_workers=num_workers)
        test_data = TranslationDataset(test_src, test_trg,
                                       src_padding=src_vocab.sentences_to_ids,
                                       trg_padding=trg_vocab.sentences_to_ids)

    logger.info("Data loaded.")
    log_data_info(train_data, dev_data, test_data, src_vocab, trg_vocab)
    return train_data, dev_data, test_data, src_vocab, trg_vocab

# should locate in top-level (global scope)
def _apply(df: pd.DataFrame, level: str = "bpe", lowercase: bool = True,
           max_len: int = -1) -> pd.DataFrame:
    tok_fn = partial(_tokenize, level=level, lowercase=lowercase)
    filter_fn = partial(_filter, max_len=max_len)
    df['src_tok'] = df['src'].apply(tok_fn)
    df['src_mask'] = df['src_tok'].apply(filter_fn)
    if 'trg' in df.columns:
        df['trg_tok'] = df['trg'].apply(tok_fn)
        df['trg_mask'] = df['trg_tok'].apply(filter_fn)
    return df

# should locate in top-level (global scope)
def _tokenize(x: str, level: str = "bpe", lowercase: bool = True):
    x = x.strip().lower() if lowercase else x.strip()
    return list(x) if level == "char" else x.split()

# should locate in top-level (global scope)
def _filter(tokens: List[str], max_len: int = -1):
    return not len(tokens) > max_len > 0


def read_data_file(path: Path, exts: Tuple[str, Union[str, None]],
                   level: str, lowercase: bool, max_len: int = -1,
                   num_workers: int = 0) \
        -> Tuple[List[List[str]], List[List[str]]]:
    """
    Read data files
    :param path: data file path
    :param exts: pair of file extensions
    :param level: tokenization
    :param lowercase: whether to lowercase or not
    :param max_len: maximum length (longer instances will be filtered out)
    :param num_workers:
    :return: pair of tokenized sentence lists
    """
    src_lang, trg_lang = exts
    src_file = path.with_suffix(f'{path.suffix}.{src_lang}')
    doc = {'src': src_file.read_text().splitlines()}
    if trg_lang:
        trg_file = path.with_suffix(f'{path.suffix}.{trg_lang}')
        doc['trg'] = trg_file.read_text().splitlines()
        assert len(doc['src']) == len(doc['trg'])

    if 'pandas' in sys.modules:
        df = pd.DataFrame.from_dict(doc)

        if len(df) > num_workers > 1:
            with torch.multiprocessing.Pool(processes=num_workers) as pool:
                df = pd.concat(pool.map(partial(_apply,
                                                level=level,
                                                lowercase=lowercase,
                                                max_len=max_len),
                                        np.array_split(df, num_workers)))
        else:
            df = _apply(df, level=level, lowercase=lowercase, max_len=max_len)

        invalid = df[~df['src_mask'] | ~df['trg_mask']].index if trg_lang \
            else df[~df['src_mask']].index
        if len(invalid) > 0:
            df = df.drop(invalid, axis=0)
            logger.warning('\t%d instances were filtered out.', len(invalid))
        src_list = df['src_tok'].tolist()
        trg_list = df['trg_tok'].tolist() if trg_lang else []

    else:
        # pandas-package not available
        src_tok = [_tokenize(x, level, lowercase) for x in doc['src']]
        if trg_lang:
            trg_tok = [_tokenize(x, level, lowercase) for x in doc['trg']]
            src_list, trg_list = zip(*[(s, t) for s, t in zip(src_tok, trg_tok)
                               if _filter(s, max_len) and _filter(t, max_len)])
            assert len(src_list) == len(trg_list)
        else:
            src_list = [s for s in src_tok if _filter(s, max_len)]
            trg_list = []
    return src_list, trg_list

# should locate in top-level (global scope)
def collate_fn(batch, src_process, trg_process, batch_class,
               pad_index, device) -> Batch:
    src_list, trg_list = zip(*batch)
    src, src_length = src_process(src_list)

    trg, trg_length = None, None
    if any(map(None.__ne__, trg_list)):
        trg, trg_length = trg_process(trg_list)

    return batch_class(torch.LongTensor(src),
                       torch.LongTensor(src_length),
                       torch.LongTensor(trg) if trg else None,
                       torch.LongTensor(trg_length) if trg_length else None,
                       pad_index, device)


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   batch_class: Batch = Batch,
                   seed: int = 42,
                   shuffle: bool = False,
                   num_workers: int = 0,
                   pad_index: int = PAD_ID,
                   device: torch.device = torch.device("cpu")) -> DataLoader:
    """
    Returns a torch DataLoader for a torch Dataset. (no bucketing)

    :param dataset: torch dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param batch_class:
    :param seed: random seed for shuffling
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :param num_workers: number of cpus for multiprocessing
    :param pad_index:
    :param device:
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

    assert dataset.src_padding is not None
    assert dataset.trg_padding is not None

    # data iterator
    return DataLoader(dataset,
                      batch_sampler=batch_sampler,
                      collate_fn=partial(collate_fn,
                                         src_process=dataset.src_padding,
                                         trg_process=dataset.trg_padding,
                                         batch_class=batch_class,
                                         pad_index=pad_index,
                                         device=device),
                      num_workers=num_workers)


class TranslationDataset(Dataset):
    """
    TranslationDataset which stores raw sentence pairs (tokenized)

    :param src: list of tokenized sentences in src language
    :param trg: list of tokenized sentences in trg language
    :param src_padding: padding function for src
    :param trg_padding: padding function for trg
    """
    def __init__(self, src: List[List[str]], trg: List[List[str]]=None,
                 src_padding: Callable=None, trg_padding: Callable=None):
        if isinstance(trg, list) and len(trg) == 0:
            trg = None
        if trg is not None:
            assert len(src) == len(trg)
        self.src = src
        self.trg = trg

        # assigned after vocab is built
        self.src_padding = src_padding
        self.trg_padding = trg_padding

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
