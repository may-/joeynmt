# coding: utf-8
"""
Data module
"""
import sys
import random
import csv
import os
import os.path
from functools import partial
from typing import Optional
import logging

from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import build_vocab, Vocabulary
from joeynmt.helpers import align_words_to_bpe, log_data_info, ConfigurationError
from joeynmt.helpers_for_audio import SpeechInstance, pad_features, get_textgrid


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
        ("data" part of configuration file)
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

    task = data_cfg.get("task", "MT")
    assert task in {"MT", "s2t"}

    # load data from files
    if task == "MT":
        src_lang = data_cfg["src"]
        trg_lang = data_cfg["trg"]
    elif task == "s2t":
        root_path = data_cfg["root_path"]
    train_path = data_cfg.get("train", None)
    dev_path = data_cfg.get("dev", None)
    test_path = data_cfg.get("test", None)

    if train_path is None and dev_path is None and test_path is None:
        raise ValueError('Please specify at least one data source path.')

    level = data_cfg["level"]
    if level not in ["word", "bpe", "char"]:
        raise ConfigurationError("Invalid segmentation level. "
                                 "Valid options: 'word', 'bpe', 'char'.")
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]
    if task == "s2t":
        max_feat_length = data_cfg.get("max_feat_length", 3000)
        num_freq = data_cfg.get("num_freq", 80)

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    if task == "MT":
        src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                               pad_token=PAD_TOKEN, tokenize=tok_fun,
                               batch_first=True, lower=lowercase,
                               unk_token=UNK_TOKEN,
                               include_lengths=True)
    elif task == "s2t":
        src_field = data.RawField(
            postprocessing=partial(pad_features, embed_size=num_freq),
            is_target=False)

    textgrid = data_cfg.get("textgrid", False)
    if textgrid:
        textgrid_field = data.RawField(is_target=False)
        word2bpe_field = data.RawField(is_target=False)

    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)

    train_data = None
    if "train" in datasets and train_path is not None:
        logger.info("Loading training data...")
        if task == "MT":
            train_data = TranslationDataset(
                path=train_path, exts=("." + src_lang, "." + trg_lang),
                fields=(src_field, trg_field),
                filter_pred=lambda x: len(vars(x)['src']) <= max_sent_length
                                      and len(vars(x)['trg']) <= max_sent_length)
        elif task == "s2t":
            fields = [('src', src_field), ('trg', trg_field)]
            kwargs = {}
            if textgrid:
                kwargs['frame_shift'] = data_cfg.get("frame_shift", 100)
                fields.extend([('textgrid', textgrid_field), ('word2bpe', word2bpe_field)])
            train_data = SpeechDataset(
                root_path=root_path, tsv_file=train_path, fields=fields,
                filter_pred=lambda x: len(vars(x)['src']) <= max_feat_length
                                      and len(vars(x)['trg']) <= max_sent_length,
                is_train=True, **kwargs)

        random_train_subset = data_cfg.get("random_train_subset", -1)
        if random_train_subset > -1:
            # select this many training examples randomly and discard the rest
            keep_ratio = random_train_subset / len(train_data)
            keep, _ = train_data.split(
                split_ratio=[keep_ratio, 1 - keep_ratio],
                random_state=random.getstate())
            train_data = keep

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    src_vocab_file = data_cfg.get("src_vocab", None)
    trg_vocab_file = data_cfg.get("trg_vocab", None)

    if task == "MT":
        assert (train_data is not None) or (src_vocab_file is not None)
    assert (train_data is not None) or (trg_vocab_file is not None)

    logger.info("Building vocabulary...")
    src_vocab = None
    if task == "MT":
        src_vocab = build_vocab(field="src", min_freq=src_min_freq,
                                max_size=src_max_size,
                                dataset=train_data, vocab_file=src_vocab_file)
    trg_vocab = build_vocab(field="trg", min_freq=trg_min_freq,
                            max_size=trg_max_size,
                            dataset=train_data, vocab_file=trg_vocab_file)

    dev_data = None
    if "dev" in datasets and dev_path is not None:
        logger.info("Loading dev data...")
        if task == "MT":
            dev_data = TranslationDataset(path=dev_path,
                                          exts=("." + src_lang, "." + trg_lang),
                                          fields=(src_field, trg_field))
        elif task == "s2t":
            dev_data = SpeechDataset(
                root_path=root_path, tsv_file=dev_path,
                fields=[('src', src_field), ('trg', trg_field)], is_train=False)

    test_data = None
    if "test" in datasets and test_path is not None:
        logger.info("Loading test data...")
        if task == "MT":
            # check if target exists
            if os.path.isfile(test_path + "." + trg_lang):
                test_data = TranslationDataset(
                    path=test_path, exts=("." + src_lang, "." + trg_lang),
                    fields=(src_field, trg_field))
            else:
                # no target is given -> create dataset from src only
                test_data = MonoDataset(path=test_path, ext="." + src_lang,
                                        field=src_field)
        elif task == "s2t":
            test_data = SpeechDataset(
                root_path=root_path, tsv_file=test_path,
                fields=[('src', src_field), ('trg', trg_field)], is_train=False)

    if task == "MT":
        src_field.vocab = src_vocab
    trg_field.vocab = trg_vocab
    logger.info("Data loaded.")

    # log data info
    log_data_info(train_data=train_data, valid_data=dev_data,
                  test_data=test_data, src_vocab=src_vocab, trg_vocab=trg_vocab)

    return train_data, dev_data, test_data, src_vocab, trg_vocab


# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    if hasattr(new, 'trg'):  # for monolingual data sets ("translate" mode)
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=False, sort=False)

    return data_iter


class MonoDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, ext: str, field: Field, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path and field.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        """

        fields = [('src', field)]

        if hasattr(path, "readline"):  # special usage: stdin
            src_file = path
        else:
            src_path = os.path.expanduser(path + ext)
            src_file = open(src_path)

        examples = []
        for src_line in src_file:
            src_line = src_line.strip()
            if src_line != '':
                examples.append(data.Example.fromlist(
                    [src_line], fields))

        src_file.close()

        super().__init__(examples, fields, **kwargs)


class SpeechDataset(TranslationDataset):
    """Defines a dataset for audio processing."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, root_path, tsv_file, fields, is_train, **kwargs):
        """Create a SpeechDataset given paths and fields.

        :param root_path: root path for both tsv and zip file.
        :param tsv_file: tsv file name.
        :param fields: A tuple containing the fields that will be used for data
                in each language.
        :param is_train: bool
        """
        assert isinstance(fields, list) and isinstance(fields[0], tuple)
        root_path = os.path.expanduser(root_path)
        headers = [name for name, _ in fields]
        if "textgrid" in headers:
            frame_shift = kwargs.get("frame_shift", 100)
        if "frame_shift" in kwargs:
            del kwargs["frame_shift"]
        assert 'src' in headers # mono dataset without 'trg' also covered by this class

        ##### expected dir structure #####
        # path/to/project/root
        # ├── train.tsv             # tsv files
        # ├── dev.tsv
        # ├── test.tsv
        # ├── fbank80.zip           # ex1) compressed: zip file with audio features (byte)
        # └── fbank80               # ex2) uncompressed: directory which contains
        #   ├── 116-288045-0.npy    #   audio features (np.ndarray, shape: (num_frames, num_freq))
        #   ├── 116-288045-1.npy
        #   └── 116-288045-2.npy
        #
        ##### tsv format (first line => headers) #####
        # - `id [TAB] src [TAB] n_frames [TAB] trg` (src + trg, for training)
        # - `id [TAB] src [TAB] n_frames`           (src only, for inference)
        #   *we need information about the number of time frames (seq_len),
        #   because we only store the path strings in working memory of the data loader,
        #   and construct numpy array everytime a batch object is instantiated.
        #
        ##### ex1) audio features saved in *.zip #####
        # id	src	n_frames	trg
        # 116-288045-0	fbank80.zip:12526644911:340288	1063	▁as ▁i ▁approach ed ▁the ▁city ▁i ▁hear d ▁bell s ▁ring ing
        # 116-288045-1	fbank80.zip:59760130943:275968	862 ▁look ing ▁a bout ▁me ▁i ▁saw ▁a ▁gentleman ▁in ▁a ▁neat ▁black ▁dress
        # 116-288045-2	fbank80.zip:25990692418:307648	961	▁he ▁must ▁have ▁realize d ▁i ▁was ▁a ▁stranger
        #
        ##### ex2) audio features saved in *.npy #####
        # id    src n_frames    trg
        # 116-288045-0	fbank80/116-288045-0.npy	1063	▁as ▁i ▁approach ed ▁the ▁city ▁i ▁hear d ▁bell s ▁ring ing
        # 116-288045-1	fbank80/116-288045-1.npy	862 ▁look ing ▁a bout ▁me ▁i ▁saw ▁a ▁gentleman ▁in ▁a ▁neat ▁black ▁dress
        # 116-288045-2	fbank80/116-288045-2.npy	961	▁he ▁must ▁have ▁realize d ▁i ▁was ▁a ▁stranger
        #
        #########################################
        tsv_path = os.path.join(root_path, f"{tsv_file}.tsv")
        if not os.path.isfile(tsv_path):
            raise FileNotFoundError(f"Dataset not found: {tsv_path}")
        examples = []
        with open(tsv_path) as f:
            reader = csv.DictReader(
                f,
                delimiter="\t",
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
            )
            for i, dic in enumerate(reader):
                try:
                    assert 'src' in dic.keys() and 'n_frames' in dic.keys()
                    example = dict()
                    _id = dic['id'] if 'id' in dic.keys() else i
                    example['src'] = SpeechInstance(os.path.join(root_path, dic['src']),
                                                    int(dic['n_frames']), str(_id))
                    if 'trg' in headers:
                        example['trg'] = dic['trg']
                    if 'textgrid' in headers and 'trg_text' in dic.keys():
                        textgrid_path = os.path.join(root_path, dic['textgrid'])
                        words = dic['trg_text'].lower()
                        example['textgrid'] = get_textgrid(textgrid_path, words, frame_shift=frame_shift)
                        example['word2bpe'] = align_words_to_bpe(dic['trg'].split(), words.split(), start=1)
                    examples.append(data.Example.fromlist([example[h] for h in headers], fields))
                except Exception as e:
                    logger.warning(f'skip: {dic} ({e})')

        assert len(examples) > 0

        self.is_train = is_train
        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

    def __repr__(self):
        return (self.__class__.__name__
                + "(fields=[" + ", ".join([f for f in self.fields.keys()]) + "], "
                + f"is_train={self.is_train}, len(examples)={len(self.examples)}")
