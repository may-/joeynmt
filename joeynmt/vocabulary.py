# coding: utf-8

"""
Vocabulary module
"""
from collections import Counter
from pathlib import Path

from typing import List, Tuple
import numpy as np

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN, \
    UNK_ID, PAD_ID, BOS_ID, EOS_ID
from joeynmt.helpers import flatten, write_list_to_file


class Vocabulary:
    """ Vocabulary represents mapping between tokens and indices. """

    def __init__(self, tokens: List[str] = None, file: Path = None) -> None:
        """
        Create vocabulary from list of tokens or file.

        Special tokens are added if not already in file or list.
        File format: token with index i is in line i.

        :param tokens: list of tokens
        :param file: file to load vocabulary from
        """
        # warning: stoi grows with unknown tokens, don't use for saving or size

        # special symbols
        self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

        # don't allow to access _stoi and _itos outside of this class
        self._stoi = {} # string to index
        self._itos = [] # index to string
        if tokens is not None:
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)

        # assign after stoi is built
        self.pad_index = self.lookup(PAD_TOKEN)
        self.bos_index = self.lookup(BOS_TOKEN)
        self.eos_index = self.lookup(EOS_TOKEN)
        assert self.pad_index == PAD_ID
        assert self.bos_index == BOS_ID
        assert self.eos_index == EOS_ID
        assert self._itos[UNK_ID] == UNK_TOKEN

    def _from_list(self, tokens: List[str] = None) -> None:
        """
        Make vocabulary from list of tokens.
        Tokens are assumed to be unique and pre-selected.
        Special symbols are added if not in list.

        :param tokens: list of tokens
        """
        self.add_tokens(tokens=self.specials+tokens)
        assert len(self._stoi) == len(self._itos)

    def _from_file(self, file: Path) -> None:
        """
        Make vocabulary from contents of file.
        File format: token with index i is in line i.

        :param file: path to file where the vocabulary is loaded from
        """
        tokens = []
        with file.open("r") as open_file:
            for line in open_file:
                tokens.append(line.strip("\n"))
        self._from_list(tokens)

    def __str__(self) -> str:
        return self._stoi.__str__()

    def to_file(self, file: Path) -> None:
        """
        Save the vocabulary to a file, by writing token with index i in line i.

        :param file: path to file where the vocabulary is written
        """
        write_list_to_file(file, self._itos)

    def add_tokens(self, tokens: List[str]) -> None:
        """
        Add list of tokens to vocabulary

        :param tokens: list of tokens to add to the vocabulary
        """
        for t in tokens:
            new_index = len(self._itos)
            # add to vocab if not already there
            if t not in self._itos:
                self._itos.append(t)
                self._stoi[t] = new_index

    def is_unk(self, token: str) -> bool:
        """
        Check whether a token is covered by the vocabulary

        :param token:
        :return: True if covered, False otherwise
        """
        return self.lookup(token) == UNK_ID

    def lookup(self, token: str) -> int:
        """
        look up the encoding dictionary.
        (needed for multiprocessing)
        :param token: surface str
        :return: token id
        """
        return self._stoi.get(token, UNK_ID)

    def __len__(self) -> int:
        return len(self._itos)

    def __eq__(self, other) -> bool:
        if isinstance(other, Vocabulary):
            return self._itos == other._itos
        return False

    def array_to_sentence(self, array: np.array, cut_at_eos=True,
                          skip_pad=True) -> List[str]:
        """
        Converts an array of IDs to a sentence, optionally cutting the result
        off at the end-of-sequence token.

        :param array: 1D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :param skip_pad: skip generated <pad> tokens
        :return: list of strings (tokens)
        """
        sentence = []
        for i in array:
            s = self._itos[i]
            if cut_at_eos and s == EOS_TOKEN:
                break
            if skip_pad and s == PAD_TOKEN:
                continue
            sentence.append(s)
        return sentence

    def arrays_to_sentences(self, arrays: np.array, cut_at_eos=True,
                            skip_pad=True) -> List[List[str]]:
        """
        Convert multiple arrays containing sequences of token IDs to their
        sentences, optionally cutting them off at the end-of-sequence token.

        :param arrays: 2D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :param skip_pad: skip generated <pad> tokens
        :return: list of list of strings (tokens)
        """
        sentences = []
        for array in arrays:
            sentences.append(
                self.array_to_sentence(array=array, cut_at_eos=cut_at_eos,
                                       skip_pad=skip_pad))
        return sentences

    def sentences_to_ids(self, sentences: List[List[str]]) \
            -> Tuple[List[List[int]], List[int]]:
        """
        Encode sentences to indices and pad sequences
        :param sentences: list of tokenized sentences
        :return: padded ids and lengths
        """
        max_len = max([len(sent) for sent in sentences]) + 2
        padded, lengths = [], []
        for sent in sentences:
            encoded = [self.lookup(s) for s in sent]
            ids = [self.bos_index] + encoded + [self.eos_index]
            offset = max(0, max_len - len(ids))
            padded.append(ids + [self.pad_index] * offset)
            lengths.append(len(ids))
        return padded, lengths

    def log_vocab(self, k: int) -> str:
        """first k vocab entities"""
        return " ".join(f'({i}) {t}' for i, t in enumerate(self._itos[:k]))


def build_vocab(max_size: int, min_freq: int, tokens: List[List[str]],
                vocab_file: Path = None) -> Vocabulary:
    """
    Builds vocabulary from given `tokens` list or `vocab_file`.

    :param max_size: maximum size of vocabulary
    :param min_freq: minimum frequency for an item to be included
    :param tokens: list of tokenized sentences (raw dataset)
    :param vocab_file: file to store the vocabulary,
        if not None, load vocabulary from here
    :return: Vocabulary created from either `tokens` or `vocab_file`
    """

    if vocab_file is not None:
        # load it from file
        vocab = Vocabulary(file=vocab_file)
    else:
        # create newly
        def filter_min(counter: Counter, min_freq: int):
            """ Filter counter by min frequency """
            filtered_counter = Counter({t: c for t, c in counter.items()
                                        if c >= min_freq})
            return filtered_counter

        def sort_and_cut(counter: Counter, limit: int):
            """ Cut counter to most frequent,
            sorted numerically and alphabetically"""
            # sort by frequency, then alphabetically
            tokens_and_frequencies = sorted(counter.items(),
                                            key=lambda tup: tup[0])
            tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
            return [i[0] for i in tokens_and_frequencies[:limit]]

        counter = Counter(flatten(tokens))
        if min_freq > -1:
            counter = filter_min(counter, min_freq)
        vocab_tokens = sort_and_cut(counter, max_size)
        assert len(vocab_tokens) <= max_size

        vocab = Vocabulary(tokens=vocab_tokens)
        assert len(vocab) <= max_size + len(vocab.specials)

    # check for all except for UNK token whether they are OOVs
    for s in vocab.specials[1:]:
        assert not vocab.is_unk(s)

    return vocab
