# coding: utf-8

"""
Implementation of a mini-batch.
"""
from typing import List
import numpy as np

import torch



class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, torch_batch, pad_index, use_cuda=False, **kwargs):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with src and trg
        length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param torch_batch:
        :param pad_index:
        :param use_cuda:
        """
        self.src, self.src_length = torch_batch.src
        self.src_mask = (self.src != pad_index).unsqueeze(1)
        self.nseqs = self.src.size(0)
        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_length = None
        self.ntokens = None
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if hasattr(torch_batch, "trg"):
            trg, trg_length = torch_batch.trg
            # trg_input is used for teacher forcing, last one is cut off
            self.trg_input = trg[:, :-1]    # shape (batch_size, seq_length)
            self.trg_length = trg_length - 1
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg[:, 1:]           # shape (batch_size, seq_length)
            # we exclude the padded areas (and blank areas) from the loss computation
            self.trg_mask = (self.trg != pad_index).unsqueeze(1)
            self.ntokens = (self.trg != pad_index).data.sum().item()

        if self.use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.src = self.src.to(self.device)
        self.src_length = self.src_length.to(self.device)
        if self.src_mask is not None:
            self.src_mask = self.src_mask.to(self.device)

        if self.trg_input is not None:
            self.trg_input = self.trg_input.to(self.device)
            self.trg = self.trg.to(self.device)
            self.trg_mask = self.trg_mask.to(self.device)

    def sort_by_src_length(self):
        """
        Sort by src length (descending) and return index to revert sort

        :return:
        """
        _, perm_index = self.src_length.sort(0, descending=True)
        rev_index = [0]*perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        sorted_src_length = self.src_length[perm_index]
        sorted_src = self.src[perm_index]
        if self.src_mask is not None:
            sorted_src_mask = self.src_mask[perm_index]
        if self.trg_input is not None:
            sorted_trg_input = self.trg_input[perm_index]
            sorted_trg_length = self.trg_length[perm_index]
            sorted_trg_mask = self.trg_mask[perm_index]
            sorted_trg = self.trg[perm_index]

        self.src = sorted_src
        self.src_length = sorted_src_length
        if self.src_mask is not None:
            self.src_mask = sorted_src_mask

        if self.trg_input is not None:
            self.trg_input = sorted_trg_input
            self.trg_mask = sorted_trg_mask
            self.trg_length = sorted_trg_length
            self.trg = sorted_trg

        if self.use_cuda:
            self._make_cuda()

        return rev_index


class SpeechBatch(Batch):
    """Batch object for speech data"""
    # pylint: disable=too-many-instance-attributes
    def __init__(self, torch_batch, pad_index, bos_index, eos_index, use_cuda=False, **kwargs):
        self.pad_index = pad_index
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.is_train = kwargs["is_train"]
        if self.is_train:
            self.steps = kwargs["steps"] # number of updates so far
            self.batch_count = kwargs["batch_count"] # batch_count (will be reset in each epoch)
            self.max_src_length = kwargs["max_src_length"]
            self.max_trg_length = kwargs["max_trg_length"] + 2 # because <s> and </s>
            self.specaugment = kwargs.get("specaugment", None)
            self.masked_lm = kwargs.get("masked_lm", None)
            self.aligned_masking = kwargs.get("aligned_masking", None)
            self.mask_sampler = kwargs.get("sampler", None)
            self.seq_subsampler = kwargs.get("subsequence", None)
            self.audio_dict = kwargs.get("audio_dict", None)

        src, src_length, trg, trg_length = self._read(torch_batch)
        self.src = torch.from_numpy(src).float()
        self.src_length = torch.from_numpy(src_length).long()
        self.src_mask = None # will be constructed in encoder
        self.src_max_len = src.shape[1]
        self.nseqs = self.src.size(0)

        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_length = None
        self.ntokens = None

        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if hasattr(torch_batch, "trg"):
            # trg_input is used for teacher forcing, last one is cut off
            self.trg_input = trg[:, :-1]
            self.trg_length = trg_length - 1
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg[:, 1:]
            # we exclude the padded areas from the loss computation
            self.trg_mask = (self.trg_input != pad_index).unsqueeze(1)
            self.ntokens = (self.trg != pad_index).data.sum().item()

        if self.use_cuda:
            self._make_cuda()

    def _read(self, torch_batch):
        (src, src_length) = torch_batch.src
        (trg, trg_length) = torch_batch.trg if hasattr(torch_batch, "trg") else (None, None)
        textgrids = torch_batch.textgrid if hasattr(torch_batch, "textgrid") else None
        word2bpe = torch_batch.word2bpe if hasattr(torch_batch, "word2bpe") else None

        if self.is_train:
            src, src_length, trg, trg_length = self._augment(
                src, src_length, trg, trg_length, textgrids, word2bpe)

        return src, src_length, trg, trg_length

    def _augment(self,
                 src_input: np.ndarray,
                 src_length: np.ndarray,
                 trg_input: torch.Tensor,
                 trg_length: torch.Tensor,
                 textgrids: List[List[List]],
                 word2bpe: List[List[List]]):
        """
        Augment Data

        :param src_input: np.ndarray, shape (batch_size, src_len, embed_size)
        :param src_length: np.ndarray, shape (batch_size)
        :param trg_input: torch.Tensor
        :param trg_length: torch.Tensor
        :param textgrids: List[List[List]]
        :param word2bpe:
        :return: src_input_aug, src_length_aug, trg_input_aug, trg_length_aug
        """
        # src_input (batch_size, num_frames, num_freq)
        batch_size, src_len, num_freq = src_input.shape
        assert src_len <= self.max_src_length, (src_len, self.max_src_length)
        src_input_aug = np.zeros((batch_size, self.max_src_length, num_freq), dtype=src_input.dtype)
        src_input_aug.fill(self.pad_index)
        src_input_aug[:, :src_len, :] = src_input
        src_length_aug = src_length.copy()

        _, trg_len = trg_input.size()
        assert trg_len == trg_length.max().item(), (trg_input.size(), trg_length)
        assert trg_len <= self.max_trg_length, (trg_len, self.max_trg_length)
        trg_input_aug = torch.ones((batch_size, self.max_trg_length), dtype=torch.long)
        trg_input_aug.new_full((batch_size, self.max_trg_length), self.pad_index)
        trg_input_aug[:, :trg_len] = trg_input[:, :trg_len]
        trg_length_aug = trg_length.clone()

        batch_positions = [{}] * batch_size
        if self.seq_subsampler is not None:
            word_start, word_end = self.seq_subsampler(word2bpe)

        # get positions to be replaced on
        if self.mask_sampler is not None:
            batch_positions = self.mask_sampler(textgrids)

        # get words to be replaced with
        if sum([1 if len(p) > 0 else 0 for p in batch_positions]) > 0 and self.masked_lm is not None:
            batch_positions = self.masked_lm(batch_positions, textgrids, self.steps)

        for i in range(batch_size):
            tg = textgrids[i] if textgrids is not None else None
            w2b = word2bpe[i] if word2bpe is not None else None
            pos = batch_positions[i]

            word_cutoff = None
            if self.seq_subsampler is not None and word_start[i] is not None and word_end[i] is not None:
                word_cutoff = (word_start[i], word_end[i])

            s_l = src_length[i]
            t_l = trg_length[i]
            specaugment_flag = False if self.specaugment is None else True
            src_cutoff_flag = False if self.seq_subsampler is None else True
            trg_cutoff_flag = False if self.seq_subsampler is None else True
            if len(pos) > 0:
                if self.masked_lm is not None: # replace trg side
                    assert w2b is not None
                    #assert w2b[-1][-1] <= trg_length[i], (w2b, trg_length[i])
                    trg_aug = self.masked_lm.replace_trg(trg_input[i, :t_l], pos, w2b, word_cutoff)
                    trg_len_aug = len(trg_aug)
                    if trg_len_aug > self.max_trg_length: # truncate
                        trg_len_aug = self.max_trg_length
                        trg_aug = trg_aug[:trg_len_aug]
                    trg_length_aug[i] = trg_len_aug
                    trg_input_aug[i, :trg_len_aug] = trg_aug
                    trg_cutoff_flag = False

                if self.aligned_masking is not None:
                    assert tg is not None
                    x_aug = self.aligned_masking(src_input[i, :s_l, :], pos, tg, word_cutoff)
                    s_l = x_aug.shape[0]
                    src_input_aug[i, :, :].fill(self.pad_index)
                    src_input_aug[i, :s_l, :] = x_aug
                    src_length_aug[i] = s_l
                    specaugment_flag = False if len(pos) > 0 else True
                    src_cutoff_flag = False

                elif self.audio_dict is not None:
                    assert tg is not None
                    x_aug = self.audio_dict(src_input[i, :s_l, :], pos, tg, word_cutoff)
                    s_l = x_aug.shape[0]
                    src_input_aug[i, :, :].fill(self.pad_index)
                    src_input_aug[i, :s_l, :] = x_aug
                    src_length_aug[i] = s_l
                    src_cutoff_flag = False

            if src_cutoff_flag:
                assert word_cutoff is not None
                word_start, word_end = word_cutoff
                if word_start is not None and word_end is not None:
                    tg_aug = [t for t in tg if len(t[-1]) > 0]
                    src_start = tg_aug[word_start][0]
                    src_end = tg_aug[word_end][1]
                    s_l = src_end - src_start + 1
                    src_input_aug[i, :, :].fill(self.pad_index)
                    src_input_aug[i, :s_l, :] = src_input[i, src_start:src_end, :]
                    src_length_aug[i] = s_l

            if trg_cutoff_flag:
                assert word_cutoff is not None
                word_start, word_end = word_cutoff
                if word_start is not None and word_end is not None:
                    trg_start = word2bpe[word_start][0]
                    trg_end = word2bpe[word_end][-1]
                    t_l = trg_end - trg_start + 1
                    trg_input_aug[i, :].fill(self.pad_index)
                    trg_input_aug[i, 1] = self.bos_index
                    trg_input_aug[i, 1:t_l+1] = trg_input[i, trg_start:trg_end]
                    trg_input_aug[i, t_l+2] = self.eos_index
                    trg_length_aug[i] = t_l + 2

            if specaugment_flag: # random masking (regardless of len(pos))
                src_input_aug[i, :s_l, :] = self.specaugment(src_input_aug[i, :s_l, :])

        max_src_len = src_length_aug.max()
        max_trg_len = trg_length_aug.max()

        del src_length
        del src_input
        del trg_length
        del trg_input

        return src_input_aug[:, :max_src_len, :], src_length_aug, \
               trg_input_aug[:, :max_trg_len], trg_length_aug
