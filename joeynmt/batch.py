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
    def __init__(self, torch_batch, pad_index, use_cuda=False, **kwargs):
        self.is_train = kwargs["is_train"]
        self.specaugment = kwargs.get("specaugment", None) if self.is_train else None

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
        textgrid = torch_batch.textgrid if hasattr(torch_batch, "textgrid") else None

        if self.is_train:
            src, src_length, trg, trg_length = self._augment(
                src, src_length, trg, trg_length, textgrid)

        return src, src_length, trg, trg_length

    def _augment(self,
                 src_input: np.ndarray,
                 src_length: np.ndarray,
                 trg_input: torch.Tensor,
                 trg_length: torch.Tensor,
                 textgrid: List[List[List]] = None):
        """
        Augment Data

        :param src_input: np.ndarray, shape (batch_size, src_len, embed_size)
        :param src_length: np.ndarray, shape (batch_size)
        :param trg_input: torch.Tensor
        :param trg_length: torch.Tensor
        :param textgrid: List[List[List]]
        :return: src_input_aug, src_length_aug, trg_input_aug, trg_length_aug
        """
        batch_size = len(src_input)
        src_input_aug = src_input.copy() # (batch_size, num_freq, num_frames)
        for i in range(batch_size):

            #trg = trg_input[i]
            #t_l = trg_length[i]
            #if textgrid:
            #    tg = textgrid[i]

            s_l = src_length[i]
            if self.specaugment:
                src_input_aug[i, :s_l, :] = self.specaugment(src_input[i, :s_l, :])

        return src_input_aug, src_length, trg_input, trg_length
