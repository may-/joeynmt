# coding: utf-8
"""
Implementation of a mini-batch.
"""
from typing import List, Optional

import torch
from torch import Tensor


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is yielded from collate_fn in torch.data.utils.DataLoader.
    """
    def __init__(self, src: Tensor, src_length: Tensor, trg: Optional[Tensor],
                 trg_length: Optional[Tensor], pad_index: int,
                 device: torch.device):
        """
        Create a new joey batch. This batch supports attributes with
        src and trg length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param src:
        :param src_length:
        :param trg:
        :param trg_length:
        :param pad_index:
        :param device:
        """
        # pylint: disable=too-many-instance-attributes
        self.src = src
        self.src_length = src_length
        self.src_mask = (self.src != pad_index).unsqueeze(1)
        self.nseqs = self.src.size(0)
        self.trg_input: Optional[Tensor] = None
        self.trg: Optional[Tensor] = None
        self.trg_mask: Optional[Tensor] = None
        self.trg_length: Optional[Tensor] = None
        self.ntokens: Optional[int] = None
        self._has_trg = trg is not None and trg_length is not None

        if self._has_trg:
            # trg_input is used for teacher forcing, last one is cut off
            self.trg_input = trg[:, :-1]
            self.trg_length = trg_length
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg[:, 1:]
            # we exclude the padded areas from the loss computation
            self.trg_mask = (self.trg_input != pad_index).unsqueeze(1)
            self.ntokens = (self.trg != pad_index).data.sum().item()

        if device.type == "cuda":
            self._make_cuda(device)

    def _make_cuda(self, device: torch.device) -> None:
        """
        Move the batch to GPU
        """
        self.src = self.src.to(device)
        self.src_mask = self.src_mask.to(device)
        self.src_length = self.src_length.to(device)

        if self._has_trg:
            self.trg_input = self.trg_input.to(device)
            self.trg = self.trg.to(device)
            self.trg_mask = self.trg_mask.to(device)

    def sort_by_src_length(self) -> List[int]:
        """
        Sort by src length (descending) and return index to revert sort

        :return: list of indices
        """
        _, perm_index = self.src_length.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        sorted_src_length = self.src_length[perm_index]
        sorted_src = self.src[perm_index]
        sorted_src_mask = self.src_mask[perm_index]
        if self._has_trg:
            sorted_trg_input = self.trg_input[perm_index]
            sorted_trg_length = self.trg_length[perm_index]
            sorted_trg_mask = self.trg_mask[perm_index]
            sorted_trg = self.trg[perm_index]

        self.src = sorted_src
        self.src_length = sorted_src_length
        self.src_mask = sorted_src_mask

        if self._has_trg:
            self.trg_input = sorted_trg_input
            self.trg_mask = sorted_trg_mask
            self.trg_length = sorted_trg_length
            self.trg = sorted_trg

        return rev_index
