# coding: utf-8

"""
Implementation of a mini-batch.
"""


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """
    def __init__(self, src, src_length, trg, trg_length, pad_index):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with src and trg
        length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param src:
        :param src_length:
        :param trg:
        :param trg_length:
        :param pad_index:
        """
        # pylint: disable=too-many-instance-attributes
        self.src = src
        self.src_length = src_length
        self.src_mask = (self.src != pad_index).unsqueeze(1)
        self.nseqs = self.src.size(0)
        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_length = None
        self.ntokens = None
        #self.use_cuda = use_cuda
        #self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if trg is not None and trg_length is not None:
            # trg_input is used for teacher forcing, last one is cut off
            self.trg_input = trg[:, :-1]
            self.trg_length = trg_length
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg[:, 1:]
            # we exclude the padded areas from the loss computation
            self.trg_mask = (self.trg_input != pad_index).unsqueeze(1)
            self.ntokens = (self.trg != pad_index).data.sum().item()

        #if self.use_cuda:
        #    self._make_cuda()

    def make_cuda(self, device) -> None:
        """
        Move the batch to GPU
        """
        self.src = self.src.to(device)
        self.src_mask = self.src_mask.to(device)
        self.src_length = self.src_length.to(device)

        if self.trg_input is not None:
            self.trg_input = self.trg_input.to(device)
            self.trg = self.trg.to(device)
            self.trg_mask = self.trg_mask.to(device)

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
        sorted_src_mask = self.src_mask[perm_index]
        if self.trg_input is not None:
            sorted_trg_input = self.trg_input[perm_index]
            sorted_trg_length = self.trg_length[perm_index]
            sorted_trg_mask = self.trg_mask[perm_index]
            sorted_trg = self.trg[perm_index]

        self.src = sorted_src
        self.src_length = sorted_src_length
        self.src_mask = sorted_src_mask

        if self.trg_input is not None:
            self.trg_input = sorted_trg_input
            self.trg_mask = sorted_trg_mask
            self.trg_length = sorted_trg_length
            self.trg = sorted_trg

        #if self.use_cuda:
        #    self._make_cuda()

        return rev_index
