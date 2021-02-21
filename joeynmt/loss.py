# coding: utf-8
"""
Module to implement training loss
"""

import torch
from torch import nn, Tensor
from torch.autograd import Variable


class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        if self.smoothing <= 0.0:
            # standard xent loss
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index,
                                        reduction='sum')
        else:
            # custom label-smoothed loss, computed with KL divergence loss
            self.criterion = nn.KLDivLoss(reduction='sum')

        self.require_ctc_layer = False

    def _smooth_targets(self, targets: Tensor, vocab_size: int):
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".

        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0-self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index,
            as_tuple=False)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    def _reshape(self, log_probs, targets):
        vocab_size = log_probs.size(-1)

        #reshape log_probs to (batch*seq_len x vocab_size)
        log_probs_flat = log_probs.contiguous().view(-1, vocab_size)

        if self.smoothing > 0:
            targets_flat = self._smooth_targets(
                targets=targets.contiguous().view(-1),
                vocab_size=vocab_size)
            # targets: distributions with batch*seq_len x vocab_size
            assert log_probs_flat.size() == targets_flat.size(), (log_probs.size() == targets_flat.size())
        else:
            # targets: indices with batch*seq_len
            targets_flat = targets.contiguous().view(-1)
            assert log_probs_flat.size(0) == targets_flat.size(0), (log_probs.size(0) == targets_flat.size(0))

        return log_probs_flat, targets_flat

    # pylint: disable=arguments-differ
    def forward(self, log_probs, **kwargs):
        """
        Compute the cross-entropy between logits and targets.

        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.

        :param log_probs: log probabilities as predicted by model
        :return:
        """
        assert "trg" in kwargs.keys()
        log_probs, targets = self._reshape(log_probs, kwargs["trg"])

        # compute loss
        loss = self.criterion(log_probs, targets)
        return (loss, )

    def __repr__(self):
        return "%s(criterion=%r, smoothing=%r)" % (
            self.__class__.__name__, self.criterion, self.smoothing)


class XentCTCLoss(XentLoss):
    """
    (label-smoothed) Cross-Entropy + CTC loss
    """
    def __init__(self, pad_index: int, bos_index: int, smoothing: float = 0.0,
                 zero_infinity: bool = True, ctc_weight: float = 0.3):
        super().__init__(pad_index=pad_index, smoothing=smoothing)

        self.require_ctc_layer = True
        self.bos_index = bos_index
        self.ctc_weight = ctc_weight
        self.ctc = nn.CTCLoss(blank=bos_index, reduction='sum',
                              zero_infinity=zero_infinity)

    # pylint: disable=arguments-differ
    def forward(self, log_probs, **kwargs):
        """
        Compute the cross-entropy loss and ctc loss

        :param log_probs: log probabilities as predicted by model
            shape (batch_size, seq_length, vocab_size)
        :return:
        """
        assert "trg" in kwargs.keys()
        assert "trg_length" in kwargs.keys()
        assert "src_mask" in kwargs.keys()
        assert "ctc_log_probs" in kwargs.keys()

        # reshape tensors for cross_entropy
        log_probs_flat, targets_flat = self._reshape(log_probs, kwargs["trg"])

        # cross_entropy loss
        xent_loss = self.criterion(log_probs_flat, targets_flat)

        # ctc_loss
        # reshape ctc_log_probs to (seq_length, batch_size, vocab_size)
        ctc_loss = self.ctc(kwargs["ctc_log_probs"].transpose(0, 1).contiguous(),
                            targets=kwargs["trg"],  # (seq_length, batch_size)
                            input_lengths=kwargs["src_mask"].squeeze(1).sum(dim=1),
                            target_lengths=kwargs["trg_length"])

        loss = (1.0-self.ctc_weight) * xent_loss + self.ctc_weight * ctc_loss
        assert loss.item() >= 0.0, "loss has to be non-negative."
        return (loss, xent_loss, ctc_loss)

    def __repr__(self):
        return "%s(criterion=%r, smoothing=%r, ctc_weight=%r)" % (
            self.__class__.__name__, self.criterion,
            self.smoothing, self.ctc_weight)
