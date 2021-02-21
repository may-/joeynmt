import torch

from joeynmt.loss import XentLoss, XentCTCLoss
from .test_helpers import TensorTestCase


class TestLoss(TensorTestCase):

    def setUp(self):
        seed = 42
        torch.manual_seed(seed)

    def test_label_smoothing(self):
        pad_index = 0
        smoothing = 0.4
        criterion = XentLoss(pad_index=pad_index, smoothing=smoothing)

        # batch x seq_len x vocab_size: 3 x 2 x 5
        predict = torch.FloatTensor(
            [[[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
             [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
             [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]
        )

        # batch x seq_len: 3 x 2
        targets = torch.LongTensor([[2, 1],
                                    [2, 0],
                                    [1, 0]])

        # test the smoothing function
        smoothed_targets = criterion._smooth_targets(targets=targets.view(-1),
                                      vocab_size=predict.size(-1))
        self.assertTensorAlmostEqual(
            smoothed_targets,
            torch.Tensor(
                [[0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
                 [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
                 [0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
                 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                 [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
                 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
        )
        assert torch.max(smoothed_targets) == 1-smoothing

        # test the loss computation
        kwargs = {"trg": targets}
        v = criterion(predict.log(), **kwargs)
        self.assertTensorAlmostEqual(v[0], 2.1326)

    def test_no_label_smoothing(self):
        pad_index = 0
        smoothing = 0.0
        criterion = XentLoss(pad_index=pad_index, smoothing=smoothing)

        # batch x seq_len x vocab_size: 3 x 2 x 5
        predict = torch.FloatTensor(
            [[[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
             [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
             [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]
        )

        # batch x seq_len: 3 x 2
        targets = torch.LongTensor([[2, 1],
                                    [2, 0],
                                    [1, 0]])

        # test the smoothing function: should still be one-hot
        smoothed_targets = criterion._smooth_targets(targets=targets.view(-1),
                                      vocab_size=predict.size(-1))

        assert torch.max(smoothed_targets) == 1
        assert torch.min(smoothed_targets) == 0

        self.assertTensorAlmostEqual(
            smoothed_targets,
            torch.Tensor(
                [[0., 0., 1., 0., 0.],
                 [0., 1., 0., 0., 0.],
                 [0., 0., 1., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 1., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]])
        )
        kwargs = {"trg": targets}
        v = criterion(predict.log(), **kwargs)
        self.assertTensorAlmostEqual(v[0], 5.6268)

    def test_ctc_loss(self):
        pad_index = 0
        bos_index = 1
        smoothing = 0.4
        ctc_weight = 0.3
        criterion = XentCTCLoss(pad_index=pad_index, bos_index=bos_index,
                                smoothing=smoothing, ctc_weight=ctc_weight)

        # batch x seq_len x vocab_size: 3 x 2 x 5
        predict = torch.FloatTensor(
            [[[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
             [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
             [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]
        )

        # batch x seq_len x vocab_size: 3 x 4 x 5
        ctc_predict = torch.FloatTensor(
            [[[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1],
              [0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
             [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1],
              [0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
             [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1],
              [0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]
        )

        # batch x seq_len
        src_mask = torch.BoolTensor(
            [[True, True, True, False],
             [True, True, False, False],
             [True, True, True, True]]
        )

        # batch x seq_len: 3 x 2
        targets = torch.LongTensor([[2, 1],
                                    [2, 0],
                                    [1, 0]])
        trg_length = torch.LongTensor([[2], [1], [1]]) # batch x 1
        src_length = torch.LongTensor([[3], [2], [4]]) # batch x 1

        # test the loss computation
        kwargs = {"trg": targets, "trg_length": trg_length,
                  "ctc_log_probs": ctc_predict.log(),
                  "src_mask": src_mask,
                  "src_length": src_length}
        total_loss, nll_loss, ctc_loss = criterion(predict.log(), **kwargs)
        self.assertTensorAlmostEqual(total_loss, 4.6294)
        self.assertTensorAlmostEqual(nll_loss, 2.1326)
        self.assertTensorAlmostEqual(ctc_loss, 10.4551)