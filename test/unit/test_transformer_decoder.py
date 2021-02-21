import torch

from joeynmt.decoders import TransformerDecoder, TransformerDecoderLayer
from .test_helpers import TensorTestCase


class TestTransformerDecoder(TensorTestCase):

    def setUp(self):
        self.emb_size = 12
        self.num_layers = 3
        self.hidden_size = 12
        self.ff_size = 24
        self.num_heads = 4
        self.dropout = 0.
        seed = 42
        torch.manual_seed(seed)

    def test_transformer_decoder_freeze(self):
        decoder = TransformerDecoder(freeze=True)
        for n, p in decoder.named_parameters():
            self.assertFalse(p.requires_grad)

    def test_transformer_decoder_output_size(self):

        vocab_size = 11
        decoder = TransformerDecoder(
            num_layers=self.num_layers, num_heads=self.num_heads,
            hidden_size=self.hidden_size, ff_size=self.ff_size,
            dropout=self.dropout, vocab_size=vocab_size)

        if not hasattr(decoder, "output_size"):
            self.fail("Missing output_size property.")

        self.assertEqual(decoder.output_size, vocab_size)

    def test_transformer_decoder_forward(self):
        batch_size = 2
        src_time_dim = 4
        trg_time_dim = 5
        vocab_size = 7

        trg_embed = torch.rand(size=(batch_size, trg_time_dim, self.emb_size))

        kwargs = {"encoder_output_size_for_ctc": self.hidden_size}
        decoder = TransformerDecoder(
            num_layers=self.num_layers, num_heads=self.num_heads,
            hidden_size=self.hidden_size, ff_size=self.ff_size,
            dropout=self.dropout, emb_dropout=self.dropout,
            vocab_size=vocab_size, **kwargs)
        self.assertTrue(hasattr(decoder, "ctc_output_layer"))
        self.assertTrue(isinstance(decoder.ctc_output_layer, torch.nn.Module))

        encoder_output = torch.rand(
            size=(batch_size, src_time_dim, self.hidden_size))

        for p in decoder.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5)

        src_mask = torch.ones(size=(batch_size, 1, src_time_dim)) == 1
        trg_mask = torch.ones(size=(batch_size, trg_time_dim, 1)) == 1

        encoder_hidden = None  # unused
        decoder_hidden = None  # unused
        unrol_steps = None  # unused

        output, states, _, _, ctc_output = decoder(
            trg_embed, encoder_output, encoder_hidden, src_mask, unrol_steps,
            decoder_hidden, trg_mask)

        output_target = torch.Tensor(
            [[[-0.8291,  0.2765, -0.0401, -0.8119,  0.1487,  0.2879,  0.0542],
              [-0.6970,  0.3906, -0.0879, -0.7729,  0.3426,  0.2602,  0.0733],
              [-0.8277,  0.3840, -0.0922, -0.8529,  0.4367,  0.4324,  0.2136],
              [-0.9068,  0.2878, -0.1306, -0.8873,  0.2240,  0.4427,  0.0881],
              [-0.9619,  0.1944, -0.0476, -0.9068,  0.3196,  0.4325,  0.1338]],
             [[-0.7912,  0.4026, -0.0350, -0.8364,  0.2085,  0.3298, -0.0281],
              [-0.7424,  0.3011, -0.0937, -0.8910,  0.3860,  0.2342,  0.0549],
              [-0.9302,  0.1826,  0.0044, -1.0060,  0.5166,  0.4258,  0.2572],
              [-0.9016,  0.2193, -0.0601, -0.9432,  0.4396,  0.3899,  0.1860],
              [-1.1026,  0.1776, -0.0025, -0.9692,  0.0896,  0.5584,  0.0539]]]
        )
        self.assertEqual(output_target.shape, output.shape)
        self.assertTensorAlmostEqual(output_target, output)

        ctc_target = torch.Tensor(
            [[[ 0.0033,  0.2913,  0.8034, -1.4238, -0.7482, -0.3856,  0.7473],
              [ 0.6427,  0.4159,  0.4899, -0.7912, -0.8450, -0.2153,  0.8037],
              [-0.1336, -0.1833,  0.5770, -1.3835, -1.0891, -0.1040,  0.6561],
              [-0.0868, -0.4087,  0.3973, -1.1419, -0.8750, -0.2796,  0.3531]],
             [[-0.3716,  0.0621,  0.4335, -0.5529, -0.6686,  0.0150,  0.0433],
              [ 0.6138,  0.6015,  0.4787, -0.9206, -0.7118, -0.4552,  1.0307],
              [ 0.1597,  0.5355,  0.5893, -0.2680, -0.3274,  0.0568,  0.2338],
              [ 0.5119,  0.9416,  1.1089, -0.6579, -0.8418,  0.0911,  1.0597]]]
        )
        self.assertEqual(ctc_target.shape, ctc_output.shape)
        self.assertTensorAlmostEqual(ctc_target, ctc_output)

        greedy_predictions = output.argmax(-1)
        expect_predictions = output_target.argmax(-1)
        self.assertTensorEqual(expect_predictions, greedy_predictions)
        states_target = torch.Tensor(
            [[[ 0.8879, -0.0974,  0.2107, -0.1444, -0.8334,  0.2333,
                0.3791, -0.1718,  0.4554, -0.1387, -0.4366, -0.4070],
              [ 0.8861,  0.0317,  0.1932, -0.1388, -0.6818,  0.2761,
                0.4604, -0.1886,  0.2083, -0.2472, -0.3908, -0.4119],
              [ 0.8850,  0.1744,  0.1932, -0.1359, -0.7000,  0.1509,
                0.5155, -0.3028,  0.4749, -0.4031, -0.4116, -0.4232],
              [ 0.9332,  0.1781,  0.1886, -0.1190, -0.6585,  0.1989,
                0.4359, -0.0950,  0.4830, -0.3038, -0.4343, -0.4058],
              [ 0.9268,  0.1882,  0.1935, -0.1103, -0.7240,  0.2795,
                0.4046, -0.1228,  0.4948, -0.4201, -0.4540, -0.3954]],
             [[ 0.9069, -0.0205,  0.2108, -0.1798, -0.6133,  0.2057,
                0.4348, -0.1056,  0.2320, -0.2214, -0.4788, -0.3683],
              [ 0.7859,  0.0713,  0.2211, -0.1623, -0.7643,  0.3284,
                0.5379, 0.0280,  0.1829, -0.2277, -0.4563, -0.3623],
              [ 0.8589,  0.1788,  0.1915, -0.1582, -0.7833,  0.2731,
                0.5247, -0.0955,  0.3663, -0.5870, -0.3828, -0.3712],
              [ 0.8682,  0.2259,  0.1937, -0.1558, -0.7200,  0.2876,
                0.4841, -0.0981,  0.3786, -0.4813, -0.4016, -0.3817],
              [ 0.9696,  0.1600,  0.1794, -0.1659, -0.5682,  0.1013,
                0.2791, -0.0022,  0.4784, -0.4591, -0.4281, -0.3789]]])
        self.assertEqual(states_target.shape, states.shape)
        self.assertTensorAlmostEqual(states_target, states)

    def test_transformer_decoder_layers(self):
        vocab_size = 7

        decoder = TransformerDecoder(
            num_layers=self.num_layers, num_heads=self.num_heads,
            hidden_size=self.hidden_size, ff_size=self.ff_size,
            dropout=self.dropout, vocab_size=vocab_size)

        self.assertEqual(len(decoder.layers), self.num_layers)

        for layer in decoder.layers:
            self.assertTrue(isinstance(layer, TransformerDecoderLayer))
            self.assertTrue(hasattr(layer, "src_trg_att"))
            self.assertTrue(hasattr(layer, "trg_trg_att"))
            self.assertTrue(hasattr(layer, "feed_forward"))
            self.assertEqual(layer.size, self.hidden_size)
            self.assertEqual(
                layer.feed_forward.pwff_layer[0].in_features, self.hidden_size)
            self.assertEqual(
                layer.feed_forward.pwff_layer[0].out_features, self.ff_size)
