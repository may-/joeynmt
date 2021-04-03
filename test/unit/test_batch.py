from test.unit.test_helpers import TensorTestCase

import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, \
    SequentialSampler

from joeynmt.batch import Batch
from joeynmt.data import TokenBatchSampler, load_data, make_data_iter


class TestBatch(TensorTestCase):

    def setUp(self):
        # minimal data config
        data_cfg = {"src": "de", "trg": "en",
                    "train": "test/data/toy/train",
                    "dev": "test/data/toy/dev",
                    "level": "char",
                    "lowercase": True,
                    "max_sent_length": 20}

        # load the data
        self.src_vocab, self.trg_vocab, self.train_data, self.dev_data, _ \
            = load_data(data_cfg)
        self.pad_index = self.trg_vocab.pad_index
        # random seeds
        self.seed = 42

    def testBatchTrainIterator(self):

        batch_size = 4
        self.assertEqual(len(self.train_data), 27)

        # make data iterator
        train_iter = make_data_iter(dataset=self.train_data,
                                    batch_size=batch_size,
                                    batch_type="sentence",
                                    shuffle=True,
                                    seed=self.seed,
                                    pad_index=self.pad_index,
                                    device=torch.device("cpu"),
                                    num_workers=0)
        self.assertTrue(isinstance(train_iter, DataLoader))
        self.assertEqual(train_iter.batch_sampler.batch_size, batch_size)
        self.assertTrue(isinstance(train_iter.batch_sampler, BatchSampler))
        self.assertTrue(isinstance(train_iter.batch_sampler.sampler,
                                   RandomSampler))  # shuffle=True
        initial_seed = train_iter.batch_sampler.sampler.generator.initial_seed()
        self.assertEqual(initial_seed, self.seed)

        expected_src0 = torch.LongTensor(
            [[2, 24, 10, 4, 5, 18, 4, 7, 17, 11, 8, 11, 5, 14, 8, 7, 25, 3, 1,
              1, 1],
             [2, 19, 15, 8, 32, 9, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [2, 14, 8, 6, 15, 4, 9, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [2, 20, 19, 12, 27, 7, 22, 5, 6, 10, 17, 11, 13, 5, 20, 19, 12, 27,
              7, 9, 3]])
        expected_src0_len = torch.LongTensor([18, 7, 8, 21])
        expected_trg0 = torch.LongTensor(
            [[7, 5, 25, 4, 19, 14, 19, 4, 8, 7, 14, 12, 4, 7, 6, 18, 18, 11, 10,
              23, 3],
             [5, 17, 6, 13, 9, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1],
             [8, 7, 6, 10, 17, 4, 13, 5, 15, 9, 3, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1],
             [18, 5, 16, 5, 15, 12, 22, 4, 10, 5, 10, 18, 5, 16, 5, 15, 12, 9,
              3, 1, 1]])
        expected_trg0_len = torch.LongTensor([22, 7, 12, 20])

        total_samples = 0
        for b in train_iter:
            self.assertTrue(isinstance(b, Batch))
            if total_samples == 0:
                self.assertTensorEqual(b.src, expected_src0)
                self.assertTensorEqual(b.src_length, expected_src0_len)
                self.assertTensorEqual(b.trg, expected_trg0)
                self.assertTensorEqual(b.trg_length, expected_trg0_len)
            total_samples += b.nseqs
            self.assertLessEqual(b.nseqs, batch_size)
        self.assertEqual(total_samples, len(self.train_data))

    def testTokenBatchTrainIterator(self):

        batch_size = 50  # num of tokens in one batch
        self.assertEqual(len(self.train_data), 27)

        # make data iterator
        train_iter = make_data_iter(dataset=self.train_data,
                                    batch_size=batch_size,
                                    batch_type="token",
                                    shuffle=True,
                                    seed=self.seed,
                                    pad_index=self.pad_index,
                                    device=torch.device("cpu"),
                                    num_workers=0)
        self.assertTrue(isinstance(train_iter, DataLoader))
        self.assertEqual(train_iter.batch_sampler.batch_size, batch_size)
        self.assertTrue(isinstance(train_iter.batch_sampler, TokenBatchSampler))
        self.assertTrue(isinstance(train_iter.batch_sampler.sampler,
                                   RandomSampler))  # shuffle=True
        initial_seed = train_iter.batch_sampler.sampler.generator.initial_seed()
        self.assertEqual(initial_seed, self.seed)

        expected_src0 = torch.LongTensor(
            [[2, 24, 10, 4, 5, 18, 4, 7, 17, 11, 8, 11, 5, 14, 8, 7, 25, 3],
             [2, 19, 15, 8, 32, 9, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [2, 14, 8, 6, 15, 4, 9, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        expected_src0_len = torch.LongTensor([18, 7, 8])
        expected_trg0 = torch.LongTensor(
            [[7, 5, 25, 4, 19, 14, 19, 4, 8, 7, 14, 12, 4, 7, 6, 18, 18, 11, 10,
              23, 3],
             [5, 17, 6, 13, 9, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [8, 7, 6, 10, 17, 4, 13, 5, 15, 9, 3, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1]])
        expected_trg0_len = torch.LongTensor([22, 7, 12])

        total_tokens = 0
        for b in train_iter:
            self.assertTrue(isinstance(b, Batch))
            if total_tokens == 0:
                self.assertTensorEqual(b.src, expected_src0)
                self.assertTensorEqual(b.src_length, expected_src0_len)
                self.assertTensorEqual(b.trg, expected_trg0)
                self.assertTensorEqual(b.trg_length, expected_trg0_len)
            total_tokens += b.ntokens
        self.assertEqual(total_tokens, 387)

    def testBatchDevIterator(self):

        batch_size = 3
        self.assertEqual(len(self.dev_data), 20)

        # make data iterator
        dev_iter = make_data_iter(dataset=self.dev_data,
                                  batch_size=batch_size,
                                  batch_type="sentence",
                                  shuffle=False,
                                  pad_index=self.pad_index,
                                  device=torch.device("cpu"),
                                  num_workers=0)
        self.assertTrue(isinstance(dev_iter, DataLoader))
        self.assertEqual(dev_iter.batch_sampler.batch_size, batch_size)
        self.assertTrue(isinstance(dev_iter.batch_sampler, BatchSampler))
        self.assertTrue(isinstance(dev_iter.batch_sampler.sampler,
                                   SequentialSampler))  # shuffle=False

        expected_src0 = torch.LongTensor(
            [[2, 29, 8, 5, 22, 5, 8, 16, 7, 19, 5, 22, 5, 24, 8, 7, 5, 7, 19,
              16, 16, 5, 31, 10, 19, 11, 8, 17, 15, 10, 6, 18, 5, 7, 4, 10, 6,
              5, 25, 3],
             [2, 10, 17, 11, 5, 28, 12, 4, 23, 4, 5, 0, 10, 17, 11, 5, 22, 5,
              14, 8, 7, 7, 5, 10, 17, 11, 5, 14, 8, 5, 31, 10, 6, 5, 9, 3, 1,
              1, 1, 1],
             [2, 29, 8, 5, 22, 5, 18, 23, 13, 4, 6, 5, 13, 8, 18, 5, 9, 3, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        expected_src0_len = torch.LongTensor([40, 36, 18])
        expected_trg0 = torch.LongTensor(
            [[13, 11, 12, 4, 22, 4, 12, 5, 4, 22, 4, 25, 7, 6, 8, 4, 14, 12, 4,
              24, 14, 5, 7, 6, 26, 17, 14, 10, 20, 4, 23, 3],
             [14, 0, 28, 4, 7, 6, 18, 18, 13, 4, 8, 5, 4, 24, 11, 4, 7, 11, 16,
              11, 4, 9, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [13, 11, 12, 4, 22, 4, 7, 11, 27, 27, 5, 4, 9, 3, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        expected_trg0_len = torch.LongTensor([33, 24, 15])

        total_samples = 0
        for b in dev_iter:
            self.assertTrue(isinstance(b, Batch))

            # test the sorting by src length
            before_sort = b.src_length
            b.sort_by_src_length()
            after_sort = b.src_length
            self.assertTensorEqual(torch.sort(before_sort, descending=True)[0],
                                   after_sort)

            if total_samples == 0:
                self.assertTensorEqual(b.src, expected_src0)
                self.assertTensorEqual(b.src_length, expected_src0_len)
                self.assertTensorEqual(b.trg, expected_trg0)
                self.assertTensorEqual(b.trg_length, expected_trg0_len)
            total_samples += b.nseqs
            self.assertLessEqual(b.nseqs, batch_size)
        self.assertEqual(total_samples, len(self.dev_data))
