# coding: utf-8
"""
Training module
"""

import argparse
import time
import shutil
from typing import List
import logging
import os
import sys
import collections
from collections import defaultdict
import pathlib
import numpy as np

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from torchtext.data import Dataset

from joeynmt.constants import UNK_TOKEN
from joeynmt.model import build_model
from joeynmt.batch import Batch, SpeechBatch
from joeynmt.helpers import load_config, log_cfg, \
    store_attention_plots, load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, delete_ckpt, \
    latest_checkpoint_update, ConfigurationError
from joeynmt.data_augmentation import Sampler, SpecAugment, \
    AlignedMasking, AudioDictAugment, MaskedLMAugment, SubSequencer
from joeynmt.model import Model, _DataParallel
from joeynmt.prediction import validate_on_data
from joeynmt.loss import XentLoss, XentCTCLoss
from joeynmt.data import load_data, make_data_iter
from joeynmt.builders import build_optimizer, build_scheduler, \
    build_gradient_clipper
from joeynmt.prediction import test
from joeynmt.metrics import EvaluationTokenizer

# for fp16 training
try:
    from apex import amp
    amp.register_half_function(torch, "einsum")
except ImportError as no_apex:
    # error handling in TrainManager object construction
    pass

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes
class TrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""
    # pylint: disable=too-many-branches,too-many-statements
    def __init__(self, model: Model, config: dict,
                 batch_class: Batch = Batch) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.

        :param model: torch module defining the model
        :param config: dictionary containing the training configurations
        :param batch_class: batch class to encapsulate the torch class
        """
        train_config = config["training"]
        self.task = config["data"].get("task", "MT")
        if self.task not in {"MT", "s2t"}:
            raise ConfigurationError("Invalid setting for `task`. "
                                     "Valid options: 'MT', 's2t'.")
        self.batch_class = batch_class

        # files for logging and storing
        self.model_dir = train_config["model_dir"]
        assert os.path.exists(self.model_dir)

        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = os.path.join(self.model_dir, "validations.txt")
        self.tb_writer = SummaryWriter(
            log_dir=os.path.join(self.model_dir, "tensorboard"))

        # model
        self.model = model
        self.model.log_parameters_list()

        # objective
        label_smoothing = train_config.get("label_smoothing", 0.0)
        if train_config.get("loss", "crossentropy") == "crossentropy-ctc":
            ctc_weight = train_config.get("ctc_weight", 0.3)
            loss_function = XentCTCLoss(pad_index=self.model.pad_index,
                                        bos_index=self.model.bos_index,
                                        smoothing=label_smoothing,
                                        ctc_weight=ctc_weight)
        else:
            loss_function = XentLoss(pad_index=self.model.pad_index,
                                     smoothing=label_smoothing)
        self.model.loss_function = loss_function
        if not self.model.loss_function.require_ctc_layer:
            self.model.decoder.ctc_output_layer = None
        logger.info(self.model)

        self.normalization = train_config.get("normalization", "batch")
        if self.normalization not in ["batch", "tokens", "none"]:
            raise ConfigurationError("Invalid normalization option."
                                     "Valid options: "
                                     "'batch', 'tokens', 'none'.")

        # optimization
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)

        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(config=train_config,
                                         parameters=model.parameters())

        # validation & early stopping
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.log_valid_sents = train_config.get("print_valid_sents", [0, 1, 2])
        self.save_latest_checkpoint = train_config.get("save_latest_ckpt", True)
        maxlen = train_config.get("keep_last_ckpts", 5)
        self.ckpt_queue = collections.deque(
            maxlen=maxlen if maxlen > 0 else None)
        metrics = train_config.get("eval_metrics", "").split(",")
        self.eval_metrics = [s.strip().lower() for s in metrics
                             if len(s.strip()) > 0]
        #backward compatibility
        if len(self.eval_metrics) == 0 \
                and "eval_metric" in train_config.keys():
            self.eval_metrics = [train_config.get("eval_metric",
                                                  "bleu").strip().lower()]
        assert len(self.eval_metrics) > 0
        for eval_metric in self.eval_metrics:
            if eval_metric not in ['bleu', 'chrf', 'token_accuracy',
                                   'sequence_accuracy', 'wer']:
                raise ConfigurationError("Invalid setting for 'eval_metric', "
                                         "valid options: 'bleu', 'chrf', "
                                         "'token_accuracy', "
                                         "'sequence_accuracy', 'wer'.")
        self.early_stopping_metric = train_config.get("early_stopping_metric",
                                                      "eval_metric")

        # early_stopping_metric decides on how to find the early stopping point:
        # ckpts are written when there's a new high/low score for this metric.
        # If we schedule after BLEU/chrf/accuracy, we want to maximize the
        # score, else we want to minimize it.
        if self.early_stopping_metric in ["ppl", "loss"]:
            self.minimize_metric = True
        elif self.early_stopping_metric in ["acc"]:
            self.minimize_metric = False
        elif self.early_stopping_metric == "eval_metric":
            if self.eval_metrics[0] in [
                    "bleu", "chrf", "token_accuracy", "sequence_accuracy"
            ]:
                self.minimize_metric = False
            # eval metric that has to get minimized: wer
            else:
                self.minimize_metric = True
        else:
            raise ConfigurationError(
                "Invalid setting for 'early_stopping_metric', "
                "valid options: 'loss', 'ppl', 'acc', 'eval_metric'.")

        # eval options
        self.level = config["data"]["level"]
        if self.level not in ["word", "bpe", "char"]:
            raise ConfigurationError("Invalid segmentation level. "
                                     "Valid options: 'word', 'bpe', 'char'.")
        test_config = config["testing"]
        self.bpe_type = test_config.get("bpe_type", "subword-nmt")
        if self.bpe_type not in ["subword-nmt", "sentencepiece"]:
            raise ConfigurationError("Invalid bpe type. Valid options: "
                                     "'subword-nmt', 'sentencepiece'.")
        self.level = config["data"]["level"]
        if self.level not in ["word", "bpe", "char"]:
            raise ConfigurationError("Invalid segmentation level. "
                                     "Valid options: 'word', 'bpe', 'char'.")
        self.sacrebleu = {
            "remove_whitespace": True,
            "remove_punctuation": True,
            "tokenize": "13a",
            "tok_fun": lambda s: list(s) if self.level=="char" else s.split()}
        if "sacrebleu" in config["testing"].keys():
            self.sacrebleu["remove_whitespace"] = test_config["sacrebleu"] \
                .get("remove_whitespace", True)
            self.sacrebleu["remove_punctuation"] = test_config["sacrebleu"] \
                .get("remove_punctuation", True)
            self.sacrebleu["tokenize"] = test_config["sacrebleu"] \
                .get("tokenize", "13a")
        if "wer" in self.eval_metrics:
            eval_tokenizer = EvaluationTokenizer(
                tokenize=self.sacrebleu["tokenize"],
                lowercase=config["data"].get("lowercase", False),
                remove_punctuation=self.sacrebleu["tokenize"],
                level=self.level)
            self.sacrebleu["tok_fun"] = eval_tokenizer.tokenize

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"])

        # data & batch handling
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.max_updates = train_config.get("updates", np.inf)
        self.batch_size = train_config["batch_size"]
        # Placeholder so that we can use the train_iter in other functions.
        self.train_iter, self.train_iter_state = None, None
        # per-device batch_size = self.batch_size // self.n_gpu
        self.batch_type = train_config.get("batch_type", "sentence")
        self.eval_batch_size = train_config.get("eval_batch_size",
                                                self.batch_size)
        # per-device eval_batch_size = self.eval_batch_size // self.n_gpu
        self.eval_batch_type = train_config.get("eval_batch_type",
                                                self.batch_type)
        # gradient accumulation
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        # generation
        self.max_output_length = train_config.get("max_output_length", None)

        # CPU / GPU
        self.use_cuda = train_config["use_cuda"] and torch.cuda.is_available()
        self.n_gpu = torch.cuda.device_count() if self.use_cuda else 0
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.model.to(self.device)

        # fp16
        self.fp16 = train_config.get("fp16", False)
        if self.fp16:
            if 'apex' not in sys.modules:
                raise ImportError("Please install apex from "
                                  "https://www.github.com/nvidia/apex "
                                  "to use fp16 training.") from no_apex
            self.model, self.optimizer = amp.initialize(self.model,
                                                        self.optimizer,
                                                        opt_level='O1')
            # opt level: one of {"O0", "O1", "O2", "O3"}
            # see https://nvidia.github.io/apex/amp.html#opt-levels

        # data augmentation
        if self.task == "s2t":
            self.data_augmentation = config["data"].get("data_augmentation", "specaugment")
            self.max_src_length = config["data"]["max_feat_length"]
            self.max_trg_length = config["data"]["max_sent_length"]
            self.specaugment = None
            if "specaugment" in config["data"].keys():
                self.specaugment = SpecAugment(**config["data"]["specaugment"])
                logger.info(self.specaugment)

            self.sampler = None
            if "sampler" in config["data"].keys():
                self.mask_sampler = Sampler(config["data"]["sampler"], unk_token=UNK_TOKEN)
                logger.info(self.mask_sampler)

            self.aligned_masking = None
            if self.data_augmentation == "aligned_masking":
                self.aligned_masking = AlignedMasking()
                #logger.info(self.aligned_masking)

            self.audio_dict = None
            if self.data_augmentation in ["audio_dict_replacement", "aligned_audio_dict"] \
                    and "audio_dict" in config["data"].keys():
                self.audio_dict = AudioDictAugment(config["data"]["audio_dict"])
                logger.info(self.audio_dict)

            self.masked_lm = None
            if self.data_augmentation in ["trg_replacement", "aligned_audio_dict"] \
                    and "masked_language_model" in config["data"].keys():
                self.masked_lm = MaskedLMAugment(config["data"]["masked_language_model"],
                                                 config["data"]['lowercase'], self.model.trg_vocab.stoi,
                                                 self.model.unk_index, self.model.bos_index,
                                                 self.model.eos_index, self.device)
                logger.info(self.masked_lm)

            self.subsequencer = None
            if "subsequence" in config["data"].keys():
                self.subsequencer = SubSequencer(**config["data"]["subsequence"])
                logger.info(self.subsequencer)

        # initialize training statistics
        self.stats = self.TrainStatistics(
            steps=0,
            is_min_lr=False,
            is_max_update=False,
            total_tokens=0,
            total_seqs=0,
            total_batches=0,
            best_ckpt_iter=0,
            best_ckpt_score=np.inf if self.minimize_metric else -np.inf,
            minimize_metric=self.minimize_metric)

        # model parameters
        if "load_model" in train_config.keys():
            self.init_from_checkpoint(
                train_config["load_model"],
                reset_best_ckpt=train_config.get("reset_best_ckpt", False),
                reset_scheduler=train_config.get("reset_scheduler", False),
                reset_optimizer=train_config.get("reset_optimizer", False),
                reset_iter_state=train_config.get("reset_iter_state", False))

        # multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1:
            self.model = _DataParallel(self.model)

        ####### end __init__() #######

    def _save_checkpoint(self, new_best: bool = True) -> None:
        """
        Save the model's current parameters and the training state to a
        checkpoint.
        The training state contains the total number of training steps,
        the total number of training tokens,
        the best checkpoint score and iteration so far,
        and optimizer and scheduler states.

        :param new_best: This boolean signals which symlink we will use for the
          new checkpoint. If it is true, we update best.ckpt, else latest.ckpt.
        """
        model_path = os.path.join(self.model_dir, f"{self.stats.steps}.ckpt")
        logger.info("Saving new checkpoint: %s", model_path)
        model_state_dict = self.model.module.state_dict() \
            if isinstance(self.model, torch.nn.DataParallel) \
            else self.model.state_dict()
        state = {
            "steps":
            self.stats.steps,
            "total_tokens":
            self.stats.total_tokens,
            "best_ckpt_score":
            self.stats.best_ckpt_score,
            "best_ckpt_iteration":
            self.stats.best_ckpt_iter,
            "model_state":
            model_state_dict,
            "optimizer_state":
            self.optimizer.state_dict(),
            "scheduler_state":
            self.scheduler.state_dict() if self.scheduler is not None else None,
            'amp_state':
            amp.state_dict() if self.fp16 else None,
            "train_iter_state":
            self.train_iter.state_dict()
        }
        torch.save(state, model_path)
        symlink_target = f"{self.stats.steps}.ckpt"
        if new_best:
            if self.ckpt_queue.maxlen is not None \
                    and len(self.ckpt_queue) == self.ckpt_queue.maxlen:
                to_delete = self.ckpt_queue.popleft()  # delete oldest ckpt
                delete_ckpt(to_delete)

            self.ckpt_queue.append(model_path)

            best_path = os.path.join(self.model_dir, "best.ckpt")
            try:
                # create/modify symbolic link for best checkpoint
                symlink_update(symlink_target, best_path)
            except OSError:
                # overwrite best.ckpt
                torch.save(state, best_path)

        if self.save_latest_checkpoint:
            last_path = os.path.join(self.model_dir, "latest.ckpt")
            previous_path = latest_checkpoint_update(symlink_target, last_path)
            # If the last ckpt is in the ckpt_queue, we don't want to delete it.
            if self.ckpt_queue.maxlen is not None:
                can_delete = True
                for ckpt_path in self.ckpt_queue:
                    if pathlib.Path(ckpt_path).resolve() == previous_path:
                        can_delete = False
                        break
                if can_delete and previous_path is not None:
                    delete_ckpt(previous_path)

    def init_from_checkpoint(self,
                             path: str,
                             reset_best_ckpt: bool = False,
                             reset_scheduler: bool = False,
                             reset_optimizer: bool = False,
                             reset_iter_state: bool = False) -> None:
        """
        Initialize the trainer from a given checkpoint file.

        This checkpoint file contains not only model parameters, but also
        scheduler and optimizer states, see `self._save_checkpoint`.

        :param path: path to checkpoint
        :param reset_best_ckpt: reset tracking of the best checkpoint,
                                use for domain adaptation with a new dev
                                set or when using a new metric for fine-tuning.
        :param reset_scheduler: reset the learning rate scheduler, and do not
                                use the one stored in the checkpoint.
        :param reset_optimizer: reset the optimizer, and do not use the one
                                stored in the checkpoint.
        :param reset_iter_state: reset the sampler's internal state and do not
                                use the one stored in the checkpoint.
        """
        logger.info("Loading model from %s", path)
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        else:
            logger.info("Reset optimizer.")

        if not reset_scheduler:
            if model_checkpoint["scheduler_state"] is not None and \
                    self.scheduler is not None:
                self.scheduler.load_state_dict(
                    model_checkpoint["scheduler_state"])
        else:
            logger.info("Reset scheduler.")

        # restore counts
        self.stats.steps = model_checkpoint["steps"]
        self.stats.total_tokens = model_checkpoint["total_tokens"]
        self.stats.total_seqs = model_checkpoint["total_seqs"] if "total_seqs" in model_checkpoint.keys() else 0
        self.stats.total_batches = model_checkpoint["total_batches"] if "total_batches" in model_checkpoint.keys() else 0

        if not reset_best_ckpt:
            self.stats.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.stats.best_ckpt_iter = model_checkpoint["best_ckpt_iteration"]
        else:
            logger.info("Reset tracking of the best checkpoint.")

        if (not reset_iter_state
                and model_checkpoint.get('train_iter_state', None) is not None):
            self.train_iter_state = model_checkpoint["train_iter_state"]

        # move parameters to cuda
        if self.use_cuda:
            self.model.to(self.device)

        # fp16
        if self.fp16 and model_checkpoint.get("amp_state", None) is not None:
            amp.load_state_dict(model_checkpoint['amp_state'])

    # pylint: disable=unnecessary-comprehension
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    def train_and_validate(self, train_data: Dataset, valid_data: Dataset) \
            -> None:
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """
        self.train_iter = make_data_iter(train_data,
                                         batch_size=self.batch_size,
                                         batch_type=self.batch_type,
                                         train=True,
                                         shuffle=self.shuffle)

        if self.train_iter_state is not None:
            self.train_iter.load_state_dict(self.train_iter_state)

        #################################################################
        # simplify accumulation logic:
        #################################################################
        # for epoch in range(epochs):
        #     self.model.zero_grad()
        #     epoch_loss = 0.0
        #     batch_loss = 0.0
        #     for i, batch in enumerate(iter(self.train_iter)):
        #
        #         # gradient accumulation:
        #         # loss.backward() inside _train_step()
        #         batch_loss += self._train_step(inputs)
        #
        #         if (i + 1) % self.batch_multiplier == 0:
        #             self.optimizer.step()     # update!
        #             self.model.zero_grad()    # reset gradients
        #             self.steps += 1           # increment counter
        #
        #             epoch_loss += batch_loss  # accumulate batch loss
        #             batch_loss = 0            # reset batch loss
        #
        #     # leftovers are just ignored.
        #################################################################

        logger.info(
            "Train stats:\n"
            "\tdevice: %s\n"
            "\tn_gpu: %d\n"
            "\t16-bits training: %r\n"
            "\tgradient accumulation: %d\n"
            "\tbatch size per device: %d\n"
            "\ttotal batch size (w. parallel & accumulation): %d", self.device,
            self.n_gpu, self.fp16, self.batch_multiplier, self.batch_size //
            self.n_gpu if self.n_gpu > 1 else self.batch_size,
            self.batch_size * self.batch_multiplier)

        for epoch_no in range(self.epochs):
            logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.train()

            # Reset statistics for each epoch.
            start = time.time()
            total_valid_duration = 0
            start_tokens = self.stats.total_tokens
            self.model.zero_grad()
            epoch_loss = 0
            batch_loss = 0
            batch_acc = 0
            auxiliary_losses = defaultdict(float)

            for i, batch in enumerate(iter(self.train_iter)):
                # create a Batch object from torchtext batch
                kwargs = {}
                if self.task=="s2t":
                    kwargs["steps"] = self.stats.steps
                    kwargs["batch_count"] = i
                    kwargs["is_train"] = True
                    kwargs["max_src_length"] = self.max_src_length
                    kwargs["max_trg_length"] = self.max_trg_length
                    kwargs["specaugment"] = self.specaugment
                    kwargs["sampler"] = self.mask_sampler
                    kwargs["aligned_masking"] = self.aligned_masking
                    kwargs["audio_dict"] = self.audio_dict
                    kwargs["masked_lm"] = self.masked_lm
                batch = self.batch_class(batch, pad_index=self.model.pad_index,
                    bos_index=self.model.bos_index, eos_index=self.model.eos_index,
                    use_cuda=self.use_cuda, **kwargs)

                # get batch loss
                loss_dict = self._train_step(batch)
                batch_loss += loss_dict['loss']
                batch_acc += loss_dict['acc']
                for k, v in loss_dict.items():
                    if k != 'loss':
                        auxiliary_losses[k] += v

                # update!
                if (i + 1) % self.batch_multiplier == 0:
                    # clip gradients (in-place)
                    if self.clip_grad_fun is not None:
                        if self.fp16:
                            self.clip_grad_fun(
                                params=amp.master_params(self.optimizer))
                        else:
                            self.clip_grad_fun(params=self.model.parameters())

                    # make gradient step
                    self.optimizer.step()

                    # decay lr
                    if self.scheduler is not None \
                            and self.scheduler_step_at == "step":
                        self.scheduler.step(self.stats.steps)

                    # reset gradients
                    self.model.zero_grad()

                    # increment step counter
                    self.stats.steps += 1
                    if self.stats.steps >= self.max_updates:
                        self.stats.is_max_update = True

                    # log learning progress
                    if self.stats.steps % self.logging_freq == 0:
                        self.tb_writer.add_scalar("learning_rate/learning_rate",
                            self.optimizer.param_groups[0]["lr"],
                            self.stats.steps)
                        self.tb_writer.add_scalar("train/train_batch_loss",
                                                  batch_loss, self.stats.steps)
                        self.tb_writer.add_scalar("train/train_batch_acc",
                                                  batch_acc, self.stats.steps)
                        for k, v in auxiliary_losses.items():
                            self.tb_writer.add_scalar(f"train/train_batch_{k}",
                                                      v, self.stats.steps)
                        elapsed = time.time() - start - total_valid_duration
                        elapsed_tokens = self.stats.total_tokens - start_tokens
                        logger.info(
                            "Epoch %3d, Step: %8d, Batch Loss: %12.6f, "
                            "Batch Acc: %.6f, Tokens per Sec: %8.0f, Lr: %.6f",
                            epoch_no + 1, self.stats.steps, batch_loss,
                            batch_acc, elapsed_tokens / elapsed,
                            self.optimizer.param_groups[0]["lr"])
                        start = time.time()
                        total_valid_duration = 0
                        start_tokens = self.stats.total_tokens

                    # Only add complete loss of full mini-batch to epoch_loss
                    epoch_loss += batch_loss    # accumulate epoch_loss
                    batch_loss = 0              # rest batch_loss
                    batch_acc = 0               # rest batch_acc
                    auxiliary_losses = defaultdict(float)

                    # validate on the entire dev set
                    if self.stats.steps % self.validation_freq == 0:
                        valid_duration = self._validate(valid_data, epoch_no)
                        total_valid_duration += valid_duration

                if self.stats.is_min_lr or self.stats.is_max_update:
                    break

            # pylint: disable=no-else-break
            if self.stats.is_min_lr:
                logger.info(
                    'Training ended since minimum lr %f was reached.',
                    self.learning_rate_min)
                break
            elif self.stats.is_max_update:
                logger.info('Training ended since maximum num. of '
                            'updates %d was reached.', self.max_updates)
                break

            logger.info('Epoch %3d: total training loss %.2f, num updates: %6d, '
                        'num instances: %6d, num tokens: %6d, num batches: %6d',
                        epoch_no + 1, epoch_loss, self.stats.steps,
                        self.stats.total_seqs, self.stats.total_tokens,
                        self.stats.total_batches)

            if self.masked_lm is not None:
                self.masked_lm.reset_stats()
        else:
            logger.info('Training ended after %3d epochs.', epoch_no + 1)
        metric_name = self.eval_metrics[0] \
            if self.early_stopping_metric == "eval_metric" \
            else self.early_stopping_metric
        logger.info('Best validation result (greedy) at step %8d: %6.2f %s.',
                    self.stats.best_ckpt_iter, self.stats.best_ckpt_score,
                    metric_name)

        self.tb_writer.close()  # close Tensorboard writer

    def _train_step(self, batch: Batch) -> Tensor:
        """
        Train the model on one batch: Compute the loss.

        :param batch: training batch
        :return: loss for batch (sum)
        """
        # reactivate training
        self.model.train()

        # get loss
        auxiliary_losses = {}
        batch_loss, nll_loss, ctc_loss, correct_tokens = self.model(
            return_type="loss", **vars(batch))
        if torch.is_tensor(nll_loss): # nll_loss is not None
            auxiliary_losses['nll_loss'] = nll_loss
        if torch.is_tensor(ctc_loss): # ctc_loss is not None
            auxiliary_losses['ctc_loss'] = ctc_loss

        # sum multi-gpu losses
        if self.n_gpu > 1:
            batch_loss = batch_loss.sum()
            correct_tokens = correct_tokens.sum()
            for l in  auxiliary_losses.keys():
                auxiliary_losses[l] = auxiliary_losses[l].sum()

        # normalize batch loss
        if self.normalization == "batch":
            normalizer = batch.nseqs
        elif self.normalization == "tokens":
            normalizer = batch.ntokens
        elif self.normalization == "none":
            normalizer = 1
        else:
            raise NotImplementedError("Only normalize by 'batch' or 'tokens' "
                                      "or summation of loss 'none' implemented")

        norm_batch_loss = batch_loss / normalizer
        for l in  auxiliary_losses.keys():
            auxiliary_losses[l] = auxiliary_losses[l] / normalizer

        if self.n_gpu > 1:
            norm_batch_loss = norm_batch_loss / self.n_gpu
            correct_tokens = correct_tokens / self.n_gpu
            for l in  auxiliary_losses.keys():
                auxiliary_losses[l] = auxiliary_losses[l] / self.n_gpu

        if self.batch_multiplier > 1:
            norm_batch_loss = norm_batch_loss / self.batch_multiplier
            for l in  auxiliary_losses.keys():
                auxiliary_losses[l] = auxiliary_losses[l] / self.batch_multiplier

        # accumulate gradients
        if self.fp16:
            with amp.scale_loss(norm_batch_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            norm_batch_loss.backward()

        # increment token counter
        self.stats.total_tokens += batch.ntokens
        self.stats.total_seqs += batch.nseqs
        self.stats.total_batches += 1

        ret = {'loss': norm_batch_loss.item(),
               'acc': (correct_tokens/batch.ntokens).item()}
        ret.update(auxiliary_losses)
        return ret

    def _validate(self, valid_data, epoch_no):
        valid_start_time = time.time()

        valid_scores, valid_loss, valid_ppl, valid_acc, valid_sources, \
        valid_sources_raw, valid_references, valid_hypotheses, \
        valid_hypotheses_raw, valid_attention_scores = \
            validate_on_data(
                batch_size=self.eval_batch_size,
                batch_class=self.batch_class,
                data=valid_data,
                eval_metrics=self.eval_metrics,
                level=self.level,
                model=self.model,
                use_cuda=self.use_cuda,
                max_output_length=self.max_output_length,
                compute_loss=True,
                beam_size=1,                # greedy validations
                batch_type=self.eval_batch_type,
                postprocess=True,           # always remove BPE for validation
                bpe_type=self.bpe_type,     # "subword-nmt" or "sentencepiece"
                sacrebleu=self.sacrebleu,   # sacrebleu options
                n_gpu=self.n_gpu,
                task=self.task
            )


        self.tb_writer.add_scalar(
            "valid/valid_loss", valid_loss, self.stats.steps)
        for eval_metric in self.eval_metrics:
            self.tb_writer.add_scalar(
                f"valid/valid_{eval_metric}", valid_scores[eval_metric],
                self.stats.steps)
        self.tb_writer.add_scalar(
            "valid/valid_ppl", valid_ppl, self.stats.steps)
        self.tb_writer.add_scalar(
            "valid/valid_acc", valid_acc, self.stats.steps)

        if self.early_stopping_metric == "loss":
            ckpt_score = valid_loss
        elif self.early_stopping_metric == "acc":
            ckpt_score = valid_acc
        elif self.early_stopping_metric in ["ppl", "perplexity"]:
            ckpt_score = valid_ppl
        else:   #if self.early_stopping_metric == "eval_metric"
            ckpt_score = valid_scores[self.eval_metrics[0]]

        if self.scheduler is not None \
                and self.scheduler_step_at == "validation":
            self.scheduler.step(ckpt_score)

        new_best = False
        if self.stats.is_best(ckpt_score):
            self.stats.best_ckpt_score = ckpt_score
            self.stats.best_ckpt_iter = self.stats.steps
            logger.info('Hooray! New best validation result [%s]!',
                self.eval_metrics[0] \
                if (self.early_stopping_metric == 'eval_metric') \
                else self.early_stopping_metric)
            new_best = True
            self._save_checkpoint(new_best)
        elif self.save_latest_checkpoint:
            self._save_checkpoint(new_best)

        # append to validation report
        self._add_report(valid_scores=valid_scores,
                         valid_loss=valid_loss,
                         valid_ppl=valid_ppl,
                         valid_acc=valid_acc,
                         eval_metrics=self.eval_metrics,
                         new_best=new_best)

        self._log_examples(
            sources_raw=[v for v in valid_sources_raw] \
                if self.task=="MT" else None,
            sources=valid_sources if self.task=="MT" else None,
            hypotheses_raw=valid_hypotheses_raw,
            hypotheses=valid_hypotheses,
            references=valid_references)

        valid_duration = time.time() - valid_start_time
        score_str = ""
        for i, eval_metric in enumerate(self.eval_metrics):
            score_str += "" if i==0 else ", "
            score_str += "%s: %6.2f" % (eval_metric, valid_scores[eval_metric])
        logger.info(
            'Validation result (greedy) at epoch %3d, '
            'step %8d: %s, loss: %8.4f, ppl: %8.4f, '
            'acc: %.4f, duration: %.4fs', epoch_no + 1, self.stats.steps,
            score_str, valid_loss, valid_ppl, valid_acc, valid_duration)

        # store validation set outputs
        self._store_outputs(valid_hypotheses)

        # store attention plots for selected valid sentences
        if valid_attention_scores:
            store_attention_plots(
                attentions=valid_attention_scores,
                targets=valid_hypotheses_raw,
                sources=[s for s in valid_data.src] \
                    if self.task== "MT" else None,
                indices=self.log_valid_sents,
                output_prefix=os.path.join(
                    self.model_dir, f"att.{self.stats.steps}"),
                tb_writer=self.tb_writer,
                steps=self.stats.steps)

        return valid_duration

    def _add_report(self,
                    valid_scores: float,
                    valid_ppl: float,
                    valid_loss: float,
                    valid_acc: float,
                    eval_metrics: List[str],
                    new_best: bool = False) -> None:
        """
        Append a one-line report to validation logging file.

        :param valid_scores: (dict) validation evaluation scores [eval_metrics]
        :param valid_ppl: validation perplexity
        :param valid_loss: validation loss (sum over whole validation set)
        :param valid_acc: validation token accuracy before decoding
        :param eval_metrics: evaluation metric, e.g. "bleu"
        :param new_best: whether this is a new best model
        """
        #current_lr = -1
        # ignores other param groups for now
        #for param_group in self.optimizer.param_groups:
        #    current_lr = param_group['lr']
        current_lr = self.optimizer.param_groups[0]['lr']

        if current_lr < self.learning_rate_min:
            self.stats.is_min_lr = True

        with open(self.valid_report_file, 'a') as opened_file:
            score_str = ""
            for eval_metric in eval_metrics:
                score_str += "{}: {:.5f}\t".format(eval_metric,
                                                   valid_scores[eval_metric])
            opened_file.write(
                "Steps: {}\tLoss: {:.5f}\tPPL: {:.5f}\tAcc: {:.5f}\t"
                "{}LR: {:.8f}\t{}\n".format(
                    self.stats.steps, valid_loss, valid_ppl, valid_acc,
                    score_str, current_lr, "*" if new_best else ""))

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad,
                                  self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info("Total params: %d", n_params)
        trainable_params = [
            n for (n, p) in self.model.named_parameters() if p.requires_grad
        ]
        logger.debug("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params

    def _log_examples(self,
                      hypotheses: List[str],
                      references: List[str],
                      sources: List[str] = None,
                      hypotheses_raw: List[List[str]] = None,
                      references_raw: List[List[str]] = None,
                      sources_raw: List[List[str]] = None,) -> None:
        """
        Log a the first `self.log_valid_sents` sentences from given examples.

        :param hypotheses: decoded hypotheses (list of strings)
        :param references: decoded references (list of strings)
        :param sources: decoded sources (list of strings)
        :param hypotheses_raw: raw hypotheses (list of list of tokens)
        :param references_raw: raw references (list of list of tokens)
        :param sources_raw: raw sources (list of list of tokens)
        """
        for p in self.log_valid_sents:

            if p >= len(hypotheses):
                continue

            logger.info("Example #%d", p)

            if sources_raw is not None:
                logger.debug("\tRaw source:     %s", sources_raw[p])
            if references_raw is not None:
                logger.debug("\tRaw reference:  %s", references_raw[p])
            if hypotheses_raw is not None:
                logger.debug("\tRaw hypothesis: %s", hypotheses_raw[p])
            if sources is not None:
                logger.info("\tSource:     %s", sources[p])
            logger.info("\tReference:  %s", references[p])
            logger.info("\tHypothesis: %s", hypotheses[p])

    def _store_outputs(self, hypotheses: List[str]) -> None:
        """
        Write current validation outputs to file in `self.model_dir.`

        :param hypotheses: list of strings
        """
        current_valid_output_file = os.path.join(self.model_dir,
                                                 f"{self.stats.steps}.hyps")
        with open(current_valid_output_file, 'w') as opened_file:
            for hyp in hypotheses:
                opened_file.write("{}\n".format(hyp))

    class TrainStatistics:
        def __init__(self,
                     steps: int = 0,
                     is_min_lr: bool = False,
                     is_max_update: bool = False,
                     total_tokens: int = 0,
                     total_batches: int = 0,
                     total_seqs: int = 0,
                     best_ckpt_iter: int = 0,
                     best_ckpt_score: float = np.inf,
                     minimize_metric: bool = True) -> None:
            # global update step counter
            self.steps = steps
            # stop training if this flag is True
            # by reaching learning rate minimum
            self.is_min_lr = is_min_lr
            # stop training if this flag is True
            # by reaching max num of updates
            self.is_max_update = is_max_update
            # number of total tokens seen so far
            self.total_tokens = total_tokens
            self.total_batches = total_batches
            self.total_seqs = total_seqs
            # store iteration point of best ckpt
            self.best_ckpt_iter = best_ckpt_iter
            # initial values for best scores
            self.best_ckpt_score = best_ckpt_score
            # minimize or maximize score
            self.minimize_metric = minimize_metric

        def is_best(self, score):
            if self.minimize_metric:
                is_best = score < self.best_ckpt_score
            else:
                is_best = score > self.best_ckpt_score
            return is_best


def train(cfg_file: str) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    """
    # read config file
    cfg = load_config(cfg_file)
    log_cfg(cfg) # write all entries of config to the log

    task = cfg["data"]["task"]  # "MT" or "s2t"

    # make logger
    model_dir = make_model_dir(cfg["training"]["model_dir"],
                               overwrite=cfg["training"].get(
                                   "overwrite", False))
    _ = make_logger(model_dir, mode="train")  # version string returned
    # TODO: save version number in model checkpoints

    # store copy of original training config in model dir
    shutil.copy2(cfg_file, os.path.join(model_dir, "config.yaml"))

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    # load the data
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(
        data_cfg=cfg["data"])

    # build an encoder-decoder model
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, config=cfg,
                           batch_class=SpeechBatch if task=="s2t" else Batch)

    # store the vocabs
    if task == "MT":
        src_vocab_file = os.path.join(
            cfg["training"]["model_dir"], "src_vocab.txt")
        src_vocab.to_file(src_vocab_file)
    trg_vocab_file = os.path.join(cfg["training"]["model_dir"], "trg_vocab.txt")
    trg_vocab.to_file(trg_vocab_file)

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)

    # predict with the best model on validation and test
    # (if test data is available)
    ckpt = os.path.join(model_dir, f"{trainer.stats.best_ckpt_iter}.ckpt")
    output_name = "{:08d}.hyps".format(trainer.stats.best_ckpt_iter)
    output_path = os.path.join(model_dir, output_name)
    datasets_to_test = {
        "dev": dev_data,
        "test": test_data,
        "src_vocab": src_vocab,
        "trg_vocab": trg_vocab
    }
    test(cfg_file,
         ckpt=ckpt,
         output_path=output_path,
         datasets=datasets_to_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Joey-NMT')
    parser.add_argument("config",
                        default="configs/default.yaml",
                        type=str,
                        help="Training configuration file (yaml).")
    args = parser.parse_args()
    train(cfg_file=args.config)
