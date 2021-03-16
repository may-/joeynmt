# coding: utf-8
"""
Training module
"""

import argparse
import time
import shutil
from typing import List
import logging
import sys
import heapq
from pathlib import Path
import numpy as np

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.multiprocessing import cpu_count

from joeynmt.model import build_model
from joeynmt.batch import Batch
from joeynmt.helpers import load_config, log_cfg, store_attention_plots, \
    load_checkpoint, make_model_dir, make_logger, set_seed, symlink_update, \
    delete_ckpt, ConfigurationError
from joeynmt.model import Model, _DataParallel
from joeynmt.prediction import validate_on_data, test
from joeynmt.loss import XentLoss
from joeynmt.data import load_data, make_data_iter
from joeynmt.builders import build_optimizer, build_scheduler, \
    build_gradient_clipper

logger = logging.getLogger(__name__)

# for fp16 training
try:
    from apex import amp
    amp.register_half_function(torch, "einsum")
except ImportError as no_apex:
    logger.debug(no_apex)
    # error handling in TrainManager object construction
    #pass # pylint: disable=unnecessary-pass


class TrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, model: Model, config: dict,
                 batch_class: Batch = Batch) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.

        :param model: torch module defining the model
        :param config: dictionary containing the training configurations
        :param batch_class: batch class to encapsulate the torch class
        """
        # pylint: disable=too-many-statements,too-many-branches
        train_config = config["training"]
        self.batch_class = batch_class

        # files for logging and storing
        self.model_dir = Path(train_config["model_dir"])
        assert self.model_dir.is_dir()

        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = self.model_dir / "validations.txt"
        self.tb_writer = SummaryWriter(
            log_dir=(self.model_dir / "tensorboard").as_posix())

        # model
        self.model = model
        self.model.log_parameters_list()

        # objective
        self.label_smoothing = train_config.get("label_smoothing", 0.0)
        self.model.loss_function = XentLoss(pad_index=self.model.pad_index,
                                            smoothing=self.label_smoothing)
        self.normalization = train_config.get("normalization", "batch")
        if self.normalization not in ["batch", "tokens", "none"]:
            raise ConfigurationError("Invalid normalization option."
                                     "Valid options: "
                                     "'batch', 'tokens', 'none'.")
        logger.info(self.model)

        # optimization
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)

        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(config=train_config,
                                         parameters=model.parameters())

        # validation & early stopping
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.log_valid_sents = train_config.get("print_valid_sents", [0, 1, 2])
        self.eval_metric = train_config.get("eval_metric", "bleu")
        if self.eval_metric not in [
                'bleu', 'chrf', 'token_accuracy', 'sequence_accuracy'
        ]:
            raise ConfigurationError("Invalid setting for 'eval_metric', "
                                     "valid options: 'bleu', 'chrf', "
                                     "'token_accuracy', 'sequence_accuracy'.")
        self.early_stopping_metric = train_config.get("early_stopping_metric",
                                                      "eval_metric")

        # early_stopping_metric decides on how to find the early stopping point:
        # ckpts are written when there's a new high/low score for this metric.
        # If we schedule after BLEU/chrf/accuracy, we want to maximize the
        # score, else we want to minimize it.
        if self.early_stopping_metric in ["ppl", "loss"]:
            self.minimize_metric = True
        elif self.early_stopping_metric == "eval_metric":
            if self.eval_metric in [
                    "bleu", "chrf", "token_accuracy", "sequence_accuracy"
            ]:
                self.minimize_metric = False
            # eval metric that has to get minimized (not yet implemented)
            else:
                self.minimize_metric = True
        else:
            raise ConfigurationError(
                "Invalid setting for 'early_stopping_metric', "
                "valid options: 'loss', 'ppl', 'eval_metric'.")

        # save/delete checkpoints
        self.num_ckpts = train_config.get("num_ckpts", 5)
        self.keep_ckpts = train_config.get("keep_ckpts", "last")
        if self.keep_ckpts not in ["best", "last"]:
            raise ConfigurationError("Invalid setting for 'keep_ckpts', "
                                     "valid options: 'best', 'last'.")
        self.ckpt_queue = []
            #heapq._heapify_max([]) \   # max heap queue
            #if self.keep_ckpts == "best" and self.minimize_metric \
            #else heapq.heapify([])     # min heap queue

        keep_latest_ckpts = train_config.get("keep_latest_ckpts", None)
        if keep_latest_ckpts is not None: # backward compatibility
            self.num_ckpts = keep_latest_ckpts
            logger.warning("`keep_latest_ckpts` option is outdated. "
                           "Please use `num_ckpts` and `keep_ckpts`, instead.")

        # eval options
        test_config = config["testing"]
        self.bpe_type = test_config.get("bpe_type", "subword-nmt")
        self.sacrebleu = {"remove_whitespace": True, "tokenize": "13a"}
        if "sacrebleu" in config["testing"].keys():
            self.sacrebleu["remove_whitespace"] = test_config["sacrebleu"] \
                .get("remove_whitespace", True)
            self.sacrebleu["tokenize"] = test_config["sacrebleu"] \
                .get("tokenize", "13a")

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"])

        # data & batch handling
        self.level = config["data"]["level"]
        if self.level not in ["word", "bpe", "char"]:
            raise ConfigurationError("Invalid segmentation level. "
                                     "Valid options: 'word', 'bpe', 'char'.")
        self.seed = train_config.get("random_seed", 42)
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

        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        # generation
        self.max_output_length = train_config.get("max_output_length", None)

        # CPU / GPU
        self.use_cuda = train_config["use_cuda"] and torch.cuda.is_available()
        self.n_gpu = torch.cuda.device_count() if self.use_cuda else 0
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.model.to(self.device)
        self.num_workers = train_config.get("num_workers", 0)
        if self.num_workers > 0:
            self.num_workers = min(cpu_count(), self.num_workers)

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

        # initialize training statistics
        self.stats = self.TrainStatistics(
            steps=0,
            is_min_lr=False,
            is_max_update=False,
            total_tokens=0,
            best_ckpt_iter=0,
            best_ckpt_score=np.inf if self.minimize_metric else -np.inf,
            minimize_metric=self.minimize_metric)

        # model parameters
        if "load_model" in train_config.keys():
            self.init_from_checkpoint(
                Path(train_config["load_model"]),
                reset_best_ckpt=train_config.get("reset_best_ckpt", False),
                reset_scheduler=train_config.get("reset_scheduler", False),
                reset_optimizer=train_config.get("reset_optimizer", False),
                reset_iter_state=train_config.get("reset_iter_state", False))

        # multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1:
            self.model = _DataParallel(self.model)

    def _save_checkpoint(self, new_best: bool, score: float) -> None:
        """
        Save the model's current parameters and the training state to a
        checkpoint.

        The training state contains the total number of training steps,
        the total number of training tokens,
        the best checkpoint score and iteration so far,
        and optimizer and scheduler states.

        :param new_best: This boolean signals which symlink we will use for the
                         new checkpoint. If it is true, we update best.ckpt.
        :param score: Validation score which is used as key of heap queue.
        """
        model_path = self.model_dir / f"{self.stats.steps}.ckpt"
        logger.info("Saving new checkpoint: %s", model_path)
        model_state_dict = self.model.module.state_dict() \
            if isinstance(self.model, torch.nn.DataParallel) \
            else self.model.state_dict()
        state = {
            "steps": self.stats.steps,
            "total_tokens": self.stats.total_tokens,
            "best_ckpt_score": self.stats.best_ckpt_score,
            "best_ckpt_iteration": self.stats.best_ckpt_iter,
            "model_state": model_state_dict,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() \
                if self.scheduler is not None else None,
            'amp_state': amp.state_dict() if self.fp16 else None,
            "train_iter_state":
            self.train_iter.batch_sampler.sampler.generator.get_state()
        }
        torch.save(state, model_path.as_posix())

        # update symlink
        symlink_target = Path(f"{self.stats.steps}.ckpt")
        # last symlink
        last_path = self.model_dir / "latest.ckpt"
        symlink_update(symlink_target, last_path)
        # best symlink
        best_path = self.model_dir / "best.ckpt"
        if new_best:
            symlink_update(symlink_target, best_path)
            assert best_path.resolve().stem == str(self.stats.best_ckpt_iter)

        # push to and pop from the heap queue
        if self.num_ckpts > 0:
            to_delete = None
            heap_key = score if self.keep_ckpts == "best" else self.stats.steps
            if len(self.ckpt_queue) < self.num_ckpts: # no pop, push only
                heapq.heappush(self.ckpt_queue, (heap_key, model_path))
            else: # pop the oldest / worst one in the queue
                if self.keep_ckpts == "best" and self.minimize_metric:
                    # pylint: disable=protected-access
                    to_delete = heapq._heappushpop_max(self.ckpt_queue,
                                                       (heap_key, model_path))
                    # pylint: enable=protected-access
                else: #if self.keep_ckpts == "last" or
                    # (self.keep_ckpts == "best" and not self.minimize_metric):
                    to_delete = heapq.heappushpop(self.ckpt_queue,
                                                  (heap_key, model_path))
                assert to_delete[1] != model_path # don't delete the latest one

            if to_delete is not None \
                    and to_delete[1].stem != best_path.resolve().stem:
                delete_ckpt(to_delete[1])         # don't delete the best one

            assert len(self.ckpt_queue) <= self.num_ckpts

    def init_from_checkpoint(self,
                             path: Path,
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
        model_checkpoint = load_checkpoint(path=path, device=self.device)

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

        if not reset_best_ckpt:
            self.stats.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.stats.best_ckpt_iter = model_checkpoint["best_ckpt_iteration"]
        else:
            logger.info("Reset tracking of the best checkpoint.")

        if (not reset_iter_state
                and model_checkpoint.get('train_iter_state', None) is not None):
            self.train_iter_state = model_checkpoint["train_iter_state"]
        else:
            logger.info("Reset data iterator (random seed: {%d}).", self.seed)

        # move parameters to cuda
        if self.use_cuda:
            self.model.to(self.device)

        # fp16
        if self.fp16 and model_checkpoint.get("amp_state", None) is not None:
            amp.load_state_dict(model_checkpoint['amp_state'])

    def train_and_validate(self, train_data: Dataset, valid_data: Dataset) \
            -> None:
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """
        # pylint: disable=unnecessary-comprehension
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements
        self.train_iter = make_data_iter(train_data,
                                         batch_size=self.batch_size,
                                         batch_type=self.batch_type,
                                         batch_class=self.batch_class,
                                         seed=self.seed,
                                         shuffle=self.shuffle,
                                         num_workers=self.num_workers,
                                         pad_index=self.model.pad_index,
                                         device=self.device)

        if self.train_iter_state is not None:
            self.train_iter.batch_sampler.sampler.generator.set_state(
                self.train_iter_state.cpu())

        #################################################################
        # simplify accumulation logic:
        #################################################################
        # for epoch in range(epochs):
        #     self.model.zero_grad()
        #     epoch_loss = 0.0
        #     batch_loss = 0.0
        #     for i, batch in enumerate(self.train_iter):
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

        # pylint: disable=logging-too-many-args
        logger.info(
            "Train stats:\n"
            "\tdevice: %s\n"
            "\tn_gpu: %d\n"
            "\t16-bits training: %r\n"
            "\tgradient accumulation: %d\n"
            "\tbatch size per device: %d\n"
            "\ttotal batch size (w. parallel & accumulation): %d\n"
            "\tnum. of multiprocessing workers: %d",
            self.device.type, self.n_gpu, self.fp16, self.batch_multiplier,
            self.batch_size//self.n_gpu if self.n_gpu > 1 else self.batch_size,
            self.batch_size * self.batch_multiplier, self.num_workers)
        # pylint: enable=logging-too-many-args

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

            for i, batch in enumerate(self.train_iter):
                # yield a joeynmt Batch object
                batch.sort_by_src_length()

                # get batch loss
                batch_loss += self._train_step(batch)

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
                        self.tb_writer.add_scalar("train/train_batch_loss",
                                                  batch_loss, self.stats.steps)
                        elapsed = time.time() - start - total_valid_duration
                        elapsed_tokens = self.stats.total_tokens - start_tokens
                        logger.info(
                            "Epoch %3d, Step: %8d, Batch Loss: %12.6f, "
                            "Tokens per Sec: %8.0f, Lr: %.6f", epoch_no + 1,
                            self.stats.steps, batch_loss,
                            elapsed_tokens / elapsed,
                            self.optimizer.param_groups[0]["lr"])
                        start = time.time()
                        total_valid_duration = 0
                        start_tokens = self.stats.total_tokens

                    # Only add complete loss of full mini-batch to epoch_loss
                    epoch_loss += batch_loss  # accumulate epoch_loss
                    batch_loss = 0  # rest batch_loss

                    # validate on the entire dev set
                    if self.stats.steps % self.validation_freq == 0:
                        valid_duration = self._validate(valid_data, epoch_no)
                        total_valid_duration += valid_duration

                if self.stats.is_min_lr or self.stats.is_max_update:
                    break

            if self.stats.is_min_lr or self.stats.is_max_update:
                log_str = f"minimum lr {self.learning_rate_min}" \
                    if self.stats.is_min_lr \
                    else f"maximum num. of updates {self.max_updates}"
                logger.info("Training ended since %s was reached.", log_str)
                break

            logger.info('Epoch %3d: total training loss %.2f', epoch_no + 1,
                        epoch_loss)
        else:
            logger.info('Training ended after %3d epochs.', epoch_no + 1)
        logger.info('Best validation result (greedy) at step %8d: %6.2f %s.',
                    self.stats.best_ckpt_iter, self.stats.best_ckpt_score,
                    self.early_stopping_metric)

        self.tb_writer.close()  # close Tensorboard writer

    def _train_step(self, batch: Batch) -> Tensor:
        """
        Train the model on one batch: Compute the loss.

        :param batch: training batch
        :return: loss for batch (sum)
        """
        # reactivate training
        self.model.train()

        # get loss (run as during training with teacher forcing)
        batch_loss, _, _, _ = self.model(return_type="loss", **vars(batch))

        # sum multi-gpu losses
        if self.n_gpu > 1:
            batch_loss = batch_loss.sum()

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

        if self.n_gpu > 1:
            norm_batch_loss = norm_batch_loss / self.n_gpu

        if self.batch_multiplier > 1:
            norm_batch_loss = norm_batch_loss / self.batch_multiplier

        # accumulate gradients
        if self.fp16:
            with amp.scale_loss(norm_batch_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            norm_batch_loss.backward()

        # increment token counter
        self.stats.total_tokens += batch.ntokens

        return norm_batch_loss.item()

    def _validate(self, valid_data, epoch_no):
        valid_start_time = time.time()

        valid_score, valid_loss, valid_ppl, valid_sources, \
        valid_sources_raw, valid_references, valid_hypotheses, \
        valid_hypotheses_raw, valid_attention_scores = \
            validate_on_data(
                batch_size=self.eval_batch_size,
                batch_class=self.batch_class,
                data=valid_data,
                eval_metric=self.eval_metric,
                level=self.level, model=self.model,
                device=self.device,
                max_output_length=self.max_output_length,
                compute_loss=True,
                beam_size=1,                # greedy validations
                batch_type=self.eval_batch_type,
                postprocess=True,           # always remove BPE for validation
                bpe_type=self.bpe_type,     # "subword-nmt" or "sentencepiece"
                sacrebleu=self.sacrebleu,   # sacrebleu options
                n_gpu=self.n_gpu
            )

        self.tb_writer.add_scalar("valid/valid_loss", valid_loss,
                                  self.stats.steps)
        self.tb_writer.add_scalar("valid/valid_score", valid_score,
                                  self.stats.steps)
        self.tb_writer.add_scalar("valid/valid_ppl", valid_ppl,
                                  self.stats.steps)

        if self.early_stopping_metric == "loss":
            ckpt_score = valid_loss
        elif self.early_stopping_metric in ["ppl", "perplexity"]:
            ckpt_score = valid_ppl
        else:
            ckpt_score = valid_score

        if self.scheduler is not None \
                and self.scheduler_step_at == "validation":
            self.scheduler.step(ckpt_score)

        # update new best
        new_best = self.stats.is_best(ckpt_score)
        if new_best:
            self.stats.best_ckpt_score = ckpt_score
            self.stats.best_ckpt_iter = self.stats.steps
            logger.info('Hooray! New best validation result [%s]!',
                        self.early_stopping_metric)

        # save checkpoints
        is_better = self.stats.is_better(ckpt_score, self.ckpt_queue) \
            if len(self.ckpt_queue) > 0 else True
        if self.keep_ckpts == "last" or \
                (self.keep_ckpts == "best" and is_better):
            self._save_checkpoint(new_best, ckpt_score)

        # append to validation report
        self._add_report(valid_score=valid_score,
                         valid_loss=valid_loss,
                         valid_ppl=valid_ppl,
                         eval_metric=self.eval_metric,
                         new_best=new_best)

        self._log_examples(sources_raw=valid_sources_raw,
                           sources=valid_sources,
                           hypotheses_raw=valid_hypotheses_raw,
                           hypotheses=valid_hypotheses,
                           references=valid_references)

        valid_duration = time.time() - valid_start_time
        logger.info(
            'Validation result (greedy) at epoch %3d, '
            'step %8d: %s: %6.2f, loss: %8.4f, ppl: %8.4f, '
            'duration: %.4fs', epoch_no + 1, self.stats.steps, self.eval_metric,
            valid_score, valid_loss, valid_ppl, valid_duration)

        # store validation set outputs
        self._store_outputs(valid_hypotheses)

        # store attention plots for selected valid sentences
        if valid_attention_scores:
            store_attention_plots(
                attentions=valid_attention_scores,
                targets=valid_hypotheses_raw,
                sources=valid_data.src,
                indices=self.log_valid_sents,
                output_prefix=(self.model_dir / f"att.{self.stats.steps}"
                               ).as_posix(),
                tb_writer=self.tb_writer,
                steps=self.stats.steps)

        return valid_duration

    def _add_report(self,
                    valid_score: float,
                    valid_ppl: float,
                    valid_loss: float,
                    eval_metric: str,
                    new_best: bool = False) -> None:
        """
        Append a one-line report to validation logging file.

        :param valid_score: validation evaluation score [eval_metric]
        :param valid_ppl: validation perplexity
        :param valid_loss: validation loss (sum over whole validation set)
        :param eval_metric: evaluation metric, e.g. "bleu"
        :param new_best: whether this is a new best model
        """
        #current_lr = -1
        # ignores other param groups for now
        #for param_group in self.optimizer.param_groups:
        #    current_lr = param_group['lr']
        current_lr = self.optimizer.param_groups[0]['lr']

        if current_lr < self.learning_rate_min:
            self.stats.is_min_lr = True

        with self.valid_report_file.open('a') as opened_file:
            opened_file.write(
                "Steps: {}\tLoss: {:.5f}\tPPL: {:.5f}\t{}: {:.5f}\t"
                "LR: {:.8f}\t{}\n".format(self.stats.steps, valid_loss,
                                          valid_ppl, eval_metric, valid_score,
                                          current_lr, "*" if new_best else ""))

    def _log_examples(self,
                      sources: List[str],
                      hypotheses: List[str],
                      references: List[str],
                      sources_raw: List[List[str]] = None,
                      hypotheses_raw: List[List[str]] = None,
                      references_raw: List[List[str]] = None) -> None:
        """
        Log a the first `self.log_valid_sents` sentences from given examples.

        :param sources: decoded sources (list of strings)
        :param hypotheses: decoded hypotheses (list of strings)
        :param references: decoded references (list of strings)
        :param sources_raw: raw sources (list of list of tokens)
        :param hypotheses_raw: raw hypotheses (list of list of tokens)
        :param references_raw: raw references (list of list of tokens)
        """
        for p in self.log_valid_sents:

            if p >= len(sources):
                continue

            logger.info("Example #%d", p)

            if sources_raw is not None:
                logger.debug("\tRaw source:     %s", sources_raw[p])
            if references_raw is not None:
                logger.debug("\tRaw reference:  %s", references_raw[p])
            if hypotheses_raw is not None:
                logger.debug("\tRaw hypothesis: %s", hypotheses_raw[p])

            logger.info("\tSource:     %s", sources[p])
            logger.info("\tReference:  %s", references[p])
            logger.info("\tHypothesis: %s", hypotheses[p])

    def _store_outputs(self, hypotheses: List[str]) -> None:
        """
        Write current validation outputs to file in `self.model_dir.`

        :param hypotheses: list of strings
        """
        current_valid_output_file = self.model_dir / f"{self.stats.steps}.hyps"
        with current_valid_output_file.open('w') as opened_file:
            for hyp in hypotheses:
                opened_file.write(f"{hyp}\n")

    class TrainStatistics:
        def __init__(self,
                     steps: int = 0,
                     is_min_lr: bool = False,
                     is_max_update: bool = False,
                     total_tokens: int = 0,
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

        def is_better(self, score, heap_queue):
            if self.minimize_metric:
                is_better = score < heapq.nsmallest(1, heap_queue)[0][0]
            else:
                is_better = score > heapq.nlargest(1, heap_queue)[0][0]
            return is_better


def train(cfg_file: str) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    """
    cfg = load_config(Path(cfg_file))

    # make logger
    model_dir = make_model_dir(Path(cfg["training"]["model_dir"]),
                               overwrite=cfg["training"].get(
                                   "overwrite", False))
    _ = make_logger(model_dir, mode="train")  # version string returned
    # TODO: save version number in model checkpoints

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    # load the data
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(
        data_cfg=cfg["data"], num_workers=cfg["training"].get("num_workers", 0))

    # build an encoder-decoder model
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, config=cfg, batch_class=Batch)

    # store copy of original training config in model dir
    shutil.copy2(cfg_file, (model_dir / "config.yaml").as_posix())

    # log all entries of config
    log_cfg(cfg)

    # store the vocabs
    src_vocab.to_file(model_dir / "src_vocab.txt")
    trg_vocab.to_file(model_dir / "trg_vocab.txt")

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)

    # predict with the best model on validation and test
    # (if test data is available)
    ckpt = model_dir / f"{trainer.stats.best_ckpt_iter}.ckpt"
    output_path = model_dir / "{:08d}.hyps".format(trainer.stats.best_ckpt_iter)
    datasets_to_test = {
        "dev": dev_data,
        "test": test_data,
        "src_vocab": src_vocab,
        "trg_vocab": trg_vocab
    }
    test(cfg_file,
         ckpt=ckpt.as_posix(),
         output_path=output_path.as_posix(),
         datasets=datasets_to_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Joey-NMT')
    parser.add_argument("config",
                        default="configs/default.yaml",
                        type=str,
                        help="Training configuration file (yaml).")
    args = parser.parse_args()
    train(cfg_file=args.config)
