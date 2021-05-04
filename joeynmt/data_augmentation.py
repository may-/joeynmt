# coding: utf-8
"""
Data Augmentation
"""
import math
import random
import string
import logging
from scipy.stats import lognorm
from functools import partial
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np

import torch

from joeynmt.helpers import bpe_postprocess
from joeynmt.helpers_for_audio import get_features

logger = logging.getLogger(__name__)


# from fairseq
class SpecAugment:
    def __init__(self,
                 freq_mask_n: int = 2,
                 freq_mask_f: int = 27,
                 time_mask_n: int = 2,
                 time_mask_t: int = 40,
                 time_mask_p: float = 1.0,
                 mask_value: Optional[float] = None):

        self.freq_mask_n = freq_mask_n
        self.freq_mask_f = freq_mask_f
        self.time_mask_n = time_mask_n
        self.time_mask_t = time_mask_t
        self.time_mask_p = time_mask_p
        self.mask_value = mask_value

    def __call__(self, spectrogram: np.ndarray):
        assert len(spectrogram.shape) == 2, "spectrogram must be a 2-D tensor."

        distorted = spectrogram.copy()  # make a copy of input spectrogram.
        num_frames, num_freqs = spectrogram.shape
        mask_value = self.mask_value

        if mask_value is None:  # if no value was specified, use local mean.
            mask_value = spectrogram.mean()

        if num_frames == 0:
            return spectrogram

        if num_freqs < self.freq_mask_f:
            return spectrogram

        for _i in range(self.freq_mask_n):
            f = np.random.randint(0, self.freq_mask_f)
            f0 = np.random.randint(0, num_freqs - f)
            if f != 0:
                distorted[:, f0 : f0 + f] = mask_value

        max_time_mask_t = min(
            self.time_mask_t, math.floor(num_frames * self.time_mask_p)
        )
        if max_time_mask_t < 1:
            return distorted

        for _i in range(self.time_mask_n):
            t = np.random.randint(0, max_time_mask_t)
            t0 = np.random.randint(0, num_frames - t)
            if t != 0:
                distorted[t0 : t0 + t, :] = mask_value

        return distorted

    def __repr__(self):
        return (self.__class__.__name__
                + f"(freq_mask_n={self.freq_mask_n}, "
                + f"freq_mask_f={self.freq_mask_f}, "
                + f"time_mask_n={self.time_mask_n}, "
                + f"time_mask_t={self.time_mask_t}, "
                + f"time_mask_p={self.time_mask_p})")

# from fairseq
class CMVN:
    """
    CMVN: Cepstral Mean and Variance Normalization
    (Utterance-level)
    """

    def __init__(self, norm_means: bool = True, norm_vars: bool = True):
        self.norm_means = norm_means
        self.norm_vars = norm_vars

    def __call__(self, x: np.ndarray) -> np.ndarray:
        orig_shape = x.shape
        mean = x.mean(axis=0)
        square_sums = (x ** 2).sum(axis=0)

        if self.norm_means:
            x = np.subtract(x, mean)
        if self.norm_vars:
            var = square_sums / x.shape[0] - mean ** 2
            std = np.sqrt(np.maximum(var, 1e-10))
            x = np.divide(x, std)

        assert orig_shape == x.shape
        return x

    def __repr__(self):
        return (self.__class__.__name__ +
                f"(norm_means={self.norm_means}, norm_vars={self.norm_vars})")


class Sampler(object):
    def __init__(self, sampler_cfg, unk_token):
        try:
            import pandas as pd
        except:
            raise ImportError("Please install pandas: `pip install pandas`")

        self.sample_type = sampler_cfg.get('sample_type', 'random')
        assert self.sample_type in {'random', 'least-freq'}

        self.sample_ratio = sampler_cfg.get('sample_ratio', 0.2)
        assert 0.0 <= self.sample_ratio <= 1.0
        self.sample_max_len = sampler_cfg.get('sample_max_len', 10)
        self.min_trg_len = sampler_cfg.get('min_trg_len', 10)
        self.token_sample_prob = sampler_cfg.get('token_sample_prob', 0.5)
        assert 0.0 <= self.token_sample_prob <= 1.0
        self.instance_sample_prob = sampler_cfg.get('instance_sample_prob', 0.5)
        assert 0.0 <= self.instance_sample_prob <= 1.0
        self.freq_dict = pd.read_csv(sampler_cfg["freq_dict_file"],
            sep='\t', index_col=0, keep_default_na=False) \
            if self.sample_type == 'least-freq' else None

        # don't chose words in this list, i.e. punctuations
        #'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        self.stop_words = list(string.punctuation)
        self.unk_token = unk_token

    def _lookup_freq(self, token):
        if self.freq_dict is not None: #self.sample_type == 'least-freq':
            if token == self.unk_token:
                return -1
            elif token not in self.freq_dict.index:
                return 0
            return self.freq_dict.loc[token].freq
        else:
            return None

    def _sample_positions(self,
                          trg: List[str],
                          start: int = -np.inf,
                          end: int = np.inf) -> Dict[int, str]:
        """
        sample mask positions
        :param trg: word-tokenized transcription (not bpe!)
        :param start: restrict mask positions between start and end
        :param end:
        :return: dict {i: new_word} => i-th word will be replaced by new_word
        """
        trg_len = (end - start) if not np.isinf(start) and not np.isinf(end) else len(trg)
        # if trg seq is too short, then do nothing
        if trg_len < self.min_trg_len:
            return {}

        # coin flip per utterance
        if np.random.binomial(1, self.instance_sample_prob) == 0:
            return {}

        # number of positions to mask
        num_masks = int(min(self.sample_max_len,
                            np.ceil(trg_len * self.sample_ratio)))

        # exclude stop words
        candidates = [(i, w, self._lookup_freq(w)) for i, w in enumerate(trg)
                      if w not in self.stop_words and start <= i <= end]
        if len(candidates) <= num_masks or len(candidates) < self.min_trg_len:
            return {}

        if self.sample_type == 'random':
            pos = random.choices(candidates, k=num_masks)
        elif self.sample_type == 'least-freq':
            pos = sorted(candidates, key=lambda x: x[-1])[:num_masks]

        # coin flip; whether to mask or not
        coin_flip = np.random.binomial(1, self.token_sample_prob, num_masks)
        positions = {x[0]: x[1] for x, flag in zip(pos, coin_flip) if flag > 0}

        return dict(positions)

    def __call__(self, textgrids: List[List[List]],
                 word_cutoff:Tuple[List, List] = None) -> List[Dict[int, str]]:
        """
        sample mask positions (word-level tokenization)
        :param textgrids:
        :param word_cutoff:
        :return: list of dict {i: new_word} => i-th word will be replaced by new_word
        """
        batch_positions = []
        word_start, word_end = (None, None) if word_cutoff is None else word_cutoff
        for j, textgrid in enumerate(textgrids):
            trg = [t[-1] for t in textgrid if len(t[-1]) > 0]
            start = -np.inf if word_start is None else word_start[j]
            end = np.inf if word_end is None else word_end[j]
            batch_positions.append(self._sample_positions(trg, start=start, end=end))
        return batch_positions

    def __repr__(self):
        freq_dict_str = "freq_dict=None" if self.freq_dict is None \
            else f"len(freq_dict)={len(self.freq_dict)}"
        return (self.__class__.__name__
                + "(" + ", ".join([
                    f"sample_type='{self.sample_type}'",
                    f"sample_ratio={self.sample_ratio}",
                    f"sample_max_len={self.sample_max_len}",
                    f"min_trg_len={self.min_trg_len}",
                    f"token_sample_prob={self.token_sample_prob}",
                    f"instance_sample_prob={self.instance_sample_prob}",
                    freq_dict_str,
                    f"len(stop_words)={len(self.stop_words)}"])
                + ")")


class AlignedMasking(object):
    def __call__(self, x: np.ndarray,
                 positions: Dict[int, str],
                 textgrids: List[List],
                 word_cutoff: Tuple[int, int] = None) -> np.ndarray:
        """
        AlignedMasking
        :param x: (np.ndarray) spectrogram features,
                    shape of (num_frames, num_freq)
        :param positions: (Dict[int, str])
        :param textgrids: (List[List])
        :param word_cutoff: (tuple(int, int))
        :return: augmented_spectrogram
        """
        if len(positions) == 0:
            return x

        assert len(x.shape) == 2, "spectrogram must be a 2-D tensor."

        distorted = x.copy()  # make a copy of input spectrogram.
        num_frames = x.shape[0]
        mask_value = x.mean()

        pos = -1
        flag = False # you need this flag because of in-sentence silent token

        ####          [[start, end, word, word_orig]]  pos
        # textgrids = [[0, 8, ''],                      -   # empty surface for silent token
        #              [8, 30, 'dey', 'dey'],           0
        #              [30, 42, 'all', 'all'],          1
        #              [42, 68, 'made', 'made'],        2
        #              [68, 76, 'der', 'der'],          3
        #              [76, 137, ''],                   -   # silence in the middle
        #              [137, 167, 'offen', 'offen'],    4
        #              [167, 200, '<unk>', "roas'"],    5   # original surface for <unk>
        #              [200, 209, 'in', "in'"],         6
        #              [209, 247, 'years', 'years'],    7
        # ...
        ####

        for align in textgrids:
            # align = [start, end, word, word_orig]
            if len(align[-1]) > 0:  # empty surface = silent
                pos += 1            # increment only when non-empty
            else:
                continue

            if pos in positions.keys():
                flag = True # enter in edit mode

            if flag:
                start = align[0]
                end = align[1]
                distorted[start:end, :] = mask_value
                flag = False

        assert num_frames == distorted.shape[0]

        if word_cutoff is not None:
            word_start, word_end = word_cutoff
            if word_start is None and word_end is None:
                return distorted
            assert min(positions.keys()) >= word_start
            assert max(positions.keys()) <= word_end
            tg = [t for t in textgrids if len(t[-1]) > 0]
            src_start = tg[word_start][0]
            src_end = tg[word_end][1]
            return distorted[src_start:src_end, :]
        return distorted


class AudioDictAugment(object):
    def __init__(self, audio_dict_cfg):
        try:
            import pandas as pd
        except:
            raise ImportError("Please install pandas: `pip install pandas`")

        # look up dictionary based on word or phoneme
        self.lookup = audio_dict_cfg.get('lookup', 'word')
        assert self.lookup in {'word', 'phoneme'}

        # construct audio dict dataframe
        audio_dict_file = Path(audio_dict_cfg['audio_dict_file']).expanduser()
        #self.audio_dict_root, _ = os.path.split(audio_dict_file)    # assume audio_dict.zip locates there
        self.audio_dict_root = audio_dict_file.parent
        self.audio_dict_df = pd.read_csv(                           # load tsv file
            audio_dict_file.as_posix(), sep='\t', index_col=0, keep_default_na=False)
        #     total 9,406,611 entries in this dict]

        # list of splits you want to use. comma separated
        # 'train-clean-100,train-clean-360,train-other-500' for 960h dataset
        self.splits = audio_dict_cfg.get('splits', 'train-clean-100')
        audio_dict_splits = [s.strip() for s in self.splits.split(',') if len(s.strip()) > 0]

        # use the specified splits only
        self.audio_dict_df = self.audio_dict_df[self.audio_dict_df['split'].isin(audio_dict_splits)]

        # gather frequency information
        self.audio_dict_df['count'] = 1
        freq = self.audio_dict_df.groupby([self.lookup], as_index=True).agg({'count': 'sum'})
        self.audio_dict = pd.merge(self.audio_dict_df, freq, how='left',
                                   on=self.lookup).drop(columns=['count_x']).rename(columns={"count_y": "count"})
        # use the words which appear more than three times only
        self.audio_dict_threshold = audio_dict_cfg.get('threshold', 3)
        self.audio_dict = self.audio_dict[self.audio_dict['count'] > self.audio_dict_threshold]

        self.mask_when_no_hit = audio_dict_cfg.get('mask_when_no_hit', False)

        self.bpe_type = 'sentencepiece' #audio_dict_cfg.get('bpe_type', 'sentencepiece')
        self.hit_count = 0
        self.query_count = 0
        self.seg_count = 0
        self.sent_count = 0

    def __call__(self, x: np.ndarray,
                 positions: Dict[int, str],
                 textgrids: List[List],
                 word_cutoff: Tuple[int, int] = None) -> List[Dict[int, str]]:
        """
        AudioDictSampling
        :param x: (np.ndarray) spectrogram features,
                    shape of (num_frames, num_freq)
        :param positions: (Dict[int, str])
        :param textgrids: (List[List])
        :param word_cutoff: (tuple(int, int))
        :return: augmented_spectrogram
        """
        assert len(x.shape) == 2, "spectrogram must be a 2-D tensor."
        n_frames = x.shape[1]

        x_new = []
        align_new = []
        edit_flag = False # you need this flag because of in-sentence silent token

        ####          [[start, end, word, word_orig]]  pos
        # textgrids = [[0, 8, ''],                      -   # empty surface for silent token
        #              [8, 30, 'dey', 'dey'],           0
        #              [30, 42, 'all', 'all'],          1
        #              [42, 68, 'made', 'made'],        2
        #              [68, 76, 'der', 'der'],          3
        #              [76, 137, ''],                   -   # silence in the middle
        #              [137, 167, 'offen', 'offen'],    4
        #              [167, 200, '<unk>', "roas'"],    5   # original surface for <unk>
        #              [200, 209, 'in', "in'"],         6
        #              [209, 247, 'years', 'years'],    7
        # ...
        ####

        pos = -1
        start = 0
        query_count = 0
        hit_count = 0
        for align in textgrids:
            if len(align[-1]) > 0:  # only when non-empty
                pos += 1            # increment
            else:                   # empty surface = silent
                x_new.append(x[align[0]:align[1], :])
                l = align[1] - align[0] + 1
                align_new.append([start, start+l])
                start += l
                continue

            if pos in positions.keys():
                edit_flag = True # enter in edit mode

            if edit_flag:
                surface = bpe_postprocess(positions[pos].replace(' ', '▁'),
                                          bpe_type=self.bpe_type)
                words = surface.split() # 'surface' can consist of multiple words!
                l = 0
                hit_flag = True
                x_new_partial = []
                for word in words:
                    #print('word', word.upper())
                    filtered = self.audio_dict[self.audio_dict[self.lookup] == word.upper()]
                    query_count += 1
                    if len(filtered) < self.audio_dict_threshold:
                        hit_flag = False
                        break
                    else:   # if found:
                        sampled = filtered.sample(n=1).iloc[0]  # sample one entry from dict
                        f = get_features(self.audio_dict_root/sampled.audio)
                        f = f[sampled.start:sampled.end, :]
                        assert f.shape[0] == (sampled.n_frames - 1)
                        hit_count += 1
                        x_new_partial.append(f)
                        l += f.shape[0]

                if hit_flag:
                    assert len(x_new_partial) == len(words) and l > 0
                    x_new.extend(x_new_partial)
                else:
                    l = align[1] - align[0] + 1
                    if self.mask_when_no_hit:   # mask features
                        f = np.full((l, n_frames), x.mean(), dtype=x.dtype)
                    else:                       # pass through
                        f = x[align[1]:align[0], :]
                    x_new.append(f)
                align_new.append([start, start+l])
                start += l

                edit_flag = False # leave edit mode

            else:
                x_new.append(x[align[0]:align[1], :])
                l = align[1] - align[0] + 1
                align_new.append([start, start+l])
                start += l
                continue

        if query_count > 0:
            self.sent_count += 1
            self.seg_count += len(x_new)
            self.query_count += query_count
            self.hit_count += hit_count

        augmented = np.concatenate(x_new)
        del x_new

        if word_cutoff is not None:
            word_start, word_end = word_cutoff
            if word_start is None and word_end is None:
                return augmented
            assert min(positions.keys()) >= word_start
            assert max(positions.keys()) <= word_end
            src_start = align_new[word_start][0]
            src_end = align_new[word_end][1]
            return augmented[src_start:src_end, :]
        return augmented

    def reset_stats(self):
        logger.info(f"hit rate: {self.hit_count/self.query_count}, "
                    f"(hit count: {self.hit_count}, "
                    f"query count: {self.query_count}, "
                    f"segment count: {self.seg_count}, "
                    f"utterance count: {self.sent_count})")
        self.hit_count = 0
        self.query_count = 0
        self.seg_count = 0
        self.sent_count = 0

    def __repr__(self):
        return (self.__class__.__name__
                + f"(lookup='{self.lookup}', "
                + f"splits='{self.splits}', "
                + f"threshold={self.audio_dict_threshold}, "
                + f"len(audio_dict)={len(self.audio_dict)}, "
                + f"mask_when_no_hit={self.mask_when_no_hit})")


class MaskedLMAugment(object):
    def __init__(self, mlm_cfg, lowercase: bool, stoi: Dict[str, int],
                 unk_index: int, bos_index: int, eos_index: int, device):
        try:
            from fairseq.models.roberta import RobertaModel
            self.model_name = mlm_cfg['language_model_path']
            # currently roberta only
            assert self.model_name.rsplit('/', 1)[-1] in {'roberta.base', 'roberta.large'}
            #roberta = torch.hub.load('pytorch/fairseq', self.model_name)
            self.roberta = RobertaModel.from_pretrained(
                self.model_name, checkpoint_file='model.pt')
            self.roberta.model.eval()
            self.roberta.model.to(device)
            self.max_length = 512
            if hasattr(self.roberta, 'cfg'):
                self.max_length = self.roberta.cfg.model.tokens_per_sample
            elif hasattr(self.roberta, 'args'):
                self.max_length = self.roberta.args.tokens_per_sample
            self.mask_token = self.roberta.task.source_dictionary.string(
                torch.tensor([self.roberta.task.mask_idx])) # <mask>
            self.bos_token = self.roberta.task.source_dictionary.bos_word # <s>
            self.eos_token = self.roberta.task.source_dictionary.eos_word # </s>
            self.pad_token = self.roberta.task.source_dictionary.pad_word # <pad>
            self.unk_token = self.roberta.task.source_dictionary.unk_word # <unk>
            self.pad_index = self.roberta.task.source_dictionary.encode_line(
                self.pad_token, append_eos=False).item() # 1
        except Exception as e:
            raise ImportError(e)

        try:
            import sentencepiece as sp
            self.model_file = mlm_cfg['sentencepiece_model_file']
            self.spm = sp.SentencePieceProcessor(model_file=self.model_file)
        except:
            raise ImportError

        self.lowercase = lowercase
        self.stoi = stoi
        self.unk_id = unk_index # unk in joeynmt's vocab
        self.bos_id = bos_index
        self.eos_id = eos_index
        self.device = device

        self.stop_token_list = []
        stop_token_file = mlm_cfg.get('stop_token_file', None)
        if stop_token_file:
            stop_token_file = Path(stop_token_file)
            if stop_token_file.is_file():
                try:
                    import json
                    with open(stop_token_file, 'r') as f:
                        stop = json.load(f)
                    stop_tokens = ' '.join([str(t) for t in stop.values()]
                        + [self.bos_token, self.eos_token, self.mask_token,
                           self.pad_token, self.unk_token])
                    self.stop_token_list = self.roberta.task.source_dictionary.encode_line(
                        stop_tokens, append_eos=False).tolist()
                except:
                    raise ImportError

    def _mask(self, trg, positions):
        tmp = []
        trg_masked = []
        resort_index = []
        for i, t in enumerate(trg):
            if i not in positions.keys():
                tmp.append(t)
            else:
                trg_masked.append(self.roberta.bpe.encode(" ".join(tmp)))
                m = " " + positions[i] if i > 0 else positions[i]
                l = len(self.roberta.bpe.encode(m).split())
                trg_masked.extend([self.mask_token]*l)
                resort_index.extend([i]*l)
                tmp = []
        return trg_masked, resort_index

    def encode(self, trg_masked):
        bpe_sentence = [self.bos_token] + trg_masked + [self.eos_token]
        tokens = self.roberta.task.source_dictionary.encode_line(
            " ".join(bpe_sentence), append_eos=False, add_if_not_exist=False
        )
        return tokens

    def __call__(self,
                 batch_positions: List[Dict[int, str]],
                 batch_textgrids: List[List[List]],
                 steps: int=0) -> List[Dict[int, str]]:
        """
        language model inference in batch level
        :param batch_positions:
        :param batch_textgrids:
        :param steps: (for logging)
        :return:
        """
        if sum([1 if len(p) > 0 else 0 for p in batch_positions]) == 0:
            return batch_positions

        batch_size = len(batch_positions)
        assert batch_size == len(batch_textgrids)
        input_ids = torch.ones((batch_size, self.max_length), dtype=torch.long)
        input_ids.new_full((batch_size, self.max_length), self.pad_index)
        resort_batch_index = []
        resort_token_index = []
        i = -1
        max_len = 0
        for positions, textgrid in zip(batch_positions, batch_textgrids):
            if len(positions) > 0:
                i += 1
                trg = [t[-1] for t in textgrid if len(t[-1]) > 0]

                trg_masked, resort_index = self._mask(trg, positions)
                trg_ids = self.encode(trg_masked)
                resort_token_index.append(resort_index)

                length = len(trg_ids)
                if max_len < length:
                    max_len = length

                input_ids[i, 0:length] = trg_ids
                resort_batch_index.append(i)
            else:
                resort_batch_index.append(None)
        assert batch_size == len(resort_batch_index)

        lm_batch_size = len([i for i in resort_batch_index if i is not None])
        input_ids = input_ids[:lm_batch_size, :max_len].to(device=self.device)
        assert input_ids.dim() == 2

        mask = (input_ids == self.roberta.task.mask_idx)
        with torch.no_grad():
            try:
                features, extra = self.roberta.model(
                    input_ids,
                    features_only=False,
                    return_all_hiddens=False,
                )
            except Exception as e:
                logger.debug(f"Error at {steps}: {e}")
                if steps > 0:
                    obj = {"input_ids": input_ids,
                           "positions": batch_positions,
                           "textgrids": batch_textgrids}
                    torch.save(obj, Path("/scratch5t/ohta/output/tmp2", f"error_{steps}.pt").as_posix())
                return batch_positions

        f = features.detach().clone().cpu()
        f[:, :, self.stop_token_list] = -np.inf # prevent to generate stop tokens
        greedy = f.argmax(-1) # greedy decoding

        replacement = []
        for k, i in enumerate(resort_batch_index):
            # k: batch index in joeynmt
            # i: current batch index in roberta
            ret = {}
            if i is not None:
                decode = greedy[i].masked_select(mask[i])
                predicted_token_bpe = self.roberta.task.source_dictionary.string(decode)

                predicted_token_word = defaultdict(str)
                for j, t in enumerate(predicted_token_bpe.split()):
                    w = resort_token_index[i][j]
                    predicted_token_word[w] += self.roberta.bpe.decode(t)

                for pos in batch_positions[k].keys(): # assign new word surface
                    word = predicted_token_word[pos] # sentencepiece
                    ret[pos] = word.lower() if self.lowercase else word
            replacement.append(ret)
        #assert len(replacement) == batch_size
        return replacement

    def replace_trg(self, trg_ids_orig, positions, word2bpe, word_cutoff=None) -> torch.Tensor:
        """
        trg replacement in sentence level
        :param trg_ids_orig:
        :param positions:
        :param word2bpe:
        :param word_cutoff:
        :return: trg_ids_aug
        """
        assert trg_ids_orig.dim() == 1, trg_ids_orig.size()
        assert trg_ids_orig[0] == self.bos_id, trg_ids_orig
        assert trg_ids_orig[-1] == self.eos_id, trg_ids_orig

        trg_ids_aug = []
        word2bpe_aug = []
        bpe_index = 1
        for word_pos, bpe_pos in enumerate(word2bpe):
            if word_pos in positions.keys(): # replace
                token_str = positions[word_pos]
                if self.lowercase:
                    token_str = token_str.lower()
                sp = [self.stoi.get(t, self.unk_id) for t
                      in self.spm.encode(token_str, out_type=str)]
            else: # do nothing (just pass through)
                sp = [trg_ids_orig[i] for i in bpe_pos]
            trg_ids_aug.extend(sp)
            word2bpe_aug.append([i+bpe_index for i in range(len(sp))])
            bpe_index += len(sp)
        #print(word2bpe)
        #print(word2bpe_aug)

        if word_cutoff is not None:
            word_start, word_end = word_cutoff
            if word_start is None and word_end is None:
                return torch.Tensor([self.bos_id] + trg_ids_aug + [self.eos_id]).long()
            trg_start = max(word2bpe_aug[word_start][0] - 1, 0)
            trg_end = word2bpe_aug[word_end][-1]
            #print(word_start, word_end, trg_start, trg_end)
            return torch.Tensor([self.bos_id] + trg_ids_aug[trg_start:trg_end] + [self.eos_id]).long()
        return torch.Tensor([self.bos_id] + trg_ids_aug + [self.eos_id]).long()

    def __repr__(self):
        return (self.__class__.__name__
                + f"(language_model_path='{self.model_name}', "
                + f"sentencepiece_model_file='{self.model_file}', "
                + f"len(stop_token_list)={len(self.stop_token_list)}, "
                + f"lowercase='{self.lowercase}', "
                + f"unk_index='{self.unk_id}', "
                + f"bos_index='{self.bos_id}', "
                + f"eos_index='{self.eos_id}', "
                + f"device='{self.device.type}')")


class SubSequencer(object):
    def __init__(self, shape, loc, scale, instance_sample_prob, max_word_length, min_word_length):
        self.length_distribution = partial(lognorm.rvs, s=shape, loc=loc, scale=scale)
        self.instance_sample_prob = instance_sample_prob
        assert 0.0 <= self.instance_sample_prob <= 1.0
        self.max_word_length = max_word_length
        self.min_word_length = min_word_length

    def __call__(self, word2bpe: List[List[List]]) -> (List, List):
        """
        sub-sequencing
        (length based on num of words, not time-frame or num of bpe tokens)
        :param word2bpe:
        :return: tuple(list, list, list)
        """
        word_start = []
        word_end = []
        batch_size = len(word2bpe)

        length = self.length_distribution(size=batch_size)
        cutoff_type = np.random.randint(0, 2, size=batch_size) # 0 -> [:l], 1 -> [j:j+l], 2 -> [-l:]
        coin_flip = np.random.binomial(1, self.instance_sample_prob, batch_size)
        for i in range(batch_size):
            l_orig = len(word2bpe[i]) # original length (num of words)
            l_aug = min(max(int(np.ceil(length[i])), self.min_word_length),
                        self.max_word_length)
            assert self.min_word_length <= l_aug <= self.max_word_length
            # length sampled from the distribution

            cutoff = cutoff_type[i]

            flag = bool(coin_flip[i])
            if l_orig > self.max_word_length:
                flag = True
                l_orig = self.max_word_length
            elif l_orig < self.min_word_length:
                flag = False

            # if sampled length is longer than the original, no cutoff
            if l_aug > l_orig:
                flag = False

            if flag is True:
                if l_orig == l_aug:
                    l_aug -= 1 # need at least 1 diff
                assert l_orig > l_aug

                if cutoff == 0: # 0 -> [:l]
                    word_start.append(0)
                    word_end.append(l_aug)
                elif cutoff == 1: # 1 -> [j:j+l]
                    # start position sampled uniformly
                    j = np.random.randint(0, l_orig - l_aug)
                    word_start.append(j)
                    word_end.append(j + l_aug)
                elif cutoff == 2: # 2 -> [-l:]
                    word_start.append(l_orig - l_aug)
                    word_end.append(l_orig)
                assert word_start[-1] < word_end[-1]
            else:
                word_start.append(None)
                word_end.append(None)

        assert batch_size == len(word_start)
        assert batch_size == len(word_end)

        return word_start, word_end


    def __repr__(self):
        return (self.__class__.__name__
                + f"(length_distribution={self.length_distribution}, "
                + f"max_word_length={self.max_word_length}, "
                + f"min_word_length={self.min_word_length})")
