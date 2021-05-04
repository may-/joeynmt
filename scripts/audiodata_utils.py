#!/usr/bin/env python3

# Adapted from https://github.com/pytorch/fairseq/blob/master/examples/speech_to_text/data_utils.py

import re
import io
import csv
import string
import itertools
from typing import Optional, Callable
from tqdm import tqdm
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import sentencepiece as sp

import torch
import torchaudio
import torchaudio.compliance.kaldi as ta_kaldi
import torchaudio.sox_effects as ta_sox

from joeynmt.constants import BOS_TOKEN, PAD_TOKEN, EOS_TOKEN, UNK_TOKEN, \
    UNK_ID, EOS_ID, BOS_ID, PAD_ID
from joeynmt.helpers_for_audio import _is_npy_data, _get_features_from_zip


def _get_torchaudio_fbank(waveform, sample_rate, n_bins=80) -> np.ndarray:
    """
    Get mel-filter bank features via TorchAudio.
    """
    waveform = torch.from_numpy(waveform)
    if len(waveform.shape) == 1:
        # Mono channel: D -> 1 x D
        waveform = waveform.unsqueeze(0)
    else:
        # Merge multiple channels to one: C x D -> 1 x D
        waveform, _ = ta_sox.apply_effects_tensor(
            waveform, sample_rate, ['channels', '1'])

    features = ta_kaldi.fbank(
        waveform, num_mel_bins=n_bins, sample_frequency=sample_rate)
    return features.numpy()


def extract_fbank_features(waveform, n_frames, utt_id,
                           feature_root: Optional[Path] = None,
                           sample_rate: int = 16000,
                           n_mel_bins: int = 80,
                           overwrite: bool = False) -> Optional[np.ndarray]:
    # pylint: disable=inconsistent-return-statements

    output_path = None
    if feature_root is not None:
        output_path = feature_root / f"{utt_id}.npy"

    if output_path is not None and output_path.is_file() and not overwrite:
        return

    _waveform = waveform * (2 ** 15)  # Kaldi compliance: 16-bit signed integers
    _waveform = _waveform.squeeze().numpy()

    try:
        features = _get_torchaudio_fbank(_waveform, sample_rate, n_mel_bins)
        assert abs(features.shape[0] - n_frames) <= 1
    except Exception as e:
        raise ValueError(f"torchaudio faild to extract mel filterbank features "
                         f"at {utt_id}. features.shape: {features.shape}, "
                         f"n_frames: {n_frames}, n_mel_bins: {n_mel_bins} {e}")

    if output_path is not None:
        np.save(output_path.as_posix(), features)
    else:
        return features


def get_zip_manifest(zip_path: Path, npy_root: Optional[Path] = None):
    manifest = {}
    with zipfile.ZipFile(zip_path, mode="r") as f:
        info = f.infolist()
    error_flag = []
    for i in tqdm(info):
        utt_id = Path(i.filename).stem
        offset, file_size = i.header_offset + 30 + len(i.filename), i.file_size
        with zip_path.open("rb") as f:
            f.seek(offset)
            data = f.read(file_size)
            try:
                assert len(data) > 1 and _is_npy_data(data), (utt_id, len(data), e)
            except Exception as e:
                print((utt_id, len(data), e))
                error_flag.append((utt_id, len(data)))
        manifest[utt_id] = f"{zip_path.name}:{offset}:{file_size}"
        # sanity check
        if npy_root is not None:
            byte_data = np.load(io.BytesIO(data))
            npy_data = np.load((npy_root / f"{utt_id}.npy").as_posix())
            assert np.allclose(byte_data, npy_data)
    if len(error_flag) > 0:
        print(error_flag)
        raise Exception
    return manifest


def get_n_frames(waveform, sample_rate):
    duration_ms = int(waveform.size(1) / sample_rate * 1000)
    n_frames = int(1 + (duration_ms - 25) / 10)
    return n_frames


def create_zip(data_root: Path, zip_path: Path):
    paths = list(data_root.glob("*.npy"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as f:
        for path in tqdm(paths):
            try:
                f.write(path, arcname=path.name)
            except Exception as e:
                raise Exception(f"{path} {e}")


def save_tsv(df: pd.DataFrame, path: Path, header: bool = True) -> None:
    df.to_csv(path.as_posix(), sep="\t", header=header, index=False,
              encoding="utf-8", escapechar="\\", quoting=csv.QUOTE_NONE)


def build_sp_model(input_path: Path, model_path_prefix: Path, **kwargs):
    """
    Build sentencepiece model
    """
    # Train SentencePiece Model
    arguments = [
        f"--input={input_path.as_posix()}",
        f"--model_prefix={model_path_prefix.as_posix()}",
        f"--model_type={kwargs.get('model_type', 'unigram')}",
        f"--vocab_size={kwargs.get('vocab_size', 5000)}",
        f"--character_coverage={kwargs.get('character_coverage', 1.0)}",
        f"--num_threads={kwargs.get('num_workers', 1)}",
        f"--unk_id={UNK_ID}",
        f"--bos_id={BOS_ID}",
        f"--eos_id={EOS_ID}",
        f"--pad_id={PAD_ID}"
    ]
    if 'user_defined_symbols' in kwargs.keys():
        arguments.append(f"--user_defined_symbols={kwargs['user_defined_symbols']}")
    sp.SentencePieceTrainer.Train(" ".join(arguments))

    # Export vocab
    spm = sp.SentencePieceProcessor()
    spm.Load(model_path_prefix.with_suffix(".model").as_posix())
    itos = {i: spm.IdToPiece(i) for i in range(spm.GetPieceSize())}
    assert (itos.get(UNK_ID) == UNK_TOKEN
            and itos.get(PAD_ID) == PAD_TOKEN
            and itos.get(BOS_ID) == BOS_TOKEN
            and itos.get(EOS_ID) == EOS_TOKEN)
    stoi = {s: i for i, s in itos.items()
            if s not in {UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN}}
    with model_path_prefix.with_suffix(".txt").open("w") as f_out:
        for s, _ in sorted(stoi.items(), key=lambda x: x[1]):
            f_out.write(f"{s}\n")
    return spm


class Normalizer:
    MAPPING = {
        'en': {
            '%': 'percent',
            '&': 'and',
            '=': 'equal to',
            '@': 'at'
        },
        'de': {
            '€': 'Euro'
        }
    }
    ESCAPE = {
        'en': [
            ('(noise)', '<noise>'),
            ('[unclear]', '<unclear>'),
            ('(applause)', '<applause>'),
            ('(laughter)', '<laughter>'),
            ('(laughing)', '<laughter>'),
            ('(laughs)', '<laughter>'),
        ],
        'de': [
            ('(Geräusch)', '<Geräusch>'),
            ('[unklar]', '<unklar>'),
            ('(Lachen)', '<Lachen>'),
            ('(Lacht)', '<Lachen>'),
            ('(lacht)', '<Lachen>'),
            ('(Gelächter)', '<Lachen>'),
            ('(Gelaechter)', '<Lachen>'),
            ('(Applaus)', '<Applaus>'),
            ('(Applause)', '<Applaus>'),
            ('(Beifall)', '<Applaus>'),
        ]
    }
    def __init__(self, lang: str = "en", lowercase: bool = True,
                 remove_punc: bool = True, normalize_num: bool = True,
                 mapping_path: Path = None, escape: bool = False):
        try:
            import sacremoses
            from sacremoses.normalize import MosesPunctNormalizer
        except:
            raise ImportError

        self.moses = MosesPunctNormalizer(lang)
        self.lowercase = lowercase
        self.remove_punc = remove_punc
        self.normalize_num = normalize_num
        self.lang = lang

        if normalize_num:
            try:
                import inflect
                self.inflect = inflect.engine()
            except:
                raise ImportError

        self.escape = self.ESCAPE[lang] if escape else None
        self.mapping = self.MAPPING[lang]
        if mapping_path:
            self.mapping_num = {}
            with Path(mapping_path).open('r') as f:
                for line in f.readlines():
                    l = line.strip().split('\t')
                    self.mapping_num[l[0]] = l[1]
        # mapping.txt (one word per line)
        ########## format:
        # orig_surface [TAB] replacement
        ########## examples:
        # g7	g seven
        # 11pm	eleven pm
        # 6am	six am
        ##########

    def _years(self, word):
        num_word = word
        s_flag = False
        if num_word.endswith("'s"):
            s_flag = True
            num_word = num_word[:-2]
        elif num_word.endswith('s'):
            s_flag = True
            num_word = num_word[:-1]

        if len(num_word) in [1, 3, 5]:
            num_word = self.inflect.number_to_words(num_word)
            if s_flag: # 1s or 100s or 10000s
                num_word += ' s'
            s_flag = False

        if len(num_word) == 2: # 50s
            try:
                w = int(num_word)
                num_word = self.inflect.number_to_words(w)
            except:
                s_flag = False

        elif len(num_word) == 4:
            try:
                w = int(num_word)

                if word.endswith('000'):
                    num_word = self.inflect.number_to_words(num_word)
                elif num_word.endswith('00'): ##
                    w1 = int(num_word[:2])
                    num_word = f"{self.inflect.number_to_words(w1)} hundred"
                elif 2000 < w < 2010:
                    num_word = self.inflect.number_to_words(num_word, andword="")
                else:
                    num_word = self.inflect.number_to_words(num_word, group=2)
            except:
                s_flag = False

        if s_flag:
            w = num_word.rsplit(' ', 1)
            num_word = self.inflect.plural(w[-1])
            if len(w) > 1:
                num_word = f"{w[0]} {num_word}"

        return num_word.lower() if self.lowercase else num_word

    def __call__(self, orig_utt):
        utt = orig_utt.lower() if self.lowercase else orig_utt
        utt = self.moses.normalize(utt)

        for k, v in self.mapping.items():
            utt = utt.replace(k, f" {v} ")

        if self.normalize_num and self.lang == "en":
            utt = utt.replace('-', ' ')
            matched_iter = re.finditer(r'([^ ]*\d+[^ ]*)', utt)

            try:
                first_match = next(matched_iter)
            except StopIteration:
                pass # if no digits, do nothing
            else:
                current_position = 0
                utterance = []

                for m in itertools.chain([first_match], matched_iter):
                    start = m.start()
                    word = m.group().strip(string.punctuation)
                    before = utt[current_position:start]
                    if len(before) > 0:
                        utterance.append(before)

                    if word in self.mapping_num.keys():
                        num_word = self.mapping_num[word]
                    else:
                        num_word = self._years(word)
                        if num_word == word:
                            num_word = self.inflect.number_to_words(num_word, andword="")

                    if len(utterance) > 0 and not utterance[-1].endswith(' '):
                        num_word = ' '+num_word
                    utterance.append(num_word)
                    current_position += start + len(word)

                if current_position < len(utt):
                    utterance.append(utt[current_position:])
                utt = ''.join(utterance)

        if self.escape is not None:
            for k, v in self.escape:
                utt = utt.replace(k, v)

            utt = re.sub(r'\([^)]+\)', self.escape[0][1], utt)
            utt = re.sub(r'\[[^\]]+\]', self.escape[1][1], utt)

        utt = re.sub(r'(\([^)]+\)|\[[^\]]+\])', ' ', utt)

        valid_char = ' a-z'
        if self.lang == 'de':
            valid_char += 'äöüß'

        if not self.normalize_num:
            valid_char += '0-9'

        if not self.lowercase:
            valid_char += 'A-Z'
            if self.lang == 'de':
                valid_char += 'ÄÖÜ'

        if self.remove_punc:
            valid_char += '\''
        else:
            valid_char += string.punctuation

        if self.escape is not None:
            valid_char += '<>'
        utt = re.sub(r'[^' + valid_char + ']', ' ', utt)
        utt = re.sub(r'( )+', ' ', utt)

        if self.lowercase:
            utt.lower()
        return utt.strip()
