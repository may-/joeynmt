#!/usr/bin/env python3

# Adapted from https://github.com/pytorch/fairseq/blob/master/examples/speech_to_text/prep_mustc_data.py


import argparse
import logging
import csv
from pathlib import Path
from itertools import groupby
from functools import partial
from typing import Tuple
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import yaml
import numpy as np
import pandas as pd
import sentencepiece as sp

from torch import Tensor
from torch.utils.data import Dataset
import torchaudio

from audiodata_utils import create_zip, extract_fbank_features, \
    get_zip_manifest, build_sp_model, Normalizer, save_tsv, get_n_frames

logger = logging.getLogger(__name__)


COLUMNS = ["id", "src", "n_frames", "trg", "speaker"]

N_MEL_FILTERS = 80
N_WORKERS = 4 #cpu_count()
SP_MODEL_TYPE = "unigram" # one of ["bpe", "unigram", "char"]
VOCAB_SIZE = 20000 #joint vocab
LOWERCASE = True

class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, num of frames, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["dev", "tst-COMMON", "tst-HE", "train"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru"]
    SAMPLE_RATE = 16000

    def __init__(self, root: Path, lang: str, split: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = Path(root) / f"en-{lang}" / "data" / split
        wav_root, txt_root = _root / "wav", _root / "txt"
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        with (txt_root / f"{split}.yaml").open("r") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in ["en", lang]:
            with (txt_root / f"{split}.{_lang}").open("r") as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            seg_group = sorted(_seg_group, key=lambda x: float(x["offset"]))
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * self.SAMPLE_RATE)
                duration = int(float(segment["duration"]) * self.SAMPLE_RATE)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        duration,
                        segment["en"],
                        segment[lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

        self.feature_root = Path(root) / f"en-{lang}" / f"fbank{N_MEL_FILTERS}"
        self.feature_root.mkdir(exist_ok=True)
        self.return_wav = True

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, str, str]:
        wav_path, offset, duration, src_utt, tgt_utt, spk_id, utt_id = self.data[n]
        if self.return_wav:
            waveform, sr = torchaudio.load(
                wav_path, frame_offset=offset, num_frames=duration)
            assert sr == self.SAMPLE_RATE
            assert duration == waveform.size(1), (duration, waveform.size(1))
            n_frames = get_n_frames(waveform, sr)
        else:
            waveform = None
            features = np.load((self.feature_root / f"{utt_id}.npy").as_posix())
            n_frames = features.shape[0]
            assert n_frames > 0, utt_id
        return waveform, n_frames, src_utt, tgt_utt, spk_id, utt_id

    def __len__(self) -> int:
        return len(self.data)


def process(data_root, languages):
    root = Path(data_root).absolute()
    for lang in languages:
        assert lang in MUSTC.LANGUAGES
        cur_root = root / f"en-{lang}"

        # normalizer
        normalizer = {'en': Normalizer(lang='en', lowercase=LOWERCASE,
                                       remove_punc=True, normalize_num=True,
                                       mapping_path=(cur_root/"mapping_en.txt"),
                                       escape=True),
                      lang: Normalizer(lang=lang, lowercase=False,
                                       remove_punc=False, normalize_num=False,
                                       escape=True)}

        # Extract features
        print(f"Create en-{lang} dataset.")
        for split in MUSTC.SPLITS:
            print(f"Fetching split {split}...")
            dataset = MUSTC(root, lang, split)
            print(f"Extracting log mel filter bank features ...")
            for wav, n_frames, _, _, spk_id, utt_id in tqdm(dataset):
                extract_fbank_features(wav, n_frames, utt_id,
                                       sample_rate=MUSTC.SAMPLE_RATE,
                                       feature_root=dataset.feature_root,
                                       n_mel_bins=N_MEL_FILTERS,
                                       overwrite=False)
        
        # Pack features into ZIP
        feature_root = cur_root / f"fbank{N_MEL_FILTERS}"
        print("ZIPing features...")
        create_zip(feature_root, feature_root.with_suffix(".zip"))
        print("Fetching ZIP manifest...")
        zip_manifest = get_zip_manifest(feature_root.with_suffix(".zip"))
        # Generate TSV manifest
        print("Generating manifest...")
        all_data = []
        with tqdm(total=len(zip_manifest)) as pbar:
            for split in MUSTC.SPLITS:
                dataset = MUSTC(root, lang, split)
                dataset.return_wav = False  # a bit faster...
                for _, n_frames, src_utt, trg_utt, spk_id, utt_id in dataset:
                    record = {
                        "id": utt_id,
                        "src": zip_manifest[utt_id],
                        "n_frames": n_frames,
                        "src_orig": src_utt,
                        "trg_orig": trg_utt,
                        "src_utt": normalizer['en'](src_utt),
                        "trg_utt": normalizer[lang](trg_utt),
                        "speaker": spk_id,
                        "split": split
                    }
                    all_data.append(record)
                    pbar.update(1)
            all_df = pd.DataFrame.from_records(all_data)
            save_tsv(all_df, (cur_root/'all_data_tmp.tsv'))

        # Generate joint vocab
        print("Building joint vocab...")
        
        raw_textfile = cur_root / f"train.en{lang}"
        train_df = all_df[all_df.split == "train"]
        train = pd.concat([train_df.src_utt, train_df.trg_utt])
        save_tsv(train, raw_textfile, header=False)

        spm_filename = cur_root / f"spm_{SP_MODEL_TYPE}{VOCAB_SIZE}"
        symbols = set([x[1] for x in (normalizer["en"].escape
                                      + normalizer[lang].escape)])
        kwargs = {'model_type': SP_MODEL_TYPE,
                  'vocab_size': VOCAB_SIZE,
                  'character_coverage': 1.0,
                  'num_workers': N_WORKERS,
                  'user_defined_symbols': ','.join(symbols)}
        spm = build_sp_model(raw_textfile, spm_filename, **kwargs)
        print("Applying vocab ...")
        for split in MUSTC.SPLITS:
            split_df = all_df[all_df.split == split]
            for _side, _task, _lang in [("src", "asr", "en"), ("trg", "st", lang)]:
                # apply sentencepiece
                split_df[f"{_side}_tok"] = split_df[f"{_side}_utt"].apply(
                    lambda x: ' '.join(spm.encode(x, out_type=str)))
                # save tsv file
                save_tsv(split_df.rename(columns={f'{_side}_tok': 'trg'})[COLUMNS],
                         cur_root / f"{split}_{_task}_spm{VOCAB_SIZE}.tsv")
                # save text file (for mt pretraining)
                save_tsv(split_df[f"{_side}_tok"],
                         cur_root / f"{split}_spm{VOCAB_SIZE}.{_lang}",
                         header=False)
            print(f'\t{split} tsv saved.')
        print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--lang", default="de", required=True, type=str, nargs='+')
    args = parser.parse_args()

    process(args.data_root, args.lang)


if __name__ == "__main__":
    main()
