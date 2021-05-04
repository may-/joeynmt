#!/usr/bin/env python3
"""
prepare LibriSpeech dataset
"""
# Adapted from https://github.com/pytorch/fairseq/blob/master/examples/speech_to_text/prep_librispeech_data.py

import argparse
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from torch import Tensor
import torchaudio
from torchaudio.datasets import LIBRISPEECH as LIBRISPEECH_torchaudio
from torchaudio.datasets.librispeech import FOLDER_IN_ARCHIVE

from audiodata_utils import create_zip, extract_fbank_features, \
    get_zip_manifest, build_sp_model, save_tsv, get_n_frames

logger = logging.getLogger(__name__)


# pre-defined parameters
N_MEL_FILTERS = 80
N_WORKERS = 16
SP_MODEL_TYPE = "unigram" # one of ["bpe", "unigram", "char"]
VOCAB_SIZE = {"train-clean-100": 5000, "train-960": 10000}


class LIBRISPEECH(LIBRISPEECH_torchaudio):
    SPLITS = [
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    ]
    SAMPLE_RATE = 16000

    def __init__(self,
                 root: Path,
                 split: str = "train-clean-100",
                 download: bool = False):
        super().__init__(root, split, FOLDER_IN_ARCHIVE, download)
        self.return_wav = True
        self.return_utt = True
        self.feature_root = root / f"fbank{N_MEL_FILTERS}"
        self.feature_root.mkdir(exist_ok=True)

    def _load_text(self, text_file, file_id):
        with text_file.open("r") as ft:
            for line in ft:
                text_id, utterance = line.strip().split(" ", 1)
                if file_id == text_id:
                    break
            else:
                # Translation not found
                raise FileNotFoundError("Translation not found for " + file_id)
        return utterance

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, str]:
        file_id = self._walker[n]
        spk_id, cpt_id, utt_no = file_id.split("-")
        utt_id = f"{spk_id}-{cpt_id}-{int(utt_no)}"
        path = Path(self._path)

        if self.return_utt:
            text_file = (path / spk_id / cpt_id
                         / f"{spk_id}-{cpt_id}").with_suffix(self._ext_txt)
            utt = self._load_text(text_file, file_id)
        else:
            utt = None

        if self.return_wav:
            audio_file = (path / spk_id / cpt_id
                          / file_id).with_suffix(self._ext_audio)
            waveform, sample_rate = torchaudio.load(audio_file.as_posix())
            assert self.SAMPLE_RATE == int(sample_rate)
            n_frames = get_n_frames(waveform, sample_rate)
        else:
            waveform = None
            features = np.load((self.feature_root / f"{utt_id}.npy").as_posix())
            n_frames = features.shape[0]
            assert n_frames > 0, utt_id

        return waveform, n_frames, utt, spk_id, utt_id


def process(output_root):
    out_root = Path(output_root).absolute()
    out_root.mkdir(exist_ok=True)

    # Extract features
    for split in LIBRISPEECH.SPLITS:
        print(f"Fetching split {split}...")
        dataset = LIBRISPEECH(out_root, split=split, download=True)
        dataset.return_utt = False  # a bit faster...
        print(f"Extracting log mel filter bank features...")
        for waveform, n_frames, _, _, utt_id in tqdm(dataset):
            extract_fbank_features(waveform, n_frames, utt_id,
                                   sample_rate=LIBRISPEECH.SAMPLE_RATE,
                                   feature_root=dataset.feature_root,
                                   n_mel_bins=N_MEL_FILTERS,
                                   overwrite=False)
    
    # Pack features into ZIP
    feature_root = out_root / f"fbank{N_MEL_FILTERS}"   # dataset.feature_root
    print("ZIPing features...")
    create_zip(feature_root, feature_root.with_suffix(".zip"))
    print("Fetching ZIP manifest...")
    #zip_manifest = get_zip_manifest(feature_root.with_suffix(".zip"))
    zip_manifest = get_zip_manifest(feature_root.with_suffix(".zip"),
                                    npy_root=feature_root)
    
    # Generate TSV manifest
    print("Generating manifest...")
    all_data = []
    with tqdm(total=len(zip_manifest)) as pbar:
        for split in LIBRISPEECH.SPLITS:
            dataset = LIBRISPEECH(out_root, split=split)
            dataset.return_wav = False  # a bit faster...
            for _, n_frames, utt, spk_id, utt_id in dataset:
                record = {"id": utt_id,
                          "src": zip_manifest[utt_id],
                          "n_frames": n_frames,
                          "trg_text": utt,  # uppercase
                          "split": split,
                          "speaker": spk_id}
                all_data.append(record)
                pbar.update(1)
        all_df = pd.DataFrame.from_records(all_data)
        save_tsv(all_df, out_root / f'all_data_tmp.tsv')

    #all_df = pd.read_csv((out_root / f'all_data_tmp.tsv').as_posix(), sep='\t',
    #                 encoding="utf-8", escapechar="\\", quoting=csv.QUOTE_NONE)

    print("Saving in TSV...")
    trainsets = {
        "train-clean-100": ["train-clean-100"],
        "train-960": ["train-clean-100", "train-clean-360", "train-other-500"]}

    for trainset, train_splits in trainsets.items():
        vocab_size = VOCAB_SIZE[trainset]
        spm_filename = out_root / f"spm_{SP_MODEL_TYPE}{vocab_size}"
        raw_textfile = out_root / f"{trainset}.en"
        train_df = all_df[all_df.split.isin(train_splits)]
        trg = train_df["trg_text"].str.lower()  # lowercase
        save_tsv(trg, raw_textfile, header=False)

        print(f"\tBuilding vocab of {trainset}...")
        kwargs = {'model_type': SP_MODEL_TYPE,
                  'vocab_size': vocab_size,
                  'character_coverage': 1.0,
                  'num_workers': N_WORKERS}
        spm = build_sp_model(raw_textfile, spm_filename, **kwargs)

        print(f"\tApplying vocab of {trainset}...")
        for split in [train_splits, "dev-clean", "dev-other",
                                    "test-clean", "test-other"]:
            if isinstance(split, list):
                df = all_df[all_df.split.isin(split)]
                out_filename = trainset
            elif isinstance(split, str):
                df = all_df[all_df.split == split]
                out_filename = split
            df["trg"] = df["trg_text"].apply(
                lambda x: ' '.join(spm.encode(x.lower(), out_type=str)))
            tsv_file = out_root / f"{out_filename}_spm{vocab_size}.tsv"
            save_tsv(df, tsv_file)
            print(f'\t{tsv_file} saved.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    args = parser.parse_args()

    process(args.data_root)


if __name__ == "__main__":
    main()
