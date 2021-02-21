# coding: utf-8
"""
This module holds various MT evaluation metrics.
"""
import unicodedata

from typing import List
import sacrebleu
import editdistance


def chrf(hypotheses, references, remove_whitespace=True):
    """
    Character F-score from sacrebleu

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param remove_whitespace: (bool)
    :return:
    """
    return sacrebleu.corpus_chrf(hypotheses=hypotheses, references=[references],
                                 remove_whitespace=remove_whitespace).score


def bleu(hypotheses, references, tokenize="13a"):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param tokenize: one of {'none', '13a', 'intl', 'zh', 'ja-mecab'}
    :return:
    """
    return sacrebleu.corpus_bleu(sys_stream=hypotheses,
                                 ref_streams=[references],
                                 tokenize=tokenize).score


def token_accuracy(hypotheses: List[List[str]], references: List[List[str]]) \
        -> float:
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses: list of tokenized hypotheses (List[List[str]])
    :param references: list of tokenized references (List[List[str]])
    :return: token accuracy (float)
    """
    correct_tokens = 0
    all_tokens = 0
    assert len(hypotheses) == len(references)
    for hyp, ref in zip(hypotheses, references):
        all_tokens += len(hyp)
        for h_i, r_i in zip(hyp, ref):
            # min(len(h), len(r)) tokens considered
            if h_i == r_i:
                correct_tokens += 1
    return (correct_tokens / all_tokens)*100 if all_tokens > 0 else 0.0


def sequence_accuracy(hypotheses, references):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    assert len(hypotheses) == len(references)
    correct_sequences = sum([1 for (hyp, ref) in zip(hypotheses, references)
                             if hyp == ref])
    return (correct_sequences / len(hypotheses))*100 if hypotheses else 0.0


def wer(hypotheses, references, tokenizer=None):
    """
    Compute word error rate

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param tokenizer: tokenize function (callable)
    :return: normalized word error rate
    """
    numerator = 0.0
    denominator = 0.0

    # sentence-level wer
    #for hyp, ref in zip(hypotheses, references):
    #    wer = editdistance.eval(tokenizer(hyp),
    #                            tokenizer(ref)) / len(tokenizer(ref))
    #    numerator += max(wer, 1.0) # can be `wer > 1` if `len(hyp) > len(ref)`
    #    denominator += 1.0
    # corpus-level wer
    for hyp, ref in zip(hypotheses, references):
        numerator += editdistance.eval(tokenizer(hyp), tokenizer(ref))
        denominator += len(tokenizer(ref))

    return (numerator / denominator) * 100 if denominator else 0.0

# from fairseq
class EvaluationTokenizer:
    """A generic evaluation-time tokenizer, which leverages built-in tokenizers
    in sacreBLEU (https://github.com/mjpost/sacrebleu). It additionally provides
    lowercasing, punctuation removal and character tokenization, which are
    applied after sacreBLEU tokenization.

    :param tokenize: (str) the type of sacreBLEU tokenizer to apply.
    :param lowercase: (bool) lowercase the text.
    :param remove_punctuation: (bool) remove punctuation (based on unicode
        category) from text.
    :param level: (str) tokenization level. {"word", "bpe", "char"}
    """

    SPACE = chr(32)
    SPACE_ESCAPE = chr(9601)
    ALL_TOKENIZER_TYPES = ["none", "13a", "intl", "zh", "ja-mecab"]

    def __init__(
            self,
            tokenize: str = "13a",
            lowercase: bool = False,
            remove_punctuation: bool = False,
            level: str = "word",
    ):
        from sacrebleu.tokenizers import TOKENIZERS

        assert tokenize in self.ALL_TOKENIZER_TYPES, f"{tokenize}, {TOKENIZERS}"
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.character_tokenization = (level == "char")
        self.tokenizer = TOKENIZERS[tokenize]

    @classmethod
    def remove_punc(cls, sent: str):
        """Remove punctuation based on Unicode category."""
        return cls.SPACE.join(
            t
            for t in sent.split(cls.SPACE)
            if not all(unicodedata.category(c)[0] == "P" for c in t)
        )

    def tokenize(self, sent: str):
        tokenized = self.tokenizer()(sent)

        if self.remove_punctuation:
            tokenized = self.remove_punc(tokenized)

        if self.character_tokenization:
            tokenized = self.SPACE.join(
                list(tokenized.replace(self.SPACE, self.SPACE_ESCAPE))
            )

        if self.lowercase:
            tokenized = tokenized.lower()

        return tokenized
