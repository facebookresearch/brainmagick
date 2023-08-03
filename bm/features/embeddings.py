# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Word embedding features.
"""
import logging
import os
import typing as tp

import spacy
import torch
from bm import events
from bm.cache import Cache, MemoryCache
from bm.utils import Frequency

from . import base

# pylint: disable=import-outside-toplevel

logger = logging.getLogger(__name__)
VALID_SPACY_LANG = {'en': 'en_core_web',  # english
                    'da': 'da_core_news',  # danish
                    'nl': 'nl_core_news',  # dutch
                    'fr': 'fr_core_news',  # french
                    'de': 'de_core_news',  # german
                    'it': 'it_core_news',  # italian
                    'nb': 'nb_core_news',  # norwegian
                    'xx': 'xx_ent_wiki'  # multilingual, only for small model
                    }


class WordEmbedding(base.Feature):
    """
    Dutch word embeddings
    see https://spacy.io/usage/models for valid languages
    """
    event_kind = "word"
    dimension = 300
    model_size = "md"
    # shared between all instances of one process for memory/speed
    _LANG = "auto"

    def __init__(self, sample_rate: Frequency, lang: str = "auto") -> None:
        super().__init__(sample_rate=sample_rate)
        if lang != "auto":
            assert lang in VALID_SPACY_LANG, f"spacy lang should be in {VALID_SPACY_LANG}"
        self.__class__._LANG = lang
        if lang == "xx":
            assert self.model_size == "sm", "Multilingual spacy model only available in small"
        self._model_cache = MemoryCache(self.__class__.__name__)

    @property
    def model_name(self):
        # Lazy attribute because LANG can change on the fly.
        assert self._LANG != "auto", "lang not yet set"
        return f"{VALID_SPACY_LANG[self._LANG]}_{self.model_size}"

    @property
    def cache(self):
        # Lazy attribute because model_name can change on the fly.
        return Cache(self.__class__.__name__, self.model_name)

    @property
    def model(self) -> tp.Any:
        try:
            return self._model_cache.get(spacy.load, name=self.model_name)
        except OSError as e:
            raise OSError(
                f'You need to run "python -m spacy download {self.model_name}"') from e

    def _compute(self, word: str) -> torch.Tensor:
        if not word:
            out = self.default_value
        else:
            out = torch.Tensor(self.model(word)[0].vector)
        return out

    def get(self, event: events.Word) -> torch.Tensor:
        if self._LANG == "auto":
            assert event.language in VALID_SPACY_LANG, f"Invalid lang {event.language}"
            self.__class__._LANG = event.language
        else:
            assert event.language == self._LANG
        return self.cache.get(self._compute, word=event.word)


class WordEmbeddingSmall(WordEmbedding):
    model_size = "sm"
    dimension = 96
    # shared between all instances of one process for memory/speed
    _MODEL = None


class PartOfSpeech(WordEmbedding):
    """Part of speech of the word given by Spacy, categorical"""
    event_kind = "word"
    model_size = "md"
    pos_vocab = ('ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
                 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'EOL', 'SPACE')
    cardinality = len(pos_vocab) + 1
    dimension = 1
    # shared between all instances of one process for memory/speed

    def __init__(self, sample_rate: Frequency, lang: str = "auto") -> None:
        super().__init__(sample_rate=sample_rate, lang=lang)

    def _compute(self, word: str) -> int:
        if not word:
            out = self.default_value
        else:
            pos = self.model(word)[0].pos_
            out = self.pos_vocab.index(pos) + 1  # + 1 for silence
        return int(out)


class BertEmbedding(base.Feature):
    """Multilingual BERT contextual embedding"""
    event_kind = "word"
    dimension = 768
    model_name = "bert-base-multilingual-cased"
    # shared between all instances of one process for memory/speed

    def __init__(self, sample_rate: Frequency, device: str = "cpu",
                 layers: tp.Tuple[int, ...] = (8, 9, 10)) -> None:
        super().__init__(sample_rate=sample_rate)
        self.cache = Cache(self.__class__.__name__)
        self.device = device
        self.layers = layers  # layer to extract the embedding from (averaged))
        # Disable Huggingface logging
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TRANSFORMERS_VERBOSITY"] = "critical"
        self._model_cache = MemoryCache(self.__class__.__name__, "model")
        self._tokenizer_cache = MemoryCache(self.__class__.__name__, "tokenizer")

    def _get_hiddens(self, string: str) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        string: one str contextual sequence
        Returns:
            - hiddens: torch.Tensor of shape (n_tokens, dim)
            - offsets: torch.LongTensor of shape (n_tokens, 2)
            For each token, the start character and end character
            that corresponds to the token.
        """
        # Tokenize
        inputs = self.tokenizer(string,
                                return_offsets_mapping=True,
                                return_tensors="pt",
                                add_special_tokens=True)

        self.model.to(self.device)
        # Compute hidden states for the sequence
        with torch.no_grad():
            out = self.model(inputs["input_ids"].to(self.model.device),
                             output_hidden_states=True)
            hiddens = torch.stack(out.hidden_states).to("cpu")
            hiddens = hiddens[:, 0]  # only one sentence here
            if self.layers is not None:
                hiddens = hiddens[self.layers, :]
            hiddens = hiddens.mean(0)  # of shape (n_tok, dim)
            assert hiddens.shape[-1] == self.dimension

            # Keep track of start_char and end_char for each word
            offsets = inputs.offset_mapping[0, :, 1].to("cpu")  # only one sentence

        return hiddens, offsets

    @property
    def model(self) -> tp.Any:
        from transformers import AutoModel
        return self._model_cache.get(AutoModel.from_pretrained, self.model_name)

    @property
    def tokenizer(self) -> tp.Any:
        from transformers import AutoTokenizer
        return self._tokenizer_cache.get(AutoTokenizer.from_pretrained, self.model_name)

    def get(self, event: events.Word) -> torch.Tensor:
        if not event.word:
            out = self.default_value
        else:
            hiddens, offsets = self.cache.get(self._get_hiddens, string=event.word_sequence)
            wid = event.word_index
            try:
                tokens = event.word_sequence.split(" ")
                assert tokens[wid] == event.word
                char_end = len(" ".join(tokens[:(wid + 1)]))
                char_start = char_end - len(event.word)
                assert event.word_sequence[char_start:char_end] == event.word
                start_token = torch.where(offsets > char_start)[0][0]
                end_token = torch.where(offsets >= char_end)[0][0] + 1
            except AssertionError:
                logger.info(
                    f"Bad word_index for word {event.word} in sequence {event.word_sequence}")
                start_token = 0
                end_token = len(hiddens)

            # Sum because keep information on the wordlen
            # We could average also
            out = hiddens[start_token:end_token].sum(0)
        return out


class XlmEmbedding(base.Feature):
    """XLM raw or contextual embeddings

    Parameter
    ---------
    contextual: bool
        use the contextual (final layer) embeddings instead of the raw (first layer)
        embeddings.
    """
    dimension = 1024
    event_kind = "word"
    _XLMR: tp.Any = None

    def __init__(self, sample_rate: Frequency, contextual: bool = False) -> None:
        super().__init__(sample_rate=sample_rate)
        self.contextual = contextual
        self.cache = Cache(self.__class__.__name__, self.contextual)

    def _compute(self, string: str) -> torch.Tensor:
        if self._XLMR is None:
            self.__class__._XLMR = torch.hub.load('pytorch/fairseq', 'xlmr.large')
            self.__class__._XLMR.eval()
        xlmr = self.__class__._XLMR
        # tokenify each word one by one so as to record
        # which token(s) they are mapped to
        words = string.split(" ")
        parts: tp.List[torch.Tensor] = []
        affectations = []
        for k, word in enumerate(words):
            wtokens = xlmr.encode(word)
            if not parts:  # add initial token
                parts.append(wtokens[:1])
            parts.append(wtokens[1:-1])
            affectations.extend([k] * parts[-1].shape[0])
        parts.append(wtokens[-1:])  # add final token
        tokens = torch.cat(parts)
        # compute outputs
        with torch.no_grad():
            all_embs = xlmr.extract_features(tokens, return_all_hiddens=True)
        embs = (all_embs[-1] if self.contextual else all_embs[0])
        embs = embs[0, 1:-1, :]  # remove bondaries which are not used
        assert embs.shape[0] == len(affectations)
        return embs, torch.Tensor(affectations)

    def get(self, event: events.Word) -> torch.Tensor:
        embs, affect = self.cache.get(self._compute, string=event.word_sequence)
        inds = affect == event.word_index
        # sum and renormalize if the word corresponds to several tokens
        return embs[inds, :].sum(axis=0) / torch.sqrt(sum(inds))
