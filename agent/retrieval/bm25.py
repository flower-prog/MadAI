from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any, Callable
import math
import re


def tokenize_bm25_text(text: str) -> list[str]:
    normalized = str(text or "").replace("_", " ").lower()
    tokens: list[str] = []
    for chunk in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized):
        if re.fullmatch(r"[a-z0-9]+", chunk):
            tokens.append(chunk)
            continue

        max_ngram = min(len(chunk), 6)
        for ngram_size in range(1, max_ngram + 1):
            for start in range(0, len(chunk) - ngram_size + 1):
                tokens.append(chunk[start : start + ngram_size])
        if len(chunk) > max_ngram:
            tokens.append(chunk)
    return tokens


class FieldedBM25Index:
    def __init__(
        self,
        documents: list[dict[str, str]],
        *,
        field_weights: dict[str, float],
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.documents = list(documents)
        self.field_weights = {
            field_name: max(float(weight), 0.0)
            for field_name, weight in dict(field_weights).items()
        }
        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer = tokenizer or tokenize_bm25_text
        self.field_names = tuple(self.field_weights.keys())
        self._field_term_frequencies: dict[str, list[Counter[str]]] = {}
        self._field_lengths: dict[str, list[int]] = {}
        self._field_avg_length: dict[str, float] = {}
        self._field_idf: dict[str, dict[str, float]] = {}
        self._build()

    def score(self, query: str) -> list[float]:
        score_by_index = self.score_subset(query)
        scores = [0.0 for _ in self.documents]
        for document_index, score in score_by_index.items():
            if 0 <= document_index < len(scores):
                scores[document_index] = float(score)
        return scores

    def score_subset(
        self,
        query: str,
        *,
        document_indexes: Iterable[int] | None = None,
    ) -> dict[int, float]:
        target_indexes = self._normalize_document_indexes(document_indexes)
        query_terms = self.tokenizer(query)
        if not query_terms:
            return {index: 0.0 for index in target_indexes}

        query_term_counts = Counter(query_terms)
        scores = {index: 0.0 for index in target_indexes}
        for field_name in self.field_names:
            field_weight = float(self.field_weights.get(field_name) or 0.0)
            if field_weight <= 0.0:
                continue
            term_frequencies = self._field_term_frequencies.get(field_name) or []
            field_lengths = self._field_lengths.get(field_name) or []
            avg_length = float(self._field_avg_length.get(field_name) or 0.0)
            idf_map = self._field_idf.get(field_name) or {}

            for doc_index in target_indexes:
                if doc_index >= len(term_frequencies):
                    continue
                term_frequency = term_frequencies[doc_index]
                if not term_frequency:
                    continue
                document_length = int(field_lengths[doc_index]) if doc_index < len(field_lengths) else 0
                field_score = 0.0
                for term, query_count in query_term_counts.items():
                    tf = int(term_frequency.get(term) or 0)
                    if tf <= 0:
                        continue
                    idf = float(idf_map.get(term) or 0.0)
                    denominator = tf + self.k1 * (
                        1.0 - self.b + (self.b * self._normalized_length(document_length, avg_length))
                    )
                    if denominator <= 0.0:
                        continue
                    field_score += query_count * idf * ((tf * (self.k1 + 1.0)) / denominator)
                scores[doc_index] = float(scores.get(doc_index) or 0.0) + (field_weight * field_score)
        return scores

    def _normalize_document_indexes(self, document_indexes: Iterable[int] | None) -> list[int]:
        if document_indexes is None:
            return list(range(len(self.documents)))

        normalized_indexes: list[int] = []
        seen: set[int] = set()
        for raw_index in list(document_indexes):
            try:
                document_index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if document_index < 0 or document_index >= len(self.documents):
                continue
            if document_index in seen:
                continue
            seen.add(document_index)
            normalized_indexes.append(document_index)
        return normalized_indexes

    def _build(self) -> None:
        document_count = len(self.documents)
        for field_name in self.field_names:
            term_frequencies: list[Counter[str]] = []
            lengths: list[int] = []
            document_frequency: Counter[str] = Counter()

            for document in self.documents:
                tokens = self.tokenizer(str(document.get(field_name) or ""))
                frequency = Counter(tokens)
                term_frequencies.append(frequency)
                lengths.append(len(tokens))
                document_frequency.update(frequency.keys())

            self._field_term_frequencies[field_name] = term_frequencies
            self._field_lengths[field_name] = lengths
            self._field_avg_length[field_name] = (sum(lengths) / len(lengths)) if lengths else 0.0
            self._field_idf[field_name] = {
                term: math.log(1.0 + ((document_count - freq + 0.5) / (freq + 0.5)))
                for term, freq in document_frequency.items()
                if freq > 0
            }

    @staticmethod
    def _normalized_length(document_length: int, avg_length: float) -> float:
        if avg_length <= 0.0:
            return 1.0
        return float(document_length) / avg_length
