from __future__ import annotations

import json
import os
from dataclasses import dataclass

import bm25s
from tqdm import tqdm


@dataclass
class SearchResult:
    track_id: str
    score: float
    rank: int


class BM25Index:
    def __init__(self, name: str, cache_dir: str):
        self.name = name
        self.cache_dir = cache_dir
        self.index_dir = os.path.join(cache_dir, "bm25", name)
        self.model = None
        self.track_ids: list[str] = []

    @staticmethod
    def _doc_to_id(doc) -> int:
        if isinstance(doc, dict):
            return int(doc["id"])
        if hasattr(doc, "item"):
            doc = doc.item()
            if isinstance(doc, dict):
                return int(doc["id"])
        return int(doc)

    def build_or_load(self, track_ids: list[str], documents: list[str], rebuild: bool = False) -> None:
        if os.path.exists(os.path.join(self.index_dir, "track_ids.json")) and not rebuild:
            self.model = bm25s.BM25.load(self.index_dir, load_corpus=True)
            with open(os.path.join(self.index_dir, "track_ids.json"), "r", encoding="utf-8") as f:
                self.track_ids = json.load(f)
            return

        os.makedirs(self.index_dir, exist_ok=True)
        tokens = bm25s.tokenize(documents, show_progress=False)
        model = bm25s.BM25()
        model.index(tokens, show_progress=False)
        model.save(self.index_dir, corpus=documents)
        with open(os.path.join(self.index_dir, "track_ids.json"), "w", encoding="utf-8") as f:
            json.dump(track_ids, f)
        self.model = model
        self.track_ids = track_ids

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        if not query.strip():
            return []
        tokens = bm25s.tokenize([query.lower()], show_progress=False)
        results = self.model.retrieve(tokens, k=top_k, return_as="tuple", show_progress=False)
        docs = results.documents[0]
        scores = results.scores[0]
        out = []
        for rank, (doc, score) in enumerate(zip(docs, scores), start=1):
            out.append(SearchResult(track_id=self.track_ids[self._doc_to_id(doc)], score=float(score), rank=rank))
        return out

    def search_many(self, queries: list[str], top_k: int) -> list[list[SearchResult]]:
        tokens = bm25s.tokenize([query.lower() for query in queries], show_progress=False)
        results = self.model.retrieve(tokens, k=top_k, return_as="tuple", show_progress=False)
        all_results = []
        for row_docs, row_scores in zip(results.documents, results.scores):
            row = []
            for rank, (doc, score) in enumerate(zip(row_docs, row_scores), start=1):
                row.append(SearchResult(track_id=self.track_ids[self._doc_to_id(doc)], score=float(score), rank=rank))
            all_results.append(row)
        return all_results


class MultiBM25Retriever:
    def __init__(self, cache_dir: str, rebuild: bool = False):
        self.cache_dir = cache_dir
        self.rebuild = rebuild
        self.indices: dict[str, BM25Index] = {}

    def build(self, track_ids: list[str], documents_by_index: dict[str, list[str]]) -> None:
        for name, docs in tqdm(documents_by_index.items(), desc="Build/load BM25 indices"):
            index = BM25Index(name=name, cache_dir=self.cache_dir)
            index.build_or_load(track_ids=track_ids, documents=docs, rebuild=self.rebuild)
            self.indices[name] = index

    def batch_search(
        self,
        query_variants_per_state: list[dict[str, str]],
        top_k_by_index: dict[str, int],
        query_weights: dict[str, float],
        index_weights: dict[str, float],
    ) -> list[list[tuple[str, str, float, list[SearchResult]]]]:
        per_state_sources: list[list[tuple[str, str, float, list[SearchResult]]]] = [
            [] for _ in query_variants_per_state
        ]
        variant_names = sorted({name for variants in query_variants_per_state for name in variants})
        for index_name, index in self.indices.items():
            top_k = top_k_by_index.get(index_name, 300)
            for variant_name in variant_names:
                queries = [variants.get(variant_name, "") for variants in query_variants_per_state]
                if not any(query.strip() for query in queries):
                    continue
                rows = index.search_many(queries, top_k=top_k)
                weight = index_weights.get(index_name, 1.0) * query_weights.get(variant_name, 1.0)
                source_name = f"{index_name}:{variant_name}"
                for state_index, row in enumerate(rows):
                    if row:
                        per_state_sources[state_index].append((source_name, index_name, weight, row))
        return per_state_sources
