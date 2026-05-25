from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from datasets import load_dataset

from .data import TRACK_EMBEDDINGS, USER_EMBEDDINGS


TRACK_EMBEDDING_CHANNELS = {
    "audio": "audio-laion_clap",
    "image": "image-siglip2",
    "track_cf": "cf-bpr",
    "attributes": "attributes-qwen3_embedding_0.6b",
    "lyrics": "lyrics-qwen3_embedding_0.6b",
    "metadata": "metadata-qwen3_embedding_0.6b",
}


def vectors_to_matrix(vectors, dtype=np.float32) -> np.ndarray:
    rows = list(vectors)
    dim = 0
    for vector in rows:
        if vector:
            dim = len(vector)
            break
    if dim == 0:
        return np.zeros((len(rows), 0), dtype=dtype)
    matrix = np.zeros((len(rows), dim), dtype=dtype)
    for index, vector in enumerate(rows):
        if len(vector) == dim:
            matrix[index] = np.asarray(vector, dtype=dtype)
    return matrix


def l2_normalize_with_mask(matrix: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    valid = np.isfinite(norms) & (norms > eps)
    out = np.zeros_like(matrix, dtype=np.float32)
    out[valid] = matrix[valid] / norms[valid, None]
    return out, valid


@dataclass
class ChannelMatrix:
    raw: np.ndarray
    normalized: np.ndarray
    valid: np.ndarray


class TrackEmbeddingStore:
    def __init__(
        self,
        dataset_name: str = TRACK_EMBEDDINGS,
        split: str = "all_tracks",
        channels: dict[str, str] | None = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.channels = channels or TRACK_EMBEDDING_CHANNELS
        dataset = load_dataset(dataset_name, split=split)
        self.track_ids = list(dataset["track_id"])
        self.track_index = {track_id: index for index, track_id in enumerate(self.track_ids)}
        self.matrices: dict[str, ChannelMatrix] = {}
        for name, column in self.channels.items():
            raw = vectors_to_matrix(dataset[column], dtype=np.float32)
            normalized, valid = l2_normalize_with_mask(raw)
            self.matrices[name] = ChannelMatrix(raw=raw, normalized=normalized, valid=valid)

    def has_track(self, track_id: str) -> bool:
        return track_id in self.track_index

    def indices_for(self, track_ids: list[str]) -> list[int]:
        return [self.track_index[track_id] for track_id in track_ids if track_id in self.track_index]

    def cosine_scores(self, channel: str, query_vector: np.ndarray) -> np.ndarray:
        matrix = self.matrices[channel]
        query = np.asarray(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if not np.isfinite(query_norm) or query_norm <= 1e-12:
            return np.full(len(self.track_ids), -np.inf, dtype=np.float32)
        scores = matrix.normalized @ (query / query_norm)
        scores[~matrix.valid] = -np.inf
        return scores

    def seed_scores(self, channel: str, seed_track_ids: list[str]) -> np.ndarray:
        indices = self.indices_for(seed_track_ids)
        if not indices:
            return np.full(len(self.track_ids), -np.inf, dtype=np.float32)
        matrix = self.matrices[channel]
        seed_vectors = matrix.normalized[indices]
        valid_seed = matrix.valid[indices]
        if not valid_seed.any():
            return np.full(len(self.track_ids), -np.inf, dtype=np.float32)
        scores = matrix.normalized @ seed_vectors[valid_seed].T
        out = scores.max(axis=1)
        out[~matrix.valid] = -np.inf
        return out.astype(np.float32)

    def topk_from_scores(self, scores: np.ndarray, top_k: int) -> list[tuple[str, float, int]]:
        if top_k <= 0:
            return []
        finite = np.isfinite(scores)
        if not finite.any():
            return []
        candidate_indices = np.flatnonzero(finite)
        values = scores[candidate_indices]
        k = min(top_k, len(values))
        top = np.argpartition(-values, k - 1)[:k]
        ordered = top[np.argsort(-values[top])]
        return [
            (self.track_ids[int(candidate_indices[index])], float(values[index]), rank)
            for rank, index in enumerate(ordered, start=1)
        ]


class UserEmbeddingStore:
    def __init__(
        self,
        dataset_name: str = USER_EMBEDDINGS,
        splits: tuple[str, ...] = ("train", "test_warm", "test_cold"),
    ):
        self.dataset_name = dataset_name
        self.user_vectors: dict[str, np.ndarray] = {}
        self.dim = 0
        for split in splits:
            dataset = load_dataset(dataset_name, split=split)
            for row in dataset:
                vector = row["cf-bpr"]
                if not vector:
                    continue
                if self.dim == 0:
                    self.dim = len(vector)
                if len(vector) == self.dim:
                    self.user_vectors[row["user_id"]] = np.asarray(vector, dtype=np.float32)

    def has_user(self, user_id: str) -> bool:
        return user_id in self.user_vectors

    def user_track_scores(self, user_id: str, track_store: TrackEmbeddingStore) -> np.ndarray:
        if user_id not in self.user_vectors:
            return np.full(len(track_store.track_ids), -np.inf, dtype=np.float32)
        user = self.user_vectors[user_id]
        matrix = track_store.matrices["track_cf"]
        if matrix.raw.shape[1] != user.shape[0]:
            return np.full(len(track_store.track_ids), -np.inf, dtype=np.float32)
        scores = matrix.raw @ user
        scores[~matrix.valid] = -np.inf
        return scores.astype(np.float32)
