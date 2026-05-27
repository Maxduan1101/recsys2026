from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from goalflow.data import TrackCatalog, as_text
from goalflow.embeddings import TrackEmbeddingStore


DEFAULT_OUT_DIR = Path("goalflow_musiccrs/experiments/qwen3_direct_dense_probe")
QWEN_CHANNELS = {
    "metadata": "metadata-qwen3_embedding_0.6b",
    "attributes": "attributes-qwen3_embedding_0.6b",
    "lyrics": "lyrics-qwen3_embedding_0.6b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe whether Qwen3 query embeddings are compatible with official track Qwen3 channels.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model-id", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--download", action="store_true", help="Allow downloading the query encoder if it is not cached locally.")
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_encoder(model_id: str, download: bool, device: str):
    local_files_only = not download
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, local_files_only=local_files_only, trust_remote_code=True)
    model.eval()
    model.to(device)
    return tokenizer, model


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


@torch.inference_mode()
def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    *,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = model(**encoded)
        pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(vectors)


def deterministic_sample(catalog: TrackCatalog, sample_size: int) -> list[str]:
    candidates = [
        track_id
        for track_id in catalog.track_ids
        if catalog.view(track_id).track_name and catalog.view(track_id).artist_name
    ]
    candidates.sort(key=lambda track_id: (-catalog.view(track_id).popularity, track_id))
    stride = max(1, len(candidates) // sample_size)
    return candidates[::stride][:sample_size]


def topk_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    k = min(top_k, scores.shape[0])
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.argsort(-scores[top])]


def exact_sanity(
    catalog: TrackCatalog,
    store: TrackEmbeddingStore,
    query_vectors: np.ndarray,
    sampled_track_ids: list[str],
    channel: str,
) -> dict[str, Any]:
    matrix = store.matrices[channel]
    track_index = store.track_index
    track_ids = store.track_ids
    exact_top20 = 0
    exact_top100 = 0
    artist_top20 = 0
    top1_counts: dict[str, int] = {}
    margins = []
    for qvec, gold_id in zip(query_vectors, sampled_track_ids):
        scores = matrix.normalized @ qvec
        scores[~matrix.valid] = -np.inf
        order = topk_indices(scores, 100)
        top_ids = [track_ids[int(index)] for index in order]
        top20 = top_ids[:20]
        top1_counts[top_ids[0]] = top1_counts.get(top_ids[0], 0) + 1
        if gold_id in top20:
            exact_top20 += 1
        if gold_id in top_ids:
            exact_top100 += 1
        gold_artist = catalog.normalized_field(gold_id, "artist_name")
        if any(catalog.normalized_field(track_id, "artist_name") == gold_artist for track_id in top20):
            artist_top20 += 1
        finite = scores[np.isfinite(scores)]
        if len(finite) >= 100:
            top_scores = np.sort(finite)[-100:]
            margins.append(float(top_scores[-1] - top_scores[0]))
    n = len(sampled_track_ids)
    max_top1_freq = max(top1_counts.values()) / n if top1_counts else 0.0
    return {
        "channel": channel,
        "sample_size": n,
        "query_dim": int(query_vectors.shape[1]),
        "track_channel_dim": int(matrix.normalized.shape[1]),
        "exact_track_top20_rate": exact_top20 / n,
        "exact_track_top100_rate": exact_top100 / n,
        "artist_top20_rate": artist_top20 / n,
        "top1_frequency_max_rate": max_top1_freq,
        "top1_unique_count": len(top1_counts),
        "top1_minus_top100_margin_p50": float(np.percentile(margins, 50)) if margins else 0.0,
        "top1_minus_top100_margin_p95": float(np.percentile(margins, 95)) if margins else 0.0,
        "passes_dimension_gate": bool(query_vectors.shape[1] == matrix.normalized.shape[1]),
        "passes_title_artist_gate": bool(exact_top20 / n >= 0.60 and artist_top20 / n >= 0.80),
        "passes_hubness_gate": bool(max_top1_freq < 0.02),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = vars(args)
    write_json(out_dir / "params.json", params)

    try:
        tokenizer, model = load_encoder(args.model_id, args.download, args.device)
    except Exception as exc:
        payload = {
            "status": "blocked",
            "reason": "query_encoder_not_available",
            "model_id": args.model_id,
            "download_allowed": bool(args.download),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "next_step": "rerun with --download only if we accept the model download cost; otherwise keep direct dense retrieval disabled.",
        }
        write_json(out_dir / "probe_result.json", payload)
        print(json.dumps(payload, indent=2))
        return

    catalog = TrackCatalog()
    sampled_track_ids = deterministic_sample(catalog, args.sample_size)
    queries = []
    for track_id in sampled_track_ids:
        view = catalog.view(track_id)
        queries.append(f"Represent this music request for retrieval: play {view.track_name} by {view.artist_name}.")

    query_vectors = encode_texts(
        queries,
        tokenizer,
        model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    store = TrackEmbeddingStore(channels=QWEN_CHANNELS)
    results = {
        "status": "ok",
        "model_id": args.model_id,
        "pooling": "attention-mask mean pooling over last_hidden_state, then L2 normalize",
        "query_norm_p50": float(np.percentile(np.linalg.norm(query_vectors, axis=1), 50)),
        "channels": [exact_sanity(catalog, store, query_vectors, sampled_track_ids, channel) for channel in QWEN_CHANNELS],
    }
    results["passes_any_safe_metadata_gate"] = any(
        item["channel"] == "metadata" and item["passes_dimension_gate"] and item["passes_title_artist_gate"] and item["passes_hubness_gate"]
        for item in results["channels"]
    )
    write_json(out_dir / "probe_result.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
