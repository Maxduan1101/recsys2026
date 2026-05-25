from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from .data import TrackCatalog, as_text
from .state import role_at_turn


@dataclass
class DocumentCollection:
    track_ids: list[str]
    documents_by_index: dict[str, list[str]]


def _append_limited(bucket: list[str], text: str, max_snippets: int) -> None:
    text = " ".join(as_text(text).split())
    if text and len(bucket) < max_snippets:
        bucket.append(text[:900])


def build_train_augmentation(
    train_dataset,
    catalog: TrackCatalog,
    cache_path: str,
    max_snippets_per_track: int = 12,
    rebuild: bool = False,
) -> dict[str, list[str]]:
    if os.path.exists(cache_path) and not rebuild:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    snippets: dict[str, list[str]] = defaultdict(list)
    for item in tqdm(train_dataset, desc="Build train-context item docs"):
        goal = item.get("conversation_goal", {})
        goal_text = as_text(goal.get("listener_goal"))
        specificity = as_text(goal.get("specificity"))
        category = as_text(goal.get("category"))
        conversations = item.get("conversations", [])
        for turn_number in range(1, 9):
            user_turn = role_at_turn(conversations, turn_number, "user")
            music_turn = role_at_turn(conversations, turn_number, "music")
            assistant_turn = role_at_turn(conversations, turn_number, "assistant")
            if not user_turn or not music_turn:
                continue
            track_id = music_turn.get("content")
            if not catalog.has_track(track_id):
                continue
            text = "\n".join(
                part
                for part in [
                    f"training_user_query: {as_text(user_turn.get('content'))}",
                    f"training_goal: {goal_text}",
                    f"training_goal_category: {category} specificity: {specificity}",
                    f"music_selection_reason: {as_text(music_turn.get('thought'))}",
                    f"assistant_explanation: {as_text(assistant_turn.get('content') if assistant_turn else '')}",
                ]
                if part.strip()
            )
            _append_limited(snippets[track_id], text, max_snippets_per_track)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(snippets, f, ensure_ascii=False)
    return snippets


def build_documents(
    catalog: TrackCatalog,
    train_augmentation: dict[str, list[str]] | None = None,
) -> DocumentCollection:
    train_augmentation = train_augmentation or {}
    track_ids = catalog.track_ids
    documents_by_index = {
        "legacy_metadata": [],
        "metadata_all": [],
        "title_artist": [],
        "album_artist": [],
        "tags": [],
        "enriched": [],
    }
    for track_id in track_ids:
        base = catalog.metadata_text(
            track_id,
            fields=["track_name", "artist_name", "album_name", "tag_list", "release_date", "popularity"],
        )
        legacy = catalog.metadata_text(track_id, fields=["track_name", "artist_name", "album_name", "release_date"])
        title_artist = catalog.metadata_text(track_id, fields=["track_name", "artist_name"])
        album_artist = catalog.metadata_text(track_id, fields=["album_name", "artist_name", "release_date"])
        tags = catalog.metadata_text(track_id, fields=["tag_list"])
        aug = "\n".join(train_augmentation.get(track_id, []))
        documents_by_index["legacy_metadata"].append(legacy)
        documents_by_index["metadata_all"].append(base)
        documents_by_index["title_artist"].append(title_artist)
        documents_by_index["album_artist"].append(album_artist)
        documents_by_index["tags"].append(tags)
        documents_by_index["enriched"].append(f"{base}\n\n{aug}" if aug else base)
    return DocumentCollection(track_ids=track_ids, documents_by_index=documents_by_index)
