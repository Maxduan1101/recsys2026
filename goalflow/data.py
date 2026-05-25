from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from datasets import load_dataset


TRACK_METADATA = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
USER_METADATA = "talkpl-ai/TalkPlayData-Challenge-User-Metadata"
TRACK_EMBEDDINGS = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
USER_EMBEDDINGS = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings"
CONVERSATION_DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"
BLIND_A_DATASET = "talkpl-ai/TalkPlayData-Challenge-Blind-A"


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(as_text(item) for item in value if item is not None)
    return str(value)


def normalize_text(value: Any) -> str:
    text = as_text(value).lower()
    return re.sub(r"\s+", " ", text).strip()


def tokenize_words(text: str) -> set[str]:
    text = as_text(text)
    return set(re.findall(r"[a-z0-9][a-z0-9'\-]+", text.lower()))


@dataclass(frozen=True)
class TrackView:
    track_id: str
    track_name: str
    artist_name: str
    album_name: str
    tag_list: str
    popularity: float
    release_date: str
    duration: int | None


class TrackCatalog:
    def __init__(self, dataset_name: str = TRACK_METADATA, split: str = "all_tracks"):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = load_dataset(dataset_name, split=split)
        self.rows: dict[str, dict[str, Any]] = {item["track_id"]: item for item in self.dataset}
        self.track_ids = list(self.rows.keys())
        self._view_cache: dict[str, TrackView] = {}
        self._normalized_cache: dict[tuple[str, str], str] = {}
        self._tag_words_cache: dict[str, set[str]] = {}
        self._release_year_cache: dict[str, int | None] = {}

    def __len__(self) -> int:
        return len(self.track_ids)

    def has_track(self, track_id: str) -> bool:
        return track_id in self.rows

    def view(self, track_id: str) -> TrackView:
        if track_id in self._view_cache:
            return self._view_cache[track_id]
        row = self.rows[track_id]
        duration = row.get("duration")
        view = TrackView(
            track_id=track_id,
            track_name=as_text(row.get("track_name")),
            artist_name=as_text(row.get("artist_name")),
            album_name=as_text(row.get("album_name")),
            tag_list=as_text(row.get("tag_list")),
            popularity=float(row.get("popularity") or 0.0),
            release_date=as_text(row.get("release_date")),
            duration=int(duration) if duration is not None else None,
        )
        self._view_cache[track_id] = view
        return view

    def metadata_text(self, track_id: str, fields: Iterable[str] | None = None) -> str:
        row = self.rows[track_id]
        if fields is None:
            fields = ["track_name", "artist_name", "album_name", "tag_list", "release_date", "popularity"]
        parts = []
        for field in fields:
            value = as_text(row.get(field))
            if value:
                parts.append(f"{field}: {value}")
        return "\n".join(parts)

    def compact_summary(self, track_id: str) -> str:
        view = self.view(track_id)
        pieces = [
            f'track "{view.track_name}"',
            f"artist {view.artist_name}",
            f"album {view.album_name}",
        ]
        if view.release_date:
            pieces.append(f"release_date {view.release_date}")
        tags = ", ".join(view.tag_list.split(", ")[:12])
        if tags:
            pieces.append(f"tags {tags}")
        return "; ".join(piece for piece in pieces if piece)

    def normalized_field(self, track_id: str, field: str) -> str:
        key = (track_id, field)
        if key not in self._normalized_cache:
            self._normalized_cache[key] = normalize_text(self.rows[track_id].get(field))
        return self._normalized_cache[key]

    def tag_words(self, track_id: str) -> set[str]:
        if track_id not in self._tag_words_cache:
            self._tag_words_cache[track_id] = tokenize_words(self.rows[track_id].get("tag_list", ""))
        return self._tag_words_cache[track_id]

    def release_year(self, track_id: str) -> int | None:
        if track_id not in self._release_year_cache:
            match = re.search(r"\b(\d{4})\b", as_text(self.rows[track_id].get("release_date")))
            self._release_year_cache[track_id] = int(match.group(1)) if match else None
        return self._release_year_cache[track_id]


def load_conversations(dataset_name: str = CONVERSATION_DATASET, split: str = "train"):
    return load_dataset(dataset_name, split=split)
