from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .bm25_retrieval import SearchResult
from .data import TrackCatalog, normalize_text, tokenize_words
from .state import ConversationState


@dataclass
class CandidateScore:
    track_id: str
    score: float
    source_ranks: dict[str, int] = field(default_factory=dict)
    boost: float = 0.0


def rrf_fuse(
    sources: list[tuple[str, str, float, list[SearchResult]]],
    rrf_k: int = 60,
) -> dict[str, CandidateScore]:
    candidates: dict[str, CandidateScore] = {}
    for source_name, _index_name, weight, results in sources:
        for result in results:
            item = candidates.setdefault(result.track_id, CandidateScore(track_id=result.track_id, score=0.0))
            item.score += weight / (rrf_k + result.rank)
            item.source_ranks[source_name] = result.rank
    return candidates


def _phrase_present(needle: str, haystack: str, min_len: int = 3) -> bool:
    needle = normalize_text(needle)
    if len(needle) < min_len:
        return False
    return needle in haystack


def _query_years(text: str) -> set[int]:
    years = {int(match) for match in re.findall(r"\b(19\d{2}|20\d{2})\b", text)}
    for decade in re.findall(r"\b([789]0)s\b", text.lower()):
        start = 1900 + int(decade)
        years.update(range(start, start + 10))
    for decade in re.findall(r"\b(2000|2010|2020)s\b", text.lower()):
        start = int(decade)
        years.update(range(start, start + 10))
    return years


def infer_intent(state: ConversationState) -> str:
    text = normalize_text(f"{state.current_user_query} {state.listener_goal}")
    if re.search(r"['\"].+['\"]", state.current_user_query) or "specific song" in text or "specific track" in text:
        return "specific_track"
    if "album" in text:
        return "album"
    if "cover" in text or "artwork" in text or "image" in text:
        return "cover_art"
    if "lyric" in text or "story" in text:
        return "lyrics_theme"
    if "artist" in text or " by " in text:
        return "artist_exploration"
    return "mood_playlist"


def score_candidate_boost(
    state: ConversationState,
    catalog: TrackCatalog,
    track_id: str,
    global_counts: Counter[str] | None = None,
) -> float:
    text = normalize_text(
        f"{state.current_user_query} {state.listener_goal} {state.specificity} "
        f"{state.user_profile.get('preferred_musical_culture', '')}"
    )
    query_words = tokenize_words(text)
    title = catalog.normalized_field(track_id, "track_name")
    artist = catalog.normalized_field(track_id, "artist_name")
    album = catalog.normalized_field(track_id, "album_name")
    boost = 0.0

    if _phrase_present(title, text):
        boost += 2.2
    if _phrase_present(artist, text):
        boost += 0.85
    if _phrase_present(album, text):
        boost += 0.55

    tag_overlap = len(catalog.tag_words(track_id) & query_words)
    boost += min(0.35, tag_overlap * 0.035)

    year = catalog.release_year(track_id)
    if year and year in _query_years(text):
        boost += 0.35

    for seed_id in state.positive_seed_ids[-4:]:
        if not catalog.has_track(seed_id):
            continue
        if catalog.normalized_field(seed_id, "artist_name") == artist:
            boost += 0.25
        if catalog.normalized_field(seed_id, "album_name") == album:
            boost += 0.2
        seed_tags = catalog.tag_words(seed_id)
        boost += min(0.18, 0.015 * len(seed_tags & catalog.tag_words(track_id)))

    for seed_id in state.negative_seed_ids[-4:]:
        if not catalog.has_track(seed_id):
            continue
        if catalog.normalized_field(seed_id, "artist_name") == artist:
            boost -= 0.28
        if catalog.normalized_field(seed_id, "album_name") == album:
            boost -= 0.18
        seed_tags = catalog.tag_words(seed_id)
        boost -= min(0.16, 0.012 * len(seed_tags & catalog.tag_words(track_id)))

    popularity = catalog.view(track_id).popularity
    boost += min(0.08, math.log1p(max(popularity, 0.0)) * 0.012)

    if global_counts:
        boost -= min(0.08, global_counts[track_id] * 0.002)
    return boost


def rerank_candidates(
    state: ConversationState,
    catalog: TrackCatalog,
    candidates: dict[str, CandidateScore],
    top_k: int = 20,
    global_counts: Counter[str] | None = None,
) -> list[str]:
    for candidate in candidates.values():
        candidate.boost = score_candidate_boost(state, catalog, candidate.track_id, global_counts=global_counts)
        candidate.score += candidate.boost

    sorted_candidates = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
    selected: list[str] = []
    selected_artists: defaultdict[str, int] = defaultdict(int)
    selected_albums: defaultdict[str, int] = defaultdict(int)
    intent = infer_intent(state)
    artist_cap = 20 if intent in {"specific_track", "artist_exploration"} else 4
    album_cap = 20 if intent in {"specific_track", "album"} else 3

    # Protect the nDCG-sensitive head.
    for candidate in sorted_candidates:
        if len(selected) >= min(5, top_k):
            break
        if candidate.track_id not in selected:
            selected.append(candidate.track_id)
            selected_artists[catalog.normalized_field(candidate.track_id, "artist_name")] += 1
            selected_albums[catalog.normalized_field(candidate.track_id, "album_name")] += 1

    remaining = [candidate for candidate in sorted_candidates if candidate.track_id not in selected]
    while len(selected) < top_k and remaining:
        best_index = 0
        best_score = -1e18
        for index, candidate in enumerate(remaining[:300]):
            artist = catalog.normalized_field(candidate.track_id, "artist_name")
            album = catalog.normalized_field(candidate.track_id, "album_name")
            score = candidate.score
            if selected_artists[artist] >= artist_cap:
                score -= 0.45
            if selected_albums[album] >= album_cap:
                score -= 0.35
            if score > best_score:
                best_score = score
                best_index = index
        picked = remaining.pop(best_index)
        selected.append(picked.track_id)
        selected_artists[catalog.normalized_field(picked.track_id, "artist_name")] += 1
        selected_albums[catalog.normalized_field(picked.track_id, "album_name")] += 1
    return selected
