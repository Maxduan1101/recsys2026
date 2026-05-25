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


def _best_rank(candidate: CandidateScore, fragments: tuple[str, ...] | None = None) -> int | None:
    ranks = [
        rank
        for source, rank in candidate.source_ranks.items()
        if fragments is None or any(fragment in source for fragment in fragments)
    ]
    return min(ranks) if ranks else None


def _source_count(candidate: CandidateScore) -> int:
    return len(candidate.source_ranks)


def _passes_controlled_gate(
    state: ConversationState,
    candidate: CandidateScore,
    legacy_rank: int | None,
) -> bool:
    intent = infer_intent(state)
    best_rank = _best_rank(candidate)
    source_count = _source_count(candidate)
    title_rank = _best_rank(candidate, ("title_artist", "quoted_entities"))
    album_rank = _best_rank(candidate, ("album_artist",))
    seed_rank = _best_rank(candidate, ("seed_current",))
    enriched_rank = _best_rank(candidate, ("enriched",))
    strong_rule_match = candidate.boost >= 0.55

    if legacy_rank is not None and legacy_rank <= 60:
        return True
    if strong_rule_match:
        return True

    if intent == "specific_track":
        return bool(
            (legacy_rank is not None and legacy_rank <= 100)
            or (title_rank is not None and title_rank <= 30)
            or (source_count >= 5 and best_rank is not None and best_rank <= 12)
        )
    if intent == "album":
        return bool(
            (legacy_rank is not None and legacy_rank <= 100)
            or (album_rank is not None and album_rank <= 35)
            or (title_rank is not None and title_rank <= 45)
            or (source_count >= 5 and best_rank is not None and best_rank <= 15)
        )
    if intent in {"cover_art", "lyrics_theme", "mood_playlist"}:
        return bool(
            (source_count >= 3 and best_rank is not None and best_rank <= 25)
            or (enriched_rank is not None and enriched_rank <= 35)
            or (state.positive_seed_ids and seed_rank is not None and seed_rank <= 45)
        )
    if intent == "artist_exploration":
        return bool(
            (title_rank is not None and title_rank <= 50)
            or (seed_rank is not None and seed_rank <= 60)
            or (source_count >= 4 and best_rank is not None and best_rank <= 35)
        )
    return bool(source_count >= 4 and best_rank is not None and best_rank <= 30)


def _append_candidate(
    selected: list[str],
    candidate: CandidateScore,
    catalog: TrackCatalog,
    selected_artists: defaultdict[str, int],
    selected_albums: defaultdict[str, int],
    artist_cap: int,
    album_cap: int,
    enforce_caps: bool,
) -> bool:
    if candidate.track_id in selected:
        return False
    artist = catalog.normalized_field(candidate.track_id, "artist_name")
    album = catalog.normalized_field(candidate.track_id, "album_name")
    if enforce_caps and selected_artists[artist] >= artist_cap:
        return False
    if enforce_caps and selected_albums[album] >= album_cap:
        return False
    selected.append(candidate.track_id)
    selected_artists[artist] += 1
    selected_albums[album] += 1
    return True


def rerank_candidates_gated(
    state: ConversationState,
    catalog: TrackCatalog,
    candidates: dict[str, CandidateScore],
    legacy_order: list[str],
    top_k: int = 20,
    global_counts: Counter[str] | None = None,
    protect_head_k: int = 5,
) -> list[str]:
    for candidate in candidates.values():
        candidate.boost = score_candidate_boost(state, catalog, candidate.track_id, global_counts=global_counts)
        candidate.score += candidate.boost

    sorted_candidates = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
    by_id = {candidate.track_id: candidate for candidate in sorted_candidates}
    selected: list[str] = []
    selected_artists: defaultdict[str, int] = defaultdict(int)
    selected_albums: defaultdict[str, int] = defaultdict(int)
    intent = infer_intent(state)
    artist_cap = 20 if intent in {"specific_track", "artist_exploration"} else 4
    album_cap = 20 if intent in {"specific_track", "album"} else 3
    protect_head_k = max(0, min(protect_head_k, top_k))
    legacy_rank_by_track = {track_id: rank for rank, track_id in enumerate(legacy_order, start=1)}

    for track_id in legacy_order[:protect_head_k]:
        candidate = by_id.get(track_id) or CandidateScore(track_id=track_id, score=0.0)
        _append_candidate(
            selected,
            candidate,
            catalog,
            selected_artists,
            selected_albums,
            artist_cap,
            album_cap,
            enforce_caps=False,
        )

    gated = [
        candidate
        for candidate in sorted_candidates
        if candidate.track_id not in selected
        and _passes_controlled_gate(state, candidate, legacy_rank_by_track.get(candidate.track_id))
    ]
    for candidate in gated:
        if len(selected) >= top_k:
            break
        _append_candidate(
            selected,
            candidate,
            catalog,
            selected_artists,
            selected_albums,
            artist_cap,
            album_cap,
            enforce_caps=len(selected) >= max(5, protect_head_k),
        )

    for track_id in legacy_order:
        if len(selected) >= top_k:
            break
        candidate = by_id.get(track_id) or CandidateScore(track_id=track_id, score=0.0)
        _append_candidate(
            selected,
            candidate,
            catalog,
            selected_artists,
            selected_albums,
            artist_cap,
            album_cap,
            enforce_caps=len(selected) >= max(5, protect_head_k),
        )

    for candidate in sorted_candidates:
        if len(selected) >= top_k:
            break
        _append_candidate(
            selected,
            candidate,
            catalog,
            selected_artists,
            selected_albums,
            artist_cap,
            album_cap,
            enforce_caps=False,
        )
    return selected


def diversify_tail(
    ranked_track_ids: list[str],
    state: ConversationState,
    catalog: TrackCatalog,
    global_counts: Counter[str],
    top_k: int = 20,
    preserve_head_k: int = 10,
    repeat_penalty: float = 0.06,
) -> list[str]:
    seen = set()
    unique_ranked = []
    for track_id in ranked_track_ids:
        if track_id in seen or not catalog.has_track(track_id):
            continue
        unique_ranked.append(track_id)
        seen.add(track_id)

    preserve_head_k = max(0, min(preserve_head_k, top_k))
    selected = unique_ranked[:preserve_head_k]
    selected_artists: defaultdict[str, int] = defaultdict(int)
    selected_albums: defaultdict[str, int] = defaultdict(int)
    for track_id in selected:
        selected_artists[catalog.normalized_field(track_id, "artist_name")] += 1
        selected_albums[catalog.normalized_field(track_id, "album_name")] += 1

    intent = infer_intent(state)
    artist_cap = 20 if intent in {"specific_track", "artist_exploration"} else 4
    album_cap = 20 if intent in {"specific_track", "album"} else 3
    remaining = [track_id for track_id in unique_ranked[preserve_head_k:] if track_id not in selected]
    rank_by_track = {track_id: index for index, track_id in enumerate(unique_ranked)}

    while len(selected) < top_k and remaining:
        best_index = 0
        best_score = -1e18
        for index, track_id in enumerate(remaining[:500]):
            artist = catalog.normalized_field(track_id, "artist_name")
            album = catalog.normalized_field(track_id, "album_name")
            rank_index = rank_by_track.get(track_id, len(unique_ranked))
            score = 1.0 / (rank_index + 1)
            score -= repeat_penalty * math.sqrt(global_counts[track_id])
            score -= 0.08 * selected_artists[artist]
            score -= 0.06 * selected_albums[album]
            if global_counts[track_id] == 0:
                score += 0.18
            if selected_artists[artist] >= artist_cap:
                score -= 0.3
            if selected_albums[album] >= album_cap:
                score -= 0.25
            if score > best_score:
                best_score = score
                best_index = index
        picked = remaining.pop(best_index)
        selected.append(picked)
        selected_artists[catalog.normalized_field(picked, "artist_name")] += 1
        selected_albums[catalog.normalized_field(picked, "album_name")] += 1

    for track_id in unique_ranked:
        if len(selected) >= top_k:
            break
        if track_id not in selected:
            selected.append(track_id)
    return selected[:top_k]
