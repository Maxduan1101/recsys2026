from __future__ import annotations

import hashlib
import re

from .data import TrackCatalog
from .fusion import infer_intent
from .state import ConversationState


def _track_phrase(catalog: TrackCatalog, track_id: str) -> str:
    view = catalog.view(track_id)
    return f'"{view.track_name}" by {view.artist_name}'


def _pick(state: ConversationState, options: list[str], salt: str = "") -> str:
    key = f"{state.session_id}:{state.turn_number}:{salt}:{state.current_user_query}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def _profile_hint(state: ConversationState) -> str:
    culture = state.user_profile.get("preferred_musical_culture")
    country = state.user_profile.get("country_name")
    if culture:
        return f"your {culture} listening background"
    if country:
        return f"your {country} profile"
    return ""


def _tag_list(catalog: TrackCatalog, track_id: str, limit: int = 3) -> list[str]:
    blocked = {
        "fuck",
        "fucking",
        "shit",
        "bitch",
        "asshole",
        "cunt",
        "nigger",
        "nigga",
    }
    tags = []
    seen = set()
    for raw_tag in catalog.view(track_id).tag_list.split(", "):
        tag = raw_tag.strip()
        key = tag.lower()
        if not tag or key in seen:
            continue
        if blocked & set(re.findall(r"[a-z]+", key)):
            continue
        if len(tag) > 36:
            continue
        tags.append(tag)
        seen.add(key)
        if len(tags) >= limit:
            break
    return tags


def _join_words(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _year_hint(catalog: TrackCatalog, track_id: str) -> str:
    year = catalog.release_year(track_id)
    return str(year) if year else ""


def _short_request(state: ConversationState, max_words: int = 12) -> str:
    text = state.current_user_query or state.listener_goal
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", text)
    if not words:
        return "this request"
    clipped = words[:max_words]
    suffix = "" if len(words) <= max_words else "..."
    return " ".join(clipped) + suffix


def _seed_phrase(state: ConversationState, catalog: TrackCatalog) -> str:
    for track_id in reversed(state.positive_seed_ids):
        if catalog.has_track(track_id):
            return _track_phrase(catalog, track_id)
    return ""


def _negative_phrase(state: ConversationState, catalog: TrackCatalog) -> str:
    for track_id in reversed(state.negative_seed_ids):
        if catalog.has_track(track_id):
            return _track_phrase(catalog, track_id)
    return ""


def _compact_metadata(catalog: TrackCatalog, track_id: str) -> str:
    view = catalog.view(track_id)
    parts = []
    tags = _tag_list(catalog, track_id, limit=3)
    if tags:
        parts.append(f"tags {_join_words(tags)}")
    year = _year_hint(catalog, track_id)
    if year:
        parts.append(f"year {year}")
    if view.album_name:
        parts.append(f"album {view.album_name}")
    if not parts:
        parts.append("title and artist metadata")
    return "; ".join(parts[:3])


def _compact_backup(catalog: TrackCatalog, track_ids: list[str]) -> str:
    backups = [_track_phrase(catalog, track_id) for track_id in track_ids[1:3]]
    if len(backups) == 2:
        return f"Backups: {backups[0]}; {backups[1]}."
    if len(backups) == 1:
        return f"Backup: {backups[0]}."
    return "The remaining list keeps wider alternatives."


def generate_response(state: ConversationState, catalog: TrackCatalog, track_ids: list[str]) -> str:
    if not track_ids:
        return "I found a few tracks that should fit the direction you described."

    first_phrase = _track_phrase(catalog, track_ids[0])
    request_hint = _short_request(state)
    metadata = _compact_metadata(catalog, track_ids[0])
    backup = _compact_backup(catalog, track_ids)
    profile = _profile_hint(state)
    feedback = _seed_phrase(state, catalog) or _negative_phrase(state, catalog)
    intent = infer_intent(state)

    if intent == "specific_track":
        lead = _pick(
            state,
            [
                f"Exact-track lead for \"{request_hint}\": {first_phrase}.",
                f"First test for the song clue \"{request_hint}\": {first_phrase}.",
                f"Closest catalog answer to \"{request_hint}\": {first_phrase}.",
                f"Likely specific-song match: {first_phrase}. Cue: \"{request_hint}\".",
            ],
            salt="lead-specific",
        )
    elif intent == "album":
        lead = _pick(
            state,
            [
                f"Album-clue lead for \"{request_hint}\": {first_phrase}.",
                f"First album-aware anchor: {first_phrase}. Cue: \"{request_hint}\".",
                f"Record-context pick: {first_phrase}. Request cue: \"{request_hint}\".",
            ],
            salt="lead-album",
        )
    elif intent == "artist_exploration":
        lead = _pick(
            state,
            [
                f"Artist-path opener: {first_phrase}. Cue: \"{request_hint}\".",
                f"Artist-led recommendation for \"{request_hint}\": {first_phrase}.",
                f"Discovery starts with {first_phrase} for the artist cue.",
            ],
            salt="lead-artist",
        )
    elif intent == "cover_art":
        lead = _pick(
            state,
            [
                f"Visual-clue anchor: {first_phrase}. Cue: \"{request_hint}\".",
                f"Cover-art search starts with {first_phrase}.",
                f"Image-related hint lead: {first_phrase}; cue \"{request_hint}\".",
            ],
            salt="lead-cover",
        )
    elif intent == "lyrics_theme":
        lead = _pick(
            state,
            [
                f"Theme/lyric lead: {first_phrase}. Cue: \"{request_hint}\".",
                f"Story-cue opener for \"{request_hint}\": {first_phrase}.",
                f"Thematic anchor: {first_phrase}; request \"{request_hint}\".",
            ],
            salt="lead-lyrics",
        )
    else:
        lead = _pick(
            state,
            [
                f"Mood/discovery lead for \"{request_hint}\": {first_phrase}.",
                f"Start with {first_phrase} for \"{request_hint}\".",
                f"Opening pick: {first_phrase}. Goal cue: \"{request_hint}\".",
                f"First recommendation: {first_phrase}; direction \"{request_hint}\".",
            ],
            salt="lead-mood",
        )

    extras = []
    if feedback:
        extras.append(f"Session clue: {feedback}.")
    if profile:
        extras.append(f"Profile cue: {profile}.")
    extra = " ".join(extras[:2])
    return " ".join(part for part in [lead, f"Grounding: {metadata}.", extra, backup] if part)
