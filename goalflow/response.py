from __future__ import annotations

import hashlib
import re
import unicodedata

from .data import TrackCatalog
from .fusion import infer_intent
from .state import ConversationState


GOOD_TAG_WORDS = {
    "acoustic",
    "alternative",
    "ambient",
    "americana",
    "ballad",
    "blues",
    "breakbeat",
    "classic",
    "classical",
    "club",
    "country",
    "dance",
    "disco",
    "dream",
    "dreamy",
    "drum",
    "dub",
    "electro",
    "electronic",
    "emo",
    "energetic",
    "experimental",
    "folk",
    "funk",
    "garage",
    "genre",
    "gospel",
    "guitar",
    "hardcore",
    "hip",
    "hop",
    "house",
    "indie",
    "instrumental",
    "jazz",
    "latin",
    "lo-fi",
    "melancholy",
    "metal",
    "mood",
    "new",
    "noise",
    "pop",
    "post",
    "prog",
    "punk",
    "r&b",
    "rap",
    "reggae",
    "relaxing",
    "rock",
    "singer",
    "ska",
    "soul",
    "soundtrack",
    "synth",
    "techno",
    "trance",
    "trip",
    "vocal",
    "wave",
}

BAD_TAG_WORDS = {
    "asshole",
    "bitch",
    "cunt",
    "fuck",
    "fucking",
    "negro",
    "negroes",
    "nigga",
    "nigger",
    "shit",
}

BAD_TAG_PHRASES = (
    "albums i own",
    "cds i own",
    "check out",
    "dirty electric guitar",
    "favourites",
    "favorites",
    "funk lift off",
    "funk tag",
    "ifs and buts",
    "i own",
    "lap dance",
    "lastfm",
    "my ",
    "playlist",
    "punk rocks",
    "rotation",
    "rock stoner",
    "seen live",
    "si related",
    "songs i",
    "songs ya",
    "sruuu",
    "to live by",
    "zielonypaw",
)

BAD_TAG_EXACT = {
    "genre",
    "mood",
    "new",
}


def _title_variant_key(value: str) -> str:
    value = re.sub(r"\([^)]*\)|\[[^]]*\]", " ", value.lower())
    value = re.sub(
        r"\b(remaster(?:ed)?|explicit|album|version|edit|radio|mix|original|feat(?:uring)?|live)\b",
        " ",
        value,
    )
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _ascii_ratio(value: str) -> float:
    if not value:
        return 0.0
    return sum(1 for char in value if ord(char) < 128) / len(value)


def _variant_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def _display_track_name(catalog: TrackCatalog, track_id: str) -> str:
    row = getattr(catalog, "rows", {}).get(track_id, {})
    raw_name = row.get("track_name") if isinstance(row, dict) else None
    if isinstance(raw_name, (list, tuple)):
        variants = []
        seen = set()
        for value in raw_name:
            variant = str(value).strip()
            key = variant.casefold()
            if variant and key not in seen:
                variants.append(variant)
                seen.add(key)
        if variants:
            keys = [_title_variant_key(variant) for variant in variants]
            nonempty_keys = [key for key in keys if key]
            duplicate_family = bool(nonempty_keys) and all(
                any(key == other or key in other or other in key for other in nonempty_keys)
                for key in nonempty_keys
            )
            if duplicate_family:
                return min(
                    variants,
                    key=lambda value: (
                        any(
                            marker in value.lower()
                            for marker in ("remaster", "explicit", "radio edit", "album version")
                        ),
                        -_ascii_ratio(value),
                        len(value),
                    ),
                )
            return variants[0]
    return catalog.view(track_id).track_name


def _display_artist_name(catalog: TrackCatalog, track_id: str) -> str:
    row = getattr(catalog, "rows", {}).get(track_id, {})
    raw_name = row.get("artist_name") if isinstance(row, dict) else None
    if isinstance(raw_name, (list, tuple)):
        variants = []
        seen = set()
        for value in raw_name:
            variant = str(value).strip()
            key = _variant_key(variant)
            if variant and key and key not in seen:
                variants.append((variant, key))
                seen.add(key)
        filtered = []
        for variant, key in variants:
            if any(key != other and key in other.split() or key != other and key in other for _, other in variants):
                continue
            filtered.append(variant)
        if filtered:
            return ", ".join(filtered)
    return catalog.view(track_id).artist_name


def _track_phrase(catalog: TrackCatalog, track_id: str) -> str:
    return f'"{_display_track_name(catalog, track_id)}" by {_display_artist_name(catalog, track_id)}'


def _pick(state: ConversationState, options: list[str], salt: str = "") -> str:
    key = f"{state.session_id}:{state.turn_number}:{salt}:{state.current_user_query}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def _profile_hint(state: ConversationState) -> str:
    culture = state.user_profile.get("preferred_musical_culture")
    country = state.user_profile.get("country_name")
    if culture:
        return f"your {_display_profile_value(culture)} listening background"
    if country:
        return f"your {_display_profile_value(country)} profile"
    return ""


def _display_profile_value(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    if text.lower() == text or text.upper() == text:
        return text.title()
    return text


def _capitalize_first(value: str) -> str:
    if not value:
        return value
    return value[0].upper() + value[1:]


def _tag_list(catalog: TrackCatalog, track_id: str, limit: int = 3, require_good_words: bool = True) -> list[str]:
    tags = []
    seen = set()
    for raw_tag in catalog.view(track_id).tag_list.split(", "):
        tag = raw_tag.strip()
        tag = re.sub(r"\s*/\s*", "-", tag)
        tag = re.sub(r"\s+", " ", tag).strip(" -_/")
        key = tag.lower()
        if not tag or key in seen or key in BAD_TAG_EXACT:
            continue
        if any(phrase in key for phrase in BAD_TAG_PHRASES):
            continue
        words = set(re.findall(r"[a-z&-]+", key))
        if BAD_TAG_WORDS & words or any(word in key for word in BAD_TAG_WORDS):
            continue
        if len(tag) > 36:
            continue
        if require_good_words and not (GOOD_TAG_WORDS & words):
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


def _request_focus(state: ConversationState, max_words: int = 18) -> str:
    current_words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", state.current_user_query)
    current_norm = " ".join(word.lower() for word in current_words)
    feedback_only = bool(
        current_words
        and len(current_words) <= 6
        and re.fullmatch(
            r"(yes|yeah|yep|no|nope|ok|okay|closer|close|more|less|another|again|continue|try|maybe|not|wrong|right|different|similar|that|this|one|please|thanks|thank you|cool|great|good|bad|fine|nah|hmm|hmmm|still)(\s+(yes|yeah|yep|no|nope|ok|okay|closer|close|more|less|another|again|continue|try|maybe|not|wrong|right|different|similar|that|this|one|please|thanks|thank|you|cool|great|good|bad|fine|nah|hmm|hmmm|still))*",
            current_norm,
        )
    )
    text = state.listener_goal if feedback_only and state.listener_goal else state.current_user_query or state.listener_goal
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", text)
    if not words:
        return "the direction you described"
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


def _compact_metadata(catalog: TrackCatalog, track_id: str, require_good_words: bool = True) -> str:
    view = catalog.view(track_id)
    parts = []
    tags = _tag_list(catalog, track_id, limit=3, require_good_words=require_good_words)
    if tags:
        parts.append(f"its catalog tags lean toward {_join_words(tags)}")
    year = _year_hint(catalog, track_id)
    if year:
        parts.append(f"the release year is {year}")
    if view.album_name:
        parts.append(f"the album context is {view.album_name}")
    if not parts:
        parts.append("the title and artist metadata are the cleanest anchors")
    return "; ".join(parts[:3])


def _metric_metadata(catalog: TrackCatalog, track_id: str, require_good_words: bool = True) -> str:
    view = catalog.view(track_id)
    parts = []
    tags = _tag_list(catalog, track_id, limit=3, require_good_words=require_good_words)
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


def _backup_detail(catalog: TrackCatalog, track_id: str) -> str:
    view = catalog.view(track_id)
    pieces = [_track_phrase(catalog, track_id)]
    year = _year_hint(catalog, track_id)
    if year:
        pieces.append(year)
    if view.album_name:
        pieces.append(view.album_name)
    return " / ".join(pieces[:3])


def _compact_backup(state: ConversationState, catalog: TrackCatalog, track_ids: list[str]) -> str:
    backups = [_backup_detail(catalog, track_id) for track_id in track_ids[1:3]]
    if len(backups) == 2:
        options = [
            f"I kept {backups[0]} and {backups[1]} close by as alternate paths.",
            f"The nearest backups are {backups[0]} and {backups[1]}.",
            f"If the first pick is slightly off, the next checks are {backups[0]} and {backups[1]}.",
        ]
        return _pick(state, options, salt="backup")
    if len(backups) == 1:
        return f"I kept {backups[0]} close by as an alternate path."
    return "The rest of the list gives the search a little room to recover."


def _natural_cue(state: ConversationState, catalog: TrackCatalog) -> str:
    seed = _seed_phrase(state, catalog)
    if seed:
        return f"The earlier positive signal from {seed} nudged the list in this direction."
    negative = _negative_phrase(state, catalog)
    if negative:
        return f"I treated {negative} as a weaker path and avoided leaning too hard on it."
    profile = _profile_hint(state)
    if profile:
        return f"I also used {profile} as a soft preference."
    return ""


def _labeled_backup(catalog: TrackCatalog, track_ids: list[str]) -> str:
    backups = [_track_phrase(catalog, track_id) for track_id in track_ids[1:3]]
    if len(backups) == 2:
        return f"Backups: {backups[0]}; {backups[1]}."
    if len(backups) == 1:
        return f"Backup: {backups[0]}."
    return "The remaining list keeps wider alternatives."


def _generate_compact_response(
    state: ConversationState,
    catalog: TrackCatalog,
    track_ids: list[str],
    require_good_words: bool = True,
) -> str:
    if not track_ids:
        return "I found a few tracks that should fit the direction you described."

    first_phrase = _track_phrase(catalog, track_ids[0])
    request_hint = _short_request(state)
    metadata = _metric_metadata(catalog, track_ids[0], require_good_words=require_good_words)
    backup = _labeled_backup(catalog, track_ids)
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


def _generate_concise_response(state: ConversationState, catalog: TrackCatalog, track_ids: list[str]) -> str:
    if not track_ids:
        return "I found a few tracks that should fit the direction you described."

    first_phrase = _track_phrase(catalog, track_ids[0])
    request_hint = _short_request(state)
    metadata = _compact_metadata(catalog, track_ids[0])
    backup = _compact_backup(state, catalog, track_ids)
    cue = _natural_cue(state, catalog)
    intent = infer_intent(state)
    opener_by_intent = {
        "specific_track": [
            f"I would try {first_phrase} first for the song clue \"{request_hint}\".",
            f"{first_phrase} is my closest first guess for \"{request_hint}\".",
            f"For this specific-track search, {first_phrase} is the lead pick.",
        ],
        "album": [
            f"I would start with {first_phrase} for the album cue \"{request_hint}\".",
            f"{first_phrase} is the cleanest record-context anchor here.",
            f"For the album-shaped hint, {first_phrase} is the first check.",
        ],
        "artist_exploration": [
            f"I would open the artist path with {first_phrase}.",
            f"{first_phrase} is the first artist-led recommendation I would test.",
            f"For this discovery path, I put {first_phrase} up front.",
        ],
        "cover_art": [
            f"I would start the visual clue search with {first_phrase}.",
            f"{first_phrase} is the first cover-art or image-hint anchor.",
            f"For the image-related cue, {first_phrase} is the lead.",
        ],
        "lyrics_theme": [
            f"I would start the theme search with {first_phrase}.",
            f"{first_phrase} is the strongest story or lyric anchor I found.",
            f"For the lyrical direction, {first_phrase} is the first check.",
        ],
    }
    opener = _pick(
        state,
        opener_by_intent.get(
            intent,
            [
                f"I would start with {first_phrase} for \"{request_hint}\".",
                f"{first_phrase} is the opening pick for this request.",
                f"For the mood you described, I put {first_phrase} first.",
            ],
        ),
        salt=f"concise-{intent}",
    )
    reason = _pick(
        state,
        [
            f"It fits because {metadata}.",
            f"The catalog evidence is that {metadata}.",
            f"I used it because {metadata}.",
        ],
        salt="concise-reason",
    )
    return " ".join(part for part in [opener, reason, cue, backup] if part)


def _track_list_phrase(catalog: TrackCatalog, track_ids: list[str], limit: int = 3) -> str:
    phrases = [_track_phrase(catalog, track_id) for track_id in track_ids[:limit]]
    return _join_words(phrases)


def _generate_setwise_response(state: ConversationState, catalog: TrackCatalog, track_ids: list[str]) -> str:
    if not track_ids:
        return "I found a few tracks that should fit the direction you described."

    request_hint = _short_request(state)
    first_phrase = _track_phrase(catalog, track_ids[0])
    shortlist = _track_list_phrase(catalog, track_ids, limit=3)
    cue = _natural_cue(state, catalog)
    intent = infer_intent(state)

    if intent == "specific_track":
        opener = _pick(
            state,
            [
                f"I would treat \"{request_hint}\" as a song-identification clue and check {shortlist} first.",
                f"For the exact-track clue, my first checks are {shortlist}.",
                f"I put {first_phrase} first, with {shortlist} as the tight search set.",
            ],
            salt="setwise-specific",
        )
        reason = "The ordering leans on title, artist, album, and repeated conversation clues before broad mood."
    elif intent == "album":
        opener = _pick(
            state,
            [
                f"For the album-shaped clue \"{request_hint}\", I would check {shortlist} first.",
                f"I kept the first checks close to the record context: {shortlist}.",
                f"The album-aware shortlist starts with {shortlist}.",
            ],
            salt="setwise-album",
        )
        reason = "I weighted album and artist context more heavily than general genre matches."
    elif intent == "artist_exploration":
        opener = _pick(
            state,
            [
                f"For the artist-led request, I would start with {shortlist}.",
                f"I opened the artist path with {shortlist}.",
                f"The first recommendations stay close to the artist cue: {shortlist}.",
            ],
            salt="setwise-artist",
        )
        reason = "The list favors nearby artist, album, and style evidence while leaving room for discovery."
    elif intent == "cover_art":
        opener = _pick(
            state,
            [
                f"For the visual clue, I would check {shortlist} first.",
                f"The cover-art or image-hint search starts with {shortlist}.",
                f"I kept the visual-clue shortlist to {shortlist}.",
            ],
            salt="setwise-cover",
        )
        reason = "I treated names, albums, and metadata as anchors because cover hints are easy to overfit."
    elif intent == "lyrics_theme":
        opener = _pick(
            state,
            [
                f"For the lyrical or theme clue, I would check {shortlist} first.",
                f"I started the theme search with {shortlist}.",
                f"The story-focused shortlist begins with {shortlist}.",
            ],
            salt="setwise-lyrics",
        )
        reason = "The ranking balances theme language with the track and artist metadata."
    else:
        opener = _pick(
            state,
            [
                f"For \"{request_hint}\", I would start with {shortlist}.",
                f"The first few picks I would try are {shortlist}.",
                f"I began with {shortlist} and then widened the rest of the list.",
            ],
            salt="setwise-general",
        )
        reason = "The front of the list follows the request text, while the tail keeps compatible alternatives."

    return " ".join(part for part in [opener, reason, cue] if part)


def _generate_natural_response(state: ConversationState, catalog: TrackCatalog, track_ids: list[str]) -> str:
    if not track_ids:
        return "I found a few tracks that should fit the direction you described."

    first_phrase = _track_phrase(catalog, track_ids[0])
    request_hint = _short_request(state)
    metadata = _compact_metadata(catalog, track_ids[0])
    backup = _compact_backup(state, catalog, track_ids)
    cue = _natural_cue(state, catalog)
    intent = infer_intent(state)

    if intent == "specific_track":
        lead = _pick(
            state,
            [
                f"For the song clue \"{request_hint}\", I would test {first_phrase} first.",
                f"{first_phrase} is my closest catalog match for \"{request_hint}\".",
                f"My first exact-track guess is {first_phrase}, based on \"{request_hint}\".",
                f"I would lead with {first_phrase} for this specific-song search.",
            ],
            salt="lead-specific",
        )
    elif intent == "album":
        lead = _pick(
            state,
            [
                f"For the album clue \"{request_hint}\", I would start with {first_phrase}.",
                f"{first_phrase} gives the cleanest album-aware anchor here.",
                f"I treated this as a record-context search and led with {first_phrase}.",
            ],
            salt="lead-album",
        )
    elif intent == "artist_exploration":
        lead = _pick(
            state,
            [
                f"For the artist-led cue \"{request_hint}\", I would open with {first_phrase}.",
                f"{first_phrase} is the first step I would try for this artist path.",
                f"I started the discovery path with {first_phrase}.",
            ],
            salt="lead-artist",
        )
    elif intent == "cover_art":
        lead = _pick(
            state,
            [
                f"For the visual clue \"{request_hint}\", I would start with {first_phrase}.",
                f"{first_phrase} is my first cover-art or image-hint anchor.",
                f"I used {first_phrase} as the lead for the visual search.",
            ],
            salt="lead-cover",
        )
    elif intent == "lyrics_theme":
        lead = _pick(
            state,
            [
                f"For the theme in \"{request_hint}\", I would start with {first_phrase}.",
                f"{first_phrase} is the strongest story or lyric anchor I found.",
                f"I led with {first_phrase} for the lyrical direction in the request.",
            ],
            salt="lead-lyrics",
        )
    else:
        lead = _pick(
            state,
            [
                f"For \"{request_hint}\", I would start with {first_phrase}.",
                f"I chose {first_phrase} first for the mood you described.",
                f"{first_phrase} is the opening pick for this discovery request.",
                f"I would try {first_phrase} first, then widen the list from there.",
            ],
            salt="lead-mood",
        )

    reason = _pick(
        state,
        [
            f"The useful metadata cue is that {metadata}.",
            f"I grounded that pick in the catalog because {metadata}.",
            f"The catalog evidence is simple: {metadata}.",
        ],
        salt="reason",
    )
    return " ".join(part for part in [lead, reason, cue, backup] if part)


def _clean_reason_parts(catalog: TrackCatalog, track_id: str) -> list[str]:
    view = catalog.view(track_id)
    parts = []
    artist = catalog.normalized_field(track_id, "artist_name")
    tags = []
    for tag in _tag_list(catalog, track_id, limit=4, require_good_words=True):
        normalized_tag = re.sub(r"\s+", " ", tag.lower()).strip()
        if normalized_tag and (normalized_tag in artist or artist in normalized_tag):
            continue
        tags.append(tag)
        if len(tags) >= 2:
            break
    if tags:
        parts.append(f"{_join_words(tags)} traits")
    year = _year_hint(catalog, track_id)
    if year:
        parts.append(f"a {year} release window")
    if view.album_name:
        parts.append(f"the album context of {view.album_name}")
    return parts[:3]


def _clean_reason(catalog: TrackCatalog, track_id: str) -> str:
    parts = _clean_reason_parts(catalog, track_id)
    if not parts:
        return "the track and artist metadata line up better than broader catalog matches"
    if len(parts) == 1:
        return parts[0]
    return _join_words(parts)


def _feedback_sentence(state: ConversationState, catalog: TrackCatalog) -> str:
    seed = _seed_phrase(state, catalog)
    if seed:
        return _pick(
            state,
            [
                f"Since {seed} was a useful signal earlier, I kept the list close to that lane.",
                f"I also carried forward the positive hint from {seed}.",
                f"The earlier fit around {seed} helped steer the nearby picks.",
            ],
            salt="polished-positive",
        )
    negative = _negative_phrase(state, catalog)
    if negative:
        return _pick(
            state,
            [
                f"I avoided leaning too heavily on the direction around {negative}, since that path looked weaker.",
                f"The list moves away from the weaker cue around {negative}.",
                f"I treated {negative} as a path to de-emphasize rather than repeat.",
            ],
            salt="polished-negative",
        )
    return ""


def _profile_sentence(state: ConversationState) -> str:
    profile = _profile_hint(state)
    if not profile:
        return ""
    profile = re.sub(r"^your\s+", "", profile, flags=re.IGNORECASE)
    return _pick(
        state,
        [
            f"I treated your {profile} as a soft tie-breaker.",
            f"Your {profile} also nudged the ordering when several tracks were close.",
            f"When the match was ambiguous, your {profile} helped choose the safer first pick.",
        ],
        salt="polished-profile",
    )


def _judge_profile_phrase(state: ConversationState) -> str:
    culture = _display_profile_value(state.user_profile.get("preferred_musical_culture"))
    country = _display_profile_value(state.user_profile.get("country_name"))
    age_group = _display_profile_value(state.user_profile.get("age_group"))
    if culture and country:
        return f"your {culture} background and {country} profile"
    if culture:
        return f"your {culture} listening background"
    if country and age_group:
        return f"your {age_group} listener profile from {country}"
    if country:
        return f"your {country} listener profile"
    return ""


def _judge_track_detail(catalog: TrackCatalog, track_id: str) -> str:
    view = catalog.view(track_id)
    tags = _tag_list(catalog, track_id, limit=3, require_good_words=True)
    bits = []
    if tags:
        bits.append(_join_words(tags))
    year = _year_hint(catalog, track_id)
    if year:
        bits.append(f"{year} release")
    if view.album_name:
        bits.append(f"album context from {view.album_name}")
    if not bits:
        bits.append("the strongest title and artist match")
    return _join_words(bits[:3])


def _judge_direction_phrase(state: ConversationState, intent: str) -> str:
    focus = _request_focus(state)
    if intent == "specific_track":
        return f"the specific song clue about \"{focus}\""
    if intent == "album":
        return f"the album-centered clue about \"{focus}\""
    if intent == "artist_exploration":
        return f"the artist-led direction in \"{focus}\""
    if intent == "cover_art":
        return f"the visual or cover-art clue in \"{focus}\""
    if intent == "lyrics_theme":
        return f"the lyric or theme clue in \"{focus}\""
    return f"the mood and style direction in \"{focus}\""


def _judge_feedback_sentence(state: ConversationState, catalog: TrackCatalog) -> str:
    seed = _seed_phrase(state, catalog)
    negative = _negative_phrase(state, catalog)
    if seed and negative:
        return _pick(
            state,
            [
                f"I kept the useful signal from {seed}, while moving away from the weaker path around {negative}.",
                f"The order keeps what worked about {seed} and avoids repeating the direction that made {negative} less convincing.",
                f"I used {seed} as the better anchor here and treated {negative} as a boundary for the search.",
            ],
            salt="judge-feedback-both",
        )
    if seed:
        return _pick(
            state,
            [
                f"Because {seed} had moved the session in the right direction, I kept the front of the list near that lane.",
                f"The positive signal from {seed} is carried into this pick instead of resetting the conversation.",
                f"I used {seed} as a useful anchor and looked for nearby evidence in the catalog.",
            ],
            salt="judge-feedback-positive",
        )
    if negative:
        return _pick(
            state,
            [
                f"I did not lean too hard on {negative}, since that earlier route looked less helpful.",
                f"The ranking backs away from the direction around {negative} and tries a cleaner match first.",
                f"I treated {negative} as a useful negative clue, so the lead pick is not just more of the same.",
            ],
            salt="judge-feedback-negative",
        )
    return ""


def _generate_judge_response(state: ConversationState, catalog: TrackCatalog, track_ids: list[str]) -> str:
    if not track_ids:
        return "I found a few tracks that should fit the direction you described."

    first_phrase = _track_phrase(catalog, track_ids[0])
    second_phrase = _track_phrase(catalog, track_ids[1]) if len(track_ids) > 1 else ""
    third_phrase = _track_phrase(catalog, track_ids[2]) if len(track_ids) > 2 else ""
    intent = infer_intent(state)
    direction = _judge_direction_phrase(state, intent)
    detail = _judge_track_detail(catalog, track_ids[0])
    profile = _judge_profile_phrase(state)
    feedback = _judge_feedback_sentence(state, catalog)

    if intent == "specific_track":
        lead = _pick(
            state,
            [
                f"I would put {first_phrase} first as the best direct answer to {direction}.",
                f"My lead answer is {first_phrase}, because it is the tightest catalog match for {direction}.",
                f"I would check {first_phrase} first before broadening the search, since it fits {direction}.",
            ],
            salt="judge-specific",
        )
    elif intent == "album":
        lead = _pick(
            state,
            [
                f"I started with {first_phrase} because it gives the strongest album-aware match for {direction}.",
                f"{first_phrase} is first because the record context is the clearest way into {direction}.",
                f"For {direction}, {first_phrase} is the most focused first check.",
            ],
            salt="judge-album",
        )
    elif intent == "artist_exploration":
        lead = _pick(
            state,
            [
                f"I led with {first_phrase} because it stays close to {direction} without making the list too narrow.",
                f"{first_phrase} is the first artist-path pick, and the rest of the ranking keeps adjacent options nearby.",
                f"For {direction}, I put {first_phrase} first and then widened into related tracks.",
            ],
            salt="judge-artist",
        )
    elif intent == "cover_art":
        lead = _pick(
            state,
            [
                f"I would test {first_phrase} first for {direction}, then use the next tracks as safeguards.",
                f"{first_phrase} is my lead for {direction}, where title and album context are the safest anchors.",
                f"For the image-based clue, I started with {first_phrase} and kept the backups close.",
            ],
            salt="judge-cover",
        )
    elif intent == "lyrics_theme":
        lead = _pick(
            state,
            [
                f"I led with {first_phrase} because it is the strongest first check for {direction}.",
                f"{first_phrase} is first because its catalog evidence lines up best with {direction}.",
                f"For the story or lyric angle, I would try {first_phrase} before the looser alternatives.",
            ],
            salt="judge-lyrics",
        )
    else:
        lead = _pick(
            state,
            [
                f"I started with {first_phrase} because it best matches {direction}.",
                f"{first_phrase} is first because it gives the cleanest entry point into {direction}.",
                f"I put {first_phrase} up front, then let the rest of the list explore nearby versions of {direction}.",
            ],
            salt="judge-mood",
        )

    reason = _pick(
        state,
        [
            f"The concrete clues I can verify are {detail}.",
            f"The grounded reason is {detail}, rather than a vague similarity claim.",
            f"What makes it a credible first pick is {detail}.",
        ],
        salt="judge-reason",
    )

    support = []
    if feedback:
        support.append(feedback)
    if profile:
        support.append(
            _pick(
                state,
                [
                    f"I also used {profile} only as a soft tie-breaker.",
                    f"When several candidates were close, {profile} helped order the safer options.",
                    f"{_capitalize_first(profile)} helped shape the final ordering without overriding the conversation.",
                ],
                salt="judge-profile",
            )
        )

    backup = ""
    if second_phrase and third_phrase:
        backup = _pick(
            state,
            [
                f"If the lead is a little off, I would next try {second_phrase}, then {third_phrase}.",
                f"I kept {second_phrase} and {third_phrase} right behind it as the closest alternate paths.",
                f"The next checks are {second_phrase} for a tighter match and {third_phrase} as a nearby fallback.",
            ],
            salt="judge-backup-two",
        )
    elif second_phrase:
        backup = f"If the lead is a little off, I would next try {second_phrase}."

    return " ".join(part for part in [lead, reason, *support[:2], backup] if part)


def _judge_v2_evidence(catalog: TrackCatalog, track_id: str) -> str:
    view = catalog.view(track_id)
    tags = _tag_list(catalog, track_id, limit=2, require_good_words=True)
    facts = []
    if tags:
        facts.append(_join_words(tags))
    year = _year_hint(catalog, track_id)
    if year:
        facts.append(f"the {year} era")
    if view.album_name:
        facts.append(f"{view.album_name} as the album anchor")
    if not facts:
        facts.append("the title and artist match")
    return _join_words(facts[:2])


def _judge_v2_feedback_clause(state: ConversationState, catalog: TrackCatalog) -> str:
    seed = _seed_phrase(state, catalog)
    negative = _negative_phrase(state, catalog)
    if seed and negative:
        return _pick(
            state,
            [
                f"keeps the useful thread from {seed} while avoiding the weaker turn around {negative}",
                f"borrows the better signal from {seed} and does not repeat the less helpful path around {negative}",
                f"uses {seed} as the center of gravity and treats {negative} as a boundary",
            ],
            salt="judge-v2-feedback-both",
        )
    if seed:
        return _pick(
            state,
            [
                f"keeps following the useful signal from {seed}",
                f"uses {seed} as the closest session anchor",
                f"continues the direction that worked around {seed}",
            ],
            salt="judge-v2-feedback-positive",
        )
    if negative:
        return _pick(
            state,
            [
                f"moves away from the less helpful route around {negative}",
                f"avoids making the same bet as {negative}",
                f"uses {negative} as a negative cue instead of repeating it",
            ],
            salt="judge-v2-feedback-negative",
        )
    return ""


def _generate_judge_v2_response(state: ConversationState, catalog: TrackCatalog, track_ids: list[str]) -> str:
    if not track_ids:
        return "I found a few tracks that should fit the direction you described."

    first_phrase = _track_phrase(catalog, track_ids[0])
    second_phrase = _track_phrase(catalog, track_ids[1]) if len(track_ids) > 1 else ""
    third_phrase = _track_phrase(catalog, track_ids[2]) if len(track_ids) > 2 else ""
    intent = infer_intent(state)
    focus = _request_focus(state)
    evidence = _judge_v2_evidence(catalog, track_ids[0])
    feedback_clause = _judge_v2_feedback_clause(state, catalog)
    profile = _judge_profile_phrase(state)

    if intent == "specific_track":
        lead = _pick(
            state,
            [
                f"I put {first_phrase} first as the most direct answer to \"{focus}\".",
                f"{first_phrase} is the tightest first check for the exact-song clue in \"{focus}\".",
                f"For \"{focus}\", the lead pick is {first_phrase} before the search opens wider.",
            ],
            salt="judge-v2-specific",
        )
    elif intent == "album":
        lead = _pick(
            state,
            [
                f"I stayed close to the record clue in \"{focus}\" and led with {first_phrase}.",
                f"{first_phrase} is first because this looks album-centered rather than purely mood-based.",
                f"For the album-shaped request, {first_phrase} gives the cleanest first anchor.",
            ],
            salt="judge-v2-album",
        )
    elif intent == "artist_exploration":
        lead = _pick(
            state,
            [
                f"I kept the artist path in focus and started with {first_phrase}.",
                f"{first_phrase} is the safest entry point for the artist-led direction in \"{focus}\".",
                f"I led with {first_phrase}, then used the rest of the list for nearby artist and style options.",
            ],
            salt="judge-v2-artist",
        )
    elif intent == "cover_art":
        lead = _pick(
            state,
            [
                f"For the visual clue in \"{focus}\", I started with {first_phrase}.",
                f"{first_phrase} is my first cover-clue check, with the next tracks kept close in case the image hint is indirect.",
                f"I treated this as a visual identification search and led with {first_phrase}.",
            ],
            salt="judge-v2-cover",
        )
    elif intent == "lyrics_theme":
        lead = _pick(
            state,
            [
                f"I treated \"{focus}\" as a lyric or story clue and put {first_phrase} first.",
                f"{first_phrase} is the clearest first play for the theme you described.",
                f"For the story angle in \"{focus}\", I would start with {first_phrase}.",
            ],
            salt="judge-v2-lyrics",
        )
    else:
        lead = _pick(
            state,
            [
                f"I leaned into \"{focus}\" and started with {first_phrase}.",
                f"{first_phrase} is the safest first play for the mood you described in \"{focus}\".",
                f"I put {first_phrase} first, then let the rest of the list explore nearby versions of \"{focus}\".",
            ],
            salt="judge-v2-mood",
        )

    reason = _pick(
        state,
        [
            f"The strongest grounded clue is {evidence}.",
            f"The evidence I can safely cite is {evidence}.",
            f"It earns the lead slot through {evidence}.",
        ],
        salt="judge-v2-reason",
    )

    context = ""
    if feedback_clause and profile:
        context = _pick(
            state,
            [
                f"The ordering {feedback_clause}; I used {profile} only as a tie-breaker.",
                f"It also {feedback_clause}; I used {profile} only to settle close calls.",
            ],
            salt="judge-v2-context-both",
        )
    elif feedback_clause:
        context = _pick(
            state,
            [
                f"The rest of the ranking {feedback_clause}.",
                f"That choice {feedback_clause}.",
            ],
            salt="judge-v2-context-feedback",
        )
    elif profile:
        context = _pick(
            state,
            [
                f"I used {profile} only as a soft tie-breaker.",
                f"{_capitalize_first(profile)} nudged the ordering without overriding the request.",
            ],
            salt="judge-v2-context-profile",
        )

    backup = ""
    if second_phrase and third_phrase:
        backup = _pick(
            state,
            [
                f"Next I would check {second_phrase}, then {third_phrase}.",
                f"{second_phrase} and {third_phrase} stay close enough to recover if the first guess misses.",
                f"The backups are {second_phrase} for continuity and {third_phrase} for a nearby alternate.",
            ],
            salt="judge-v2-backups",
        )
    elif second_phrase:
        backup = f"Next I would check {second_phrase}."

    return " ".join(part for part in [lead, reason, context, backup] if part)


def _generate_judge_v3_response(state: ConversationState, catalog: TrackCatalog, track_ids: list[str]) -> str:
    if not track_ids:
        return "I found a few tracks that should fit the direction you described."

    first_phrase = _track_phrase(catalog, track_ids[0])
    second_phrase = _track_phrase(catalog, track_ids[1]) if len(track_ids) > 1 else ""
    third_phrase = _track_phrase(catalog, track_ids[2]) if len(track_ids) > 2 else ""
    intent = infer_intent(state)
    direction = _judge_direction_phrase(state, intent)
    first_detail = _judge_track_detail(catalog, track_ids[0])
    second_detail = _judge_track_detail(catalog, track_ids[1]) if len(track_ids) > 1 else ""
    feedback = _judge_feedback_sentence(state, catalog)
    profile = _judge_profile_phrase(state)

    lead = _pick(
        state,
        [
            f"I read this as {direction}, so I put {first_phrase} first.",
            f"My first pick is {first_phrase} because the conversation points most strongly toward {direction}.",
            f"I treated the request as {direction} and used {first_phrase} as the lead answer.",
        ],
        salt=f"judge-v3-lead-{intent}",
    )
    reason = _pick(
        state,
        [
            f"The verifiable catalog evidence is {first_detail}.",
            f"The concrete match I can cite is {first_detail}.",
            f"I am grounding that choice in {first_detail}.",
        ],
        salt="judge-v3-reason",
    )

    context_parts = []
    if feedback:
        context_parts.append(feedback)
    if profile:
        context_parts.append(
            _pick(
                state,
                [
                    f"I only used {profile} after the conversation clues were matched.",
                    f"{_capitalize_first(profile)} helped break close ties, but did not override the request.",
                    f"I treated {profile} as a secondary signal for the final order.",
                ],
                salt="judge-v3-profile",
            )
        )

    backup = ""
    if second_phrase and third_phrase:
        backup = _pick(
            state,
            [
                f"I put {second_phrase} next because {second_detail}, with {third_phrase} as a nearby fallback.",
                f"If the lead is not quite the one, {second_phrase} is the next focused check and {third_phrase} gives the list another close route.",
                f"The second and third checks are {second_phrase} and {third_phrase}, so the list can recover without drifting too far.",
            ],
            salt="judge-v3-backup-two",
        )
    elif second_phrase:
        backup = f"If the lead is not quite the one, {second_phrase} is the next focused check."

    return " ".join(part for part in [lead, reason, *context_parts[:2], backup] if part)


def _generate_judge_mix_response(state: ConversationState, catalog: TrackCatalog, track_ids: list[str]) -> str:
    bucket = int(
        hashlib.md5(f"{state.session_id}:{state.turn_number}:judge_mix".encode("utf-8")).hexdigest()[:8],
        16,
    ) % 10
    style = [
        "judge_v2",
        "judge_v2",
        "judge_v2",
        "judge_v2",
        "concise",
        "concise",
        "natural",
        "natural",
        "setwise",
        "setwise",
    ][bucket]
    if style == "judge_v2":
        return _generate_judge_v2_response(state, catalog, track_ids)
    if style == "concise":
        return _generate_concise_response(state, catalog, track_ids)
    if style == "natural":
        return _generate_natural_response(state, catalog, track_ids)
    return _generate_setwise_response(state, catalog, track_ids)


def _generate_judge_brief_response(state: ConversationState, catalog: TrackCatalog, track_ids: list[str]) -> str:
    if not track_ids:
        return "I found a few tracks that should fit the direction you described."

    first_phrase = _track_phrase(catalog, track_ids[0])
    second_phrase = _track_phrase(catalog, track_ids[1]) if len(track_ids) > 1 else ""
    third_phrase = _track_phrase(catalog, track_ids[2]) if len(track_ids) > 2 else ""
    intent = infer_intent(state)
    focus = _request_focus(state, max_words=14)
    evidence = _judge_v2_evidence(catalog, track_ids[0])
    feedback_clause = _judge_v2_feedback_clause(state, catalog)
    profile = _judge_profile_phrase(state)

    opener = _pick(
        state,
        [
            f"Lead pick: {first_phrase} for \"{focus}\".",
            f"I would check {first_phrase} first for \"{focus}\".",
            f"First answer: {first_phrase}, aimed at \"{focus}\".",
            f"{first_phrase} is the front-runner for \"{focus}\".",
            f"For this {intent.replace('_', ' ')} request, I put {first_phrase} first.",
        ],
        salt=f"judge-brief-opener-{intent}",
    )
    reason = _pick(
        state,
        [
            f"Reason: {evidence}.",
            f"Catalog signal: {evidence}.",
            f"Grounding: {evidence}.",
            f"Verified cue: {evidence}.",
            f"Best visible evidence: {evidence}.",
        ],
        salt="judge-brief-reason",
    )

    context = ""
    if feedback_clause and profile:
        context = _pick(
            state,
            [
                f"It also {feedback_clause}; I used {profile} only to break close ties.",
                f"The list {feedback_clause}, with {profile} as a secondary cue.",
            ],
            salt="judge-brief-context-both",
        )
    elif feedback_clause:
        context = _pick(
            state,
            [
                f"The rest of the list {feedback_clause}.",
                f"I used the history because it {feedback_clause}.",
            ],
            salt="judge-brief-context-feedback",
        )
    elif profile:
        context = _pick(
            state,
            [
                f"I used {profile} only to break close ties.",
                f"I used {profile} as a secondary cue.",
            ],
            salt="judge-brief-context-profile",
        )

    backup = ""
    if second_phrase and third_phrase:
        backup = _pick(
            state,
            [
                f"Next checks: {second_phrase}; {third_phrase}.",
                f"Backups: {second_phrase}, then {third_phrase}.",
                f"If it misses, try {second_phrase} and {third_phrase}.",
            ],
            salt="judge-brief-backups",
        )
    elif second_phrase:
        backup = f"Next check: {second_phrase}."

    return " ".join(part for part in [opener, reason, context, backup] if part)


def _generate_judge_planned_response(state: ConversationState, catalog: TrackCatalog, track_ids: list[str]) -> str:
    if not track_ids:
        return "I found a few tracks that should fit the direction you described."

    first_phrase = _track_phrase(catalog, track_ids[0])
    second_phrase = _track_phrase(catalog, track_ids[1]) if len(track_ids) > 1 else ""
    third_phrase = _track_phrase(catalog, track_ids[2]) if len(track_ids) > 2 else ""
    intent = infer_intent(state)
    focus = _request_focus(state, max_words=16)
    evidence = _judge_v2_evidence(catalog, track_ids[0])
    feedback_clause = _judge_v2_feedback_clause(state, catalog)
    profile = _judge_profile_phrase(state)

    if intent == "specific_track":
        lead = _pick(
            state,
            [
                f"I read this as an exact-song search around \"{focus}\", so I would start with {first_phrase}.",
                f"For the specific track clue in \"{focus}\", {first_phrase} is the first answer I would test.",
                f"The request sounds like a song-identification turn, and {first_phrase} is the tightest lead.",
            ],
            salt="judge-planned-specific",
        )
    elif intent == "album":
        lead = _pick(
            state,
            [
                f"I treated \"{focus}\" as an album-aware clue, so I would open with {first_phrase}.",
                f"The request points more to record context than a loose mood, which makes {first_phrase} my first check.",
                f"For the album-shaped hint, I would put {first_phrase} at the front of the list.",
            ],
            salt="judge-planned-album",
        )
    elif intent == "artist_exploration":
        lead = _pick(
            state,
            [
                f"I kept the artist direction in \"{focus}\" central and started with {first_phrase}.",
                f"For this artist-led turn, {first_phrase} is the safest opener before widening the list.",
                f"I would begin the discovery path with {first_phrase}, then use the next tracks for nearby artist or style options.",
            ],
            salt="judge-planned-artist",
        )
    elif intent == "cover_art":
        lead = _pick(
            state,
            [
                f"I treated \"{focus}\" as a visual clue and used {first_phrase} as the first catalog anchor.",
                f"For the cover-art style hint, I would test {first_phrase} first and keep the next picks close.",
                f"The image-related clue needs a cautious first guess, so I led with {first_phrase}.",
            ],
            salt="judge-planned-cover",
        )
    elif intent == "lyrics_theme":
        lead = _pick(
            state,
            [
                f"I read \"{focus}\" as a lyric or story cue, and {first_phrase} is the clearest first play.",
                f"For the theme you described, I would start with {first_phrase} before trying looser matches.",
                f"The request feels theme-driven, so I put {first_phrase} first and kept the rest nearby.",
            ],
            salt="judge-planned-lyrics",
        )
    else:
        lead = _pick(
            state,
            [
                f"I read \"{focus}\" as a mood and style request, so I would start with {first_phrase}.",
                f"For the direction in \"{focus}\", {first_phrase} gives the cleanest first step.",
                f"I put {first_phrase} first because it is the strongest entry point into the sound you described.",
            ],
            salt="judge-planned-mood",
        )

    reason = _pick(
        state,
        [
            f"The safest evidence I can cite is {evidence}, so the explanation stays grounded in catalog data.",
            f"It has a clearer catalog footing through {evidence}, rather than just a generic similarity claim.",
            f"What makes the lead defensible is {evidence}; I avoided inventing details beyond the metadata.",
        ],
        salt="judge-planned-reason",
    )

    context = ""
    if feedback_clause and profile:
        context = _pick(
            state,
            [
                f"The ordering {feedback_clause}, with {profile} used only after the conversation clues matched.",
                f"I also let the list {feedback_clause}; {profile} only helps break close ties.",
            ],
            salt="judge-planned-context-both",
        )
    elif feedback_clause:
        context = _pick(
            state,
            [
                f"The rest of the list {feedback_clause} instead of resetting the session.",
                f"I also used the session history because it {feedback_clause}.",
            ],
            salt="judge-planned-context-feedback",
        )
    elif profile:
        context = _pick(
            state,
            [
                f"I used {profile} as a secondary cue, not as a replacement for the request itself.",
                f"{_capitalize_first(profile)} nudged the ordering only when the track evidence was close.",
            ],
            salt="judge-planned-context-profile",
        )

    backup = ""
    if second_phrase and third_phrase:
        backup = _pick(
            state,
            [
                f"I kept {second_phrase} and {third_phrase} next so the top of the list can recover without drifting away.",
                f"If the first pick misses, {second_phrase} is the nearest follow-up and {third_phrase} gives a slightly wider route.",
                f"The next two checks, {second_phrase} and {third_phrase}, cover adjacent evidence rather than unrelated variety.",
            ],
            salt="judge-planned-backups",
        )
    elif second_phrase:
        backup = f"If the first pick misses, {second_phrase} is the nearest follow-up."

    return " ".join(part for part in [lead, reason, context, backup] if part)


def _generate_judge_compact_mix_response(
    state: ConversationState,
    catalog: TrackCatalog,
    track_ids: list[str],
) -> str:
    bucket = int(
        hashlib.md5(
            f"{state.session_id}:{state.turn_number}:judge_compact_mix".encode("utf-8")
        ).hexdigest()[:8],
        16,
    ) % 12
    style = [
        "judge_v2",
        "judge_v2",
        "judge_v2",
        "judge_brief",
        "judge_brief",
        "compact_broad",
        "compact_broad",
        "compact_broad",
        "concise",
        "natural",
        "setwise",
        "setwise",
    ][bucket]
    if style == "judge_v2":
        return _generate_judge_v2_response(state, catalog, track_ids)
    if style == "judge_brief":
        return _generate_judge_brief_response(state, catalog, track_ids)
    if style == "compact_broad":
        return _generate_compact_response(state, catalog, track_ids, require_good_words=False)
    if style == "concise":
        return _generate_concise_response(state, catalog, track_ids)
    if style == "natural":
        return _generate_natural_response(state, catalog, track_ids)
    return _generate_setwise_response(state, catalog, track_ids)


def _generate_judge_clean_mix_response(
    state: ConversationState,
    catalog: TrackCatalog,
    track_ids: list[str],
) -> str:
    bucket = int(
        hashlib.md5(
            f"{state.session_id}:{state.turn_number}:judge_clean_mix".encode("utf-8")
        ).hexdigest()[:8],
        16,
    ) % 12
    style = [
        "judge_v2",
        "judge_v2",
        "judge_v2",
        "judge_brief",
        "judge_brief",
        "judge_brief",
        "compact",
        "compact",
        "compact",
        "compact",
        "compact",
        "compact",
    ][bucket]
    if style == "judge_v2":
        return _generate_judge_v2_response(state, catalog, track_ids)
    if style == "judge_brief":
        return _generate_judge_brief_response(state, catalog, track_ids)
    return _generate_compact_response(state, catalog, track_ids)


def _generate_judge_balanced_mix_response(
    state: ConversationState,
    catalog: TrackCatalog,
    track_ids: list[str],
) -> str:
    bucket = int(
        hashlib.md5(
            f"{state.session_id}:{state.turn_number}:judge_balanced_mix".encode("utf-8")
        ).hexdigest()[:8],
        16,
    ) % 12
    style = [
        "judge_planned",
        "judge_planned",
        "judge_planned",
        "judge_v2",
        "judge_v2",
        "judge_brief",
        "judge_brief",
        "compact",
        "compact",
        "compact",
        "natural",
        "setwise",
    ][bucket]
    if style == "judge_planned":
        return _generate_judge_planned_response(state, catalog, track_ids)
    if style == "judge_v2":
        return _generate_judge_v2_response(state, catalog, track_ids)
    if style == "judge_brief":
        return _generate_judge_brief_response(state, catalog, track_ids)
    if style == "natural":
        return _generate_natural_response(state, catalog, track_ids)
    if style == "setwise":
        return _generate_setwise_response(state, catalog, track_ids)
    return _generate_compact_response(state, catalog, track_ids)


def _generate_polished_response(state: ConversationState, catalog: TrackCatalog, track_ids: list[str]) -> str:
    if not track_ids:
        return "I found a few tracks that should fit the direction you described."

    first_phrase = _track_phrase(catalog, track_ids[0])
    request_hint = _short_request(state)
    reason = _clean_reason(catalog, track_ids[0])
    backup_phrases = [_track_phrase(catalog, track_id) for track_id in track_ids[1:3]]
    intent = infer_intent(state)

    if intent == "specific_track":
        lead = _pick(
            state,
            [
                f"My best first answer to the song clue \"{request_hint}\" is {first_phrase}.",
                f"I would check {first_phrase} first for the specific track clue \"{request_hint}\".",
                f"For the exact-song clue \"{request_hint}\", {first_phrase} is the strongest lead I found.",
            ],
            salt="polished-specific",
        )
    elif intent == "album":
        lead = _pick(
            state,
            [
                f"For the album-shaped clue \"{request_hint}\", I would start with {first_phrase}.",
                f"{first_phrase} is the cleanest record-context lead for this request.",
                f"I treated this as an album-aware search and put {first_phrase} first.",
            ],
            salt="polished-album",
        )
    elif intent == "artist_exploration":
        lead = _pick(
            state,
            [
                f"For the artist-led request \"{request_hint}\", I would open with {first_phrase}.",
                f"{first_phrase} gives the strongest first step for the artist direction in \"{request_hint}\".",
                f"I started the artist path with {first_phrase} and kept nearby options after it for \"{request_hint}\".",
            ],
            salt="polished-artist",
        )
    elif intent == "cover_art":
        lead = _pick(
            state,
            [
                f"For the visual clue \"{request_hint}\", I would test {first_phrase} first.",
                f"{first_phrase} is my first anchor for the cover or image hint in \"{request_hint}\".",
                f"I used {first_phrase} as the lead while keeping the rest of the list close to \"{request_hint}\".",
            ],
            salt="polished-cover",
        )
    elif intent == "lyrics_theme":
        lead = _pick(
            state,
            [
                f"For the lyric or theme clue, I would start with {first_phrase}.",
                f"{first_phrase} is the strongest first check for the story in \"{request_hint}\".",
                f"I led with {first_phrase} because it best matches the theme in \"{request_hint}\".",
            ],
            salt="polished-lyrics",
        )
    else:
        lead = _pick(
            state,
            [
                f"For \"{request_hint}\", I would start with {first_phrase}.",
                f"I put {first_phrase} first for the mood and style you described.",
                f"{first_phrase} is the safest opening pick before widening the list.",
                f"I would try {first_phrase} first, then use the rest of the ranking as nearby alternatives.",
            ],
            salt="polished-mood",
        )

    reason_sentence = _pick(
        state,
        [
            f"It fits because the metadata points to {reason}.",
            f"The useful catalog evidence is {reason}.",
            f"I chose it because the strongest evidence is {reason}.",
        ],
        salt="polished-reason",
    )
    backup_sentence = ""
    if len(backup_phrases) == 2:
        backup_sentence = _pick(
            state,
            [
                f"If that is not quite right, I would next try {backup_phrases[0]} and {backup_phrases[1]}.",
                f"The next two checks are {backup_phrases[0]} and {backup_phrases[1]}.",
                f"I kept {backup_phrases[0]} and {backup_phrases[1]} as close backups.",
            ],
            salt="polished-backups",
        )
    elif len(backup_phrases) == 1:
        backup_sentence = f"If that is not quite right, I would next try {backup_phrases[0]}."

    cues = [_feedback_sentence(state, catalog), _profile_sentence(state)]
    return " ".join(part for part in [lead, reason_sentence, *cues, backup_sentence] if part)


def generate_response(
    state: ConversationState,
    catalog: TrackCatalog,
    track_ids: list[str],
    style: str = "compact",
) -> str:
    if style == "compact":
        return _generate_compact_response(state, catalog, track_ids)
    if style == "compact_broad":
        return _generate_compact_response(state, catalog, track_ids, require_good_words=False)
    if style == "concise":
        return _generate_concise_response(state, catalog, track_ids)
    if style == "setwise":
        return _generate_setwise_response(state, catalog, track_ids)
    if style == "natural":
        return _generate_natural_response(state, catalog, track_ids)
    if style == "polished":
        return _generate_polished_response(state, catalog, track_ids)
    if style == "judge_v1":
        return _generate_judge_response(state, catalog, track_ids)
    if style == "judge_v2":
        return _generate_judge_v2_response(state, catalog, track_ids)
    if style == "judge_v3":
        return _generate_judge_v3_response(state, catalog, track_ids)
    if style == "judge_mix":
        return _generate_judge_mix_response(state, catalog, track_ids)
    if style == "judge_brief":
        return _generate_judge_brief_response(state, catalog, track_ids)
    if style == "judge_planned":
        return _generate_judge_planned_response(state, catalog, track_ids)
    if style == "judge_compact_mix":
        return _generate_judge_compact_mix_response(state, catalog, track_ids)
    if style == "judge_clean_mix":
        return _generate_judge_clean_mix_response(state, catalog, track_ids)
    if style == "judge_balanced_mix":
        return _generate_judge_balanced_mix_response(state, catalog, track_ids)
    raise ValueError(f"Unsupported response style: {style!r}")
