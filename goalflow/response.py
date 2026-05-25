from __future__ import annotations

import hashlib
import re

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
    "nigga",
    "nigger",
    "shit",
}

BAD_TAG_PHRASES = (
    "albums i own",
    "cds i own",
    "check out",
    "favourites",
    "favorites",
    "ifs and buts",
    "i own",
    "lastfm",
    "my ",
    "playlist",
    "rotation",
    "seen live",
    "si related",
    "songs i",
    "songs ya",
    "sruuu",
    "to live by",
)


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


def _tag_list(catalog: TrackCatalog, track_id: str, limit: int = 3, require_good_words: bool = True) -> list[str]:
    tags = []
    seen = set()
    for raw_tag in catalog.view(track_id).tag_list.split(", "):
        tag = raw_tag.strip()
        key = tag.lower()
        if not tag or key in seen:
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
    raise ValueError(f"Unsupported response style: {style!r}")
