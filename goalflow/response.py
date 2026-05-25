from __future__ import annotations

import hashlib

from .data import TrackCatalog
from .fusion import infer_intent
from .state import ConversationState


def _track_phrase(catalog: TrackCatalog, track_id: str) -> str:
    view = catalog.view(track_id)
    return f'"{view.track_name}" by {view.artist_name}'


def _variant_index(state: ConversationState, modulo: int) -> int:
    key = f"{state.session_id}:{state.turn_number}:{state.current_user_query}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def _profile_hint(state: ConversationState) -> str:
    culture = state.user_profile.get("preferred_musical_culture")
    country = state.user_profile.get("country_name")
    if culture:
        return f" with your {culture} listening background in mind"
    if country:
        return f" while keeping your {country} profile in mind"
    return ""


def _tag_hint(catalog: TrackCatalog, track_id: str) -> str:
    tags = [tag for tag in catalog.view(track_id).tag_list.split(", ") if tag][:4]
    if len(tags) >= 2:
        return ", ".join(tags[:-1]) + f", and {tags[-1]}"
    return ", ".join(tags)


def generate_response(state: ConversationState, catalog: TrackCatalog, track_ids: list[str]) -> str:
    if not track_ids:
        return "I found a few tracks that should fit the direction you described."

    first = catalog.view(track_ids[0])
    first_phrase = _track_phrase(catalog, track_ids[0])
    second_phrase = _track_phrase(catalog, track_ids[1]) if len(track_ids) > 1 else ""
    third_phrase = _track_phrase(catalog, track_ids[2]) if len(track_ids) > 2 else ""
    tags = _tag_hint(catalog, track_ids[0])
    goal_hint = state.listener_goal.rstrip(".")
    intent = infer_intent(state)
    profile = _profile_hint(state)
    variant = _variant_index(state, 3)

    if intent == "specific_track":
        responses = [
            (
                f"I would start with {first_phrase}, since it best matches the specific clue in your request. "
                f"I kept the follow-up picks close to the same artist, era, or style so the alternatives stay useful."
            ),
            (
                f"My strongest match is {first_phrase}. The next tracks stay near the same title, artist, or release context "
                f"rather than drifting into a generic playlist."
            ),
            (
                f"{first_phrase} is the clearest answer to the track-identification hint. I added nearby options like "
                f"{second_phrase or 'the next few picks'} in case the clue points to a neighboring version or release."
            ),
        ]
        return responses[variant]
    if intent == "album":
        responses = [
            (
                f"{first_phrase} is the strongest album-aware pick here, with {first.album_name} anchoring the match. "
                f"I then added tracks that preserve the album, artist, or release-era context you pointed toward."
            ),
            (
                f"I led with {first_phrase} because the album signal around {first.album_name} is the cleanest fit. "
                f"The rest of the list stays close to that release family without repeating one title."
            ),
            (
                f"For the album clue, {first_phrase} gives the best anchor. {second_phrase or 'The next recommendation'} "
                f"keeps the same record-context path open."
            ),
        ]
        return responses[variant]
    if intent == "artist_exploration":
        responses = [
            (
                f"I leaned into the artist direction with {first_phrase}. "
                f"{second_phrase + ' gives you another nearby path, and ' if second_phrase else ''}"
                f"the remaining picks keep discovery focused without repeating one narrow lane too much."
            ),
            (
                f"{first_phrase} is my entry point for this artist-led request{profile}. "
                f"I mixed in related tracks so the list has continuity plus a little room to explore."
            ),
            (
                f"I started from {first_phrase} and then widened the circle with {second_phrase or 'nearby catalog matches'}. "
                f"That should keep the artist thread visible while avoiding a flat block of duplicates."
            ),
        ]
        return responses[variant]
    if state.positive_seed_ids:
        seed = _track_phrase(catalog, state.positive_seed_ids[-1]) if catalog.has_track(state.positive_seed_ids[-1]) else "the previous closer match"
        responses = [
            (
                f"Because {seed} seemed to move in the right direction, I started with {first_phrase} and carried forward "
                f"the shared feel around {tags or 'the same musical texture'}. The rest of the list keeps that thread while adding variety."
            ),
            (
                f"I treated {seed} as useful feedback and moved further in that lane with {first_phrase}. "
                f"{third_phrase or 'The later picks'} adds a slightly different angle so the set does not become too narrow."
            ),
            (
                f"Since the earlier match was getting closer, {first_phrase} is the lead pick here. "
                f"I favored tracks with similar cues, especially {tags or 'matching mood and style signals'}, while leaving space for discovery."
            ),
        ]
        return responses[variant]
    if intent == "cover_art":
        responses = [
            (
                f"I started with {first_phrase}; its metadata gives the closest available anchor for the visual or cover-art clue. "
                f"I also included related artist and album-context matches in case the image hint maps to a nearby release."
            ),
            (
                f"For the cover-art clue, {first_phrase} is the safest lead. I used album and artist context for the rest "
                f"because visual descriptions often land near a release rather than one exact track."
            ),
            (
                f"{first_phrase} is my first guess from the visual hint. The surrounding picks keep the same album-catalog area "
                f"so a close cover or companion release still has a chance to appear."
            ),
        ]
        return responses[variant]
    if intent == "lyrics_theme":
        responses = [
            (
                f"{first_phrase} is my lead pick for the lyrical or thematic direction in your request. "
                f"The follow-up tracks stay close to the same mood and story cues rather than just matching popularity."
            ),
            (
                f"I chose {first_phrase} first because it best matches the theme-language in the conversation. "
                f"The next recommendations keep the same emotional shape, with {second_phrase or 'a nearby track'} as a backup route."
            ),
            (
                f"For the lyric or story clue, {first_phrase} gives the strongest anchor. I then kept the list around "
                f"{tags or 'similar mood and genre cues'} so the explanation stays grounded."
            ),
        ]
        return responses[variant]
    responses = [
        (
            f"For {goal_hint or 'the mood you described'}, I would start with {first_phrase}: it lines up with "
            f"{tags or 'the style signals in the request'}. I also added nearby tracks that keep the same energy while broadening the mix."
        ),
        (
            f"I led with {first_phrase}{profile} because it fits the requested mood through "
            f"{tags or 'its genre and metadata cues'}. {second_phrase or 'The next pick'} keeps the flow connected without making the set repetitive."
        ),
        (
            f"{first_phrase} is the first pick I would try for this request. I followed it with "
            f"{second_phrase or 'similar catalog matches'} and {third_phrase or 'a few wider options'} to balance fit, freshness, and variety."
        ),
    ]
    return responses[variant]
