from __future__ import annotations

from .data import TrackCatalog
from .fusion import infer_intent
from .state import ConversationState


def _track_phrase(catalog: TrackCatalog, track_id: str) -> str:
    view = catalog.view(track_id)
    return f'"{view.track_name}" by {view.artist_name}'


def generate_response(state: ConversationState, catalog: TrackCatalog, track_ids: list[str]) -> str:
    if not track_ids:
        return "I found a few tracks that should fit the direction you described."

    first = catalog.view(track_ids[0])
    first_phrase = _track_phrase(catalog, track_ids[0])
    second_phrase = _track_phrase(catalog, track_ids[1]) if len(track_ids) > 1 else ""
    tags = ", ".join(first.tag_list.split(", ")[:4])
    goal_hint = state.listener_goal.rstrip(".")
    intent = infer_intent(state)

    if intent == "specific_track":
        return (
            f"I would start with {first_phrase}, since it best matches the specific clue in your request. "
            f"I also kept the rest of the list close to the same artist, era, or style so the alternatives stay useful."
        )
    if intent == "album":
        return (
            f"{first_phrase} is the strongest album-aware pick here, with {first.album_name} anchoring the match. "
            f"I then added nearby tracks that preserve the album, artist, or release-era context you pointed toward."
        )
    if intent == "artist_exploration":
        return (
            f"I leaned into the artist direction with {first_phrase}. "
            f"{second_phrase + ' gives you another nearby path, and ' if second_phrase else ''}"
            f"the remaining picks keep the discovery focused without repeating one narrow lane too much."
        )
    if state.positive_seed_ids:
        seed = _track_phrase(catalog, state.positive_seed_ids[-1]) if catalog.has_track(state.positive_seed_ids[-1]) else "the previous closer match"
        return (
            f"Because {seed} seemed to move in the right direction, I started with {first_phrase} and carried forward "
            f"the shared feel around {tags or 'the same musical texture'}. The rest of the list keeps that thread while adding some variety."
        )
    if intent == "cover_art":
        return (
            f"I started with {first_phrase}; its metadata gives the closest available anchor for the visual or cover-art clue. "
            f"I also included related artist and album-context matches in case the image hint maps to a nearby release."
        )
    if intent == "lyrics_theme":
        return (
            f"{first_phrase} is my lead pick for the lyrical or thematic direction in your request. "
            f"The follow-up tracks stay close to the same mood and story cues rather than just matching popularity."
        )
    return (
        f"For {goal_hint or 'the mood you described'}, I would start with {first_phrase}: it lines up with "
        f"{tags or 'the style signals in the request'}. I also added nearby tracks that keep the same energy while broadening the mix."
    )
