from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .data import TrackCatalog, as_text, normalize_text


POSITIVE_LABEL = "MOVES_TOWARD_GOAL"
NEGATIVE_LABEL = "DOES_NOT_MOVE_TOWARD_GOAL"


@dataclass
class ConversationState:
    session_id: str
    user_id: str
    turn_number: int
    session_date: str
    current_user_query: str
    user_profile: dict[str, Any]
    conversation_goal: dict[str, Any]
    history_turns: list[dict[str, Any]]
    progress_by_turn: dict[int, str | None]
    previous_music_track_ids: list[str] = field(default_factory=list)
    positive_seed_ids: list[str] = field(default_factory=list)
    negative_seed_ids: list[str] = field(default_factory=list)
    gold_track_id: str | None = None

    @property
    def listener_goal(self) -> str:
        return as_text(self.conversation_goal.get("listener_goal"))

    @property
    def specificity(self) -> str:
        return as_text(self.conversation_goal.get("specificity"))

    @property
    def category(self) -> str:
        return as_text(self.conversation_goal.get("category"))


def role_at_turn(conversations: list[dict[str, Any]], turn_number: int, role: str) -> dict[str, Any] | None:
    for turn in conversations:
        if turn.get("turn_number") == turn_number and turn.get("role") == role:
            return turn
    return None


def progress_map(goal_progress_assessments: list[dict[str, Any]]) -> dict[int, str | None]:
    out = {}
    for item in goal_progress_assessments or []:
        out[int(item.get("turn_number"))] = item.get("goal_progress_assessment")
    return out


def track_label_for_history_turn(progress: dict[int, str | None], turn_number: int) -> str | None:
    # The released data has no label at turn 1 and labels at turns 2..8.
    # Samples show label t describes the user's reaction to the music at t-1.
    return progress.get(turn_number + 1)


def build_state_for_dev_turn(item: dict[str, Any], turn_number: int) -> ConversationState:
    conversations = item["conversations"]
    progress = progress_map(item.get("goal_progress_assessments", []))
    user_turn = role_at_turn(conversations, turn_number, "user")
    music_turn = role_at_turn(conversations, turn_number, "music")
    history = [turn for turn in conversations if int(turn.get("turn_number", 0)) < turn_number]

    previous_music = []
    positives = []
    negatives = []
    for turn in history:
        if turn.get("role") != "music":
            continue
        track_id = turn.get("content")
        previous_music.append(track_id)
        label = track_label_for_history_turn(progress, int(turn.get("turn_number")))
        if label == POSITIVE_LABEL:
            positives.append(track_id)
        elif label == NEGATIVE_LABEL:
            negatives.append(track_id)

    return ConversationState(
        session_id=item["session_id"],
        user_id=item["user_id"],
        turn_number=turn_number,
        session_date=item.get("session_date", ""),
        current_user_query=as_text(user_turn.get("content") if user_turn else ""),
        user_profile=item.get("user_profile", {}),
        conversation_goal=item.get("conversation_goal", {}),
        history_turns=history,
        progress_by_turn=progress,
        previous_music_track_ids=previous_music,
        positive_seed_ids=positives,
        negative_seed_ids=negatives,
        gold_track_id=music_turn.get("content") if music_turn else None,
    )


def build_state_for_blind_item(item: dict[str, Any]) -> ConversationState:
    conversations = item["conversations"]
    progress = progress_map(item.get("goal_progress_assessments", []))
    current = conversations[-1]
    turn_number = int(current.get("turn_number"))
    history = conversations[:-1]

    previous_music = []
    positives = []
    negatives = []
    for turn in history:
        if turn.get("role") != "music":
            continue
        track_id = turn.get("content")
        previous_music.append(track_id)
        label = track_label_for_history_turn(progress, int(turn.get("turn_number")))
        if label == POSITIVE_LABEL:
            positives.append(track_id)
        elif label == NEGATIVE_LABEL:
            negatives.append(track_id)

    return ConversationState(
        session_id=item["session_id"],
        user_id=item["user_id"],
        turn_number=turn_number,
        session_date=item.get("session_date", ""),
        current_user_query=as_text(current.get("content")),
        user_profile=item.get("user_profile", {}),
        conversation_goal=item.get("conversation_goal", {}),
        history_turns=history,
        progress_by_turn=progress,
        previous_music_track_ids=previous_music,
        positive_seed_ids=positives,
        negative_seed_ids=negatives,
    )


def history_as_text(history_turns: list[dict[str, Any]], catalog: TrackCatalog, max_turns: int = 18) -> str:
    chunks = []
    for turn in history_turns[-max_turns:]:
        role = turn.get("role", "")
        content = as_text(turn.get("content"))
        if role == "music" and catalog.has_track(content):
            content = catalog.compact_summary(content)
            role = "recommended_music"
        thought = as_text(turn.get("thought"))
        if thought and role == "music":
            content = f"{content}. recommendation_reason: {thought}"
        chunks.append(f"{role}: {content}")
    return "\n".join(chunks)


def user_profile_text(profile: dict[str, Any]) -> str:
    fields = ["age_group", "gender", "country_name", "preferred_language", "preferred_musical_culture"]
    parts = [f"{field}: {as_text(profile.get(field))}" for field in fields if as_text(profile.get(field))]
    return "\n".join(parts)


def seed_text(state: ConversationState, catalog: TrackCatalog) -> str:
    parts = []
    if state.positive_seed_ids:
        positive = "\n".join(catalog.compact_summary(track_id) for track_id in state.positive_seed_ids[-4:] if catalog.has_track(track_id))
        parts.append(f"positive previous recommendations:\n{positive}")
    if state.negative_seed_ids:
        negative = "\n".join(catalog.compact_summary(track_id) for track_id in state.negative_seed_ids[-4:] if catalog.has_track(track_id))
        parts.append(f"negative previous recommendations:\n{negative}")
    return "\n".join(part for part in parts if part.strip())


def legacy_history_query(state: ConversationState, catalog: TrackCatalog) -> str:
    lines = []
    for turn in state.history_turns:
        role = turn.get("role", "")
        content = as_text(turn.get("content"))
        if role == "music" and catalog.has_track(content):
            role = "assistant"
            row = catalog.rows[content]
            parts = [f"track_id: {content}"]
            for field in ["track_name", "artist_name", "album_name", "release_date"]:
                value = as_text(row.get(field))
                if value:
                    parts.append(f"{field}: {value}")
            content = ", ".join(parts)
        lines.append(f"{role}: {content}")
    lines.append(f"user: {state.current_user_query}")
    return "\n".join(lines)


def state_text(state: ConversationState, catalog: TrackCatalog) -> str:
    return "\n\n".join(
        part
        for part in [
            f"current_user_query: {state.current_user_query}",
            f"conversation_goal: {state.listener_goal}",
            f"goal_category: {state.category} specificity: {state.specificity}",
            f"user_profile:\n{user_profile_text(state.user_profile)}",
            f"history:\n{history_as_text(state.history_turns, catalog)}",
            seed_text(state, catalog),
        ]
        if part.strip()
    )


def query_variants(state: ConversationState, catalog: TrackCatalog) -> dict[str, str]:
    current = state.current_user_query
    goal = state.listener_goal
    seeds = seed_text(state, catalog)
    quoted = " ".join(re.findall(r"[\"']([^\"']{2,80})[\"']", current + " " + goal))
    variants = {
        "legacy_history": legacy_history_query(state, catalog),
        "current": current,
        "goal": goal,
        "current_goal": f"{current}\n{goal}",
        "seed_current": f"{seeds}\n{current}\n{goal}",
    }
    if quoted:
        variants["quoted_entities"] = quoted
    return {name: text for name, text in variants.items() if normalize_text(text)}
