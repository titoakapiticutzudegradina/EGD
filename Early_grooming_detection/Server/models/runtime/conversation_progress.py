from __future__ import annotations

MESSAGES_AT_10_PERCENT = 50
PROGRESS_CHECKPOINTS: tuple[float, ...] = (0.1, 0.2, 0.4, 0.8, 1.0)


def full_conversation_messages() -> int:
    return MESSAGES_AT_10_PERCENT * 10


def messages_at_checkpoint(checkpoint: float) -> int:
    return int(round(MESSAGES_AT_10_PERCENT * 10 * checkpoint))


def progress_from_message_count(message_count: int) -> float:
    if message_count <= 0:
        return 0.0
    total = full_conversation_messages()
    return min(1.0, message_count / total)


def conversation_progress(block_message_counts: list[int]) -> float:
    cumulative = sum(block_message_counts)
    return progress_from_message_count(cumulative)


def resolve_conversation_progress(
    message_count: int,
    block_message_counts: list[int] | None = None,
) -> float:
    if block_message_counts:
        cumulative = sum(block_message_counts)
        if cumulative > 0:
            return progress_from_message_count(cumulative)
    return progress_from_message_count(message_count)
