#map authors to speaker tags
def _speaker_map(messages: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for m in messages:
        a = m.get("author")
        if not a:
            continue
        if a not in mapping:
            mapping[a] = f"SPK{len(mapping) + 1}"
    return mapping

#convert messages to text
def messages_to_text(messages: list[dict]) -> str:
    speaker_tags = _speaker_map(messages)
    parts: list[str] = []
    for msg in messages:
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        author = msg.get("author")
        spk = speaker_tags.get(author, "SPK?")
        parts.append(f"{spk}: {text}")
    return "\n".join(parts)
