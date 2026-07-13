from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

_LINE_RE = re.compile(r"^(?P<spk>SPK\d+):\s*(?P<msg>.*)$")
_YOU_RE = re.compile(r"\b(you|your|yours|u|ur)\b", re.IGNORECASE)


@dataclass(frozen=True)
class MetadataFeatures:
    DIM: int = 7

    @staticmethod
    def from_text(text: str, *, progress: float | None = None) -> np.ndarray:
        progress_v = 1.0 if progress is None else float(progress)
        progress_v = float(np.clip(progress_v, 0.0, 1.0))

        speaker_counts: dict[str, int] = {}
        msg_lengths: list[int] = []
        num_questions = 0
        num_personal_questions = 0
        num_messages = 0

        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = _LINE_RE.match(line)
            if not m:
                spk = "SPK?"
                msg = line
            else:
                spk = m.group("spk")
                msg = (m.group("msg") or "").strip()

            if not msg:
                continue

            num_messages += 1
            speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
            msg_lengths.append(len(msg))

            if "?" in msg:
                num_questions += 1
                if _YOU_RE.search(msg):
                    num_personal_questions += 1

        avg_len = float(sum(msg_lengths) / max(len(msg_lengths), 1))
        unique_speakers = len(speaker_counts)
        most_active = max(speaker_counts.values()) if speaker_counts else 0
        ratio_most_active = float(most_active / max(num_messages, 1))

        return np.array(
            [
                math.log1p(num_messages),
                math.log1p(avg_len),
                math.log1p(num_questions),
                math.log1p(num_personal_questions),
                progress_v,
                math.log1p(unique_speakers),
                ratio_most_active,
            ],
            dtype=np.float32,
        )


def batch_metadata_features(
    texts: list[str], *, progress: list[float] | None = None
) -> np.ndarray:
    if progress is None:
        return np.stack(
            [MetadataFeatures.from_text(t, progress=None) for t in texts], axis=0
        )
    if len(progress) != len(texts):
        raise ValueError("progress must be same length as texts")
    return np.stack(
        [MetadataFeatures.from_text(t, progress=p) for t, p in zip(texts, progress)],
        axis=0,
    )
