"""One JSONL trace file per run, one JSON object per line (event).

Events, in order: run_start, then llm_call / tool_call / llm_error as they
happen, then run_end. Nothing secret is ever passed in: the loop has no access
to API keys, and provider-native message content is not written.
"""

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from agent.llm.base import Message


def message_record(m: Message) -> dict:
    """The traced form of a message: role, text, tool calls and tool results."""
    record = {"role": m.role, "text": m.text}
    if m.tool_calls:
        record["tool_calls"] = [asdict(c) for c in m.tool_calls]
    if m.tool_results:
        record["tool_results"] = [asdict(r) for r in m.tool_results]
    return record


class Tracer:
    def __init__(self, traces_dir: str | Path):
        traces_dir = Path(traces_dir)
        traces_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex[:12]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = traces_dir / f"{stamp}_{self.run_id}.jsonl"
        self._file = open(self.path, "w", encoding="utf-8")

    def event(self, kind: str, **fields) -> None:
        record = {"event": kind, "run_id": self.run_id, "timestamp": datetime.now(timezone.utc).isoformat(), **fields}
        self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
