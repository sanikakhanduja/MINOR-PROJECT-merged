"""Recorded LLM responses, so an LLM run can be repeated without calling the API.

Each LLM decision is stored under a key made from everything that determines
the request: provider, model, temperature, system prompt and user prompt.
Running the same seed again with the same store returns the recorded
directive, token counts, latency and returned model version, so the run
reproduces exactly and costs nothing.

Modes
  ``record``  call the API on a miss and store the answer; reuse stored answers
  ``replay``  never call the API; a missing answer is an error

The store is a JSON-lines file (one response per line), appended to as the
run goes, so a crash keeps everything recorded so far.
"""

import hashlib
import json
import os
from typing import Any, Dict, Optional

MODES = ("record", "replay")


class ResponseNotRecorded(RuntimeError):
    pass


def request_key(provider: str, model: Optional[str], temperature: float, system: str, user: str) -> str:
    blob = json.dumps([provider, model, round(float(temperature), 4), system, user], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseStore:
    def __init__(self, path: str, mode: str = "record"):
        if mode not in MODES:
            raise ValueError(f"Unknown response-store mode '{mode}'. Choose from {MODES}.")
        self.path = path
        self.mode = mode
        self.records: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self.records[rec["key"]] = rec

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        rec = self.records.get(key)
        if rec is None:
            self.misses += 1
            if self.mode == "replay":
                raise ResponseNotRecorded(f"No recorded LLM response for request {key[:12]} in {self.path}")
            return None
        self.hits += 1
        return rec

    def put(self, key: str, record: Dict[str, Any]) -> None:
        rec = {"key": key, **record}
        self.records[key] = rec
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
