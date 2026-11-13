"""Replay file exporter conforming to the benchmark schema."""

import os
import gzip
import json
from typing import Optional
from amr_repair.replay.models import Replay


def export_replay(replay: Replay, filepath: str, indent: Optional[int] = None) -> None:
    """Exports a Replay object to JSON (compact by default) or gzip-compressed JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    json_str = replay.model_dump_json(indent=indent)

    if filepath.endswith(".gz"):
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            f.write(json_str)
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_str)
