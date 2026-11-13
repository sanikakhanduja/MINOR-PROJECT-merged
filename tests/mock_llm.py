"""A tiny OpenAI-compatible fake LLM used by the tests.

It is plugged into the *real* ``openai`` client through an httpx MockTransport,
so the full path openai SDK -> Instructor -> Pydantic validation -> controller
is exercised without network access. It reads the incident JSON from the
prompt and answers like a sensible strategist would.
"""

import json
import re
from typing import List

import httpx
import openai


class FakeLLM:
    def __init__(self, hallucinate_first: bool = False, solver: str = "windowed_pbs"):
        self.requests: List[dict] = []
        self.hallucinate_first = hallucinate_first
        self.solver = solver

    def _incident(self, body: dict) -> dict:
        user = [m for m in body["messages"] if m["role"] == "user"][0]["content"]
        m = re.search(r"Incident detected:\n(\{.*\})", user, re.S)
        payload = json.loads(m.group(1)) if m else {}
        return payload.get("incident", {})

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        inc = self._incident(body)
        robots = sorted(set(inc.get("must_replan", [])) | {
            r["id"] for r in inc.get("robots", []) if r["role"] != "nearby"}) or inc.get("candidate_robots", [])[:1]
        if self.hallucinate_first and len(self.requests) == 1:
            robots = robots + [99999]  # a robot that does not exist
        directive = {"robots": robots, "solver": self.solver, "priority": robots,
                     "horizon": 25, "rationale": "fake LLM decision"}
        return httpx.Response(200, json={
            "id": f"cmpl-{len(self.requests)}", "object": "chat.completion", "created": 0, "model": "fake",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(directive)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        })

    def client(self) -> openai.OpenAI:
        return openai.OpenAI(api_key="test", base_url="http://fake-llm/v1", max_retries=0,
                             http_client=httpx.Client(transport=httpx.MockTransport(self.handler)))
