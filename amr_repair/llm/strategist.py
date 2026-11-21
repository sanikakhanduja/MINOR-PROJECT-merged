"""LLM strategist: IncidentRecord -> Directive, via Instructor + Pydantic.

Providers (all OpenAI-compatible):

* ``gemini``    GEMINI_API_KEY, GEMINI_MODEL (default gemini-2.5-flash),    GEMINI_BASE_URL
* ``groq``      GROQ_API_KEY,  GROQ_MODEL   (default openai/gpt-oss-20b),  GROQ_BASE_URL
* ``openai``    OPENAI_API_KEY, OPENAI_MODEL (default gpt-4o-mini),        OPENAI_BASE_URL
* ``ollama``    local server,  OLLAMA_MODEL (default llama3.2),            OLLAMA_BASE_URL
* ``heuristic`` deterministic rules, **no LLM** (logged as such, never counted as an LLM call)
* ``auto``      first available of gemini -> groq -> openai -> ollama; if none is available this is an
                error unless ``allow_heuristic=True`` (then a loud warning is logged).

Select with ``Strategist(provider=...)`` or the ``LLM_PROVIDER`` environment variable.
Every call records which provider/model actually answered, latency and token
usage (summed over Instructor's schema re-asks) in ``self.last_call``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from amr_repair.llm.prompts import SYSTEM_PROMPT, format_user_prompt
from amr_repair.llm.response_store import ResponseNotRecorded, ResponseStore, request_key
from amr_repair.llm.schema import Directive, grounded_directive
from amr_repair.sentinel.incident import IncidentRecord

logger = logging.getLogger(__name__)

PROVIDERS = ("auto", "gemini", "groq", "openai", "ollama", "heuristic", "custom")
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-20b",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2",
}
DEFAULT_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434/v1",
}


class LLMUnavailableError(RuntimeError):
    pass


_ENV_LOADED = False


def _load_env_file() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
    if not os.path.exists(env_file):
        return
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
    except OSError as e:  # pragma: no cover
        logger.debug("Could not load .env: %s", e)


@dataclass
class LLMCall:
    provider: str
    model: Optional[str]
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_requests: int = 0       # HTTP requests incl. Instructor re-asks
    error: Optional[str] = None
    directive: Optional[Directive] = None
    returned_model: Optional[str] = None  # model version the API reported in its response
    replayed: bool = False                # answer came from the response store, no API call

    @property
    def is_llm(self) -> bool:
        return self.provider not in ("heuristic", "none") and self.error is None


class Strategist:
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        client: Any = None,
        force_heuristic: bool = False,
        allow_heuristic: bool = True,
        timeout_s: float = 30.0,
        schema_retries: int = 1,
        max_tokens: int = 4096,
        response_store: Optional[ResponseStore] = None,
    ):
        _load_env_file()
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.schema_retries = schema_retries
        self.max_tokens = max_tokens
        self.allow_heuristic = allow_heuristic
        self.response_store = response_store
        self.replayed_calls = 0
        self.client = None
        self.model: Optional[str] = None
        self._instructor = None
        self._usage: List[Any] = []

        self.last_call: LLMCall = LLMCall(provider="none", model=None)
        self.total_calls = 0          # real LLM decisions
        self.total_api_requests = 0   # HTTP requests incl. re-asks
        self.total_tokens = 0
        self.heuristic_decisions = 0
        self.llm_errors = 0

        requested = "heuristic" if force_heuristic else (provider or os.environ.get("LLM_PROVIDER", "auto")).lower()
        if requested not in PROVIDERS:
            raise ValueError(f"Unknown LLM provider '{requested}'. Choose from {PROVIDERS}.")

        if client is not None:
            self.provider = "custom"
            self.client = client
            self.model = model or "custom"
        elif response_store is not None and response_store.mode == "replay" and requested not in ("auto", "heuristic"):
            # Replaying recorded answers: no API key or network needed
            self.provider = requested
            self.model = model or os.environ.get(f"{requested.upper()}_MODEL") or DEFAULT_MODELS.get(requested)
        else:
            self.provider = self._resolve(requested, model)

        if self.client is not None:
            self._setup_instructor()
        logger.info("Strategist provider=%s model=%s", self.provider, self.model)

    # ---------------------------------------------------------------- setup
    def _resolve(self, requested: str, model: Optional[str]) -> str:
        if requested == "heuristic":
            return "heuristic"
        order = ["gemini", "groq", "openai", "ollama"] if requested == "auto" else [requested]
        problems = []
        for prov in order:
            ok, why = self._try_provider(prov, model)
            if ok:
                return prov
            problems.append(f"{prov}: {why}")
        msg = "No LLM provider available (" + "; ".join(problems) + ")."
        if requested != "auto" or not self.allow_heuristic:
            raise LLMUnavailableError(
                msg + " Set GEMINI_API_KEY / GROQ_API_KEY / OPENAI_API_KEY or start Ollama, or use provider='heuristic'."
            )
        logger.warning("%s Falling back to the deterministic HEURISTIC strategist: "
                       "no LLM will be called and results are not LLM-guided.", msg)
        return "heuristic"

    def _try_provider(self, prov: str, model: Optional[str]):
        from openai import OpenAI
        base = os.environ.get(f"{prov.upper()}_BASE_URL", DEFAULT_URLS[prov])
        mdl = model or os.environ.get(f"{prov.upper()}_MODEL") or DEFAULT_MODELS[prov]
        if prov in ("gemini", "groq", "openai"):
            key = os.environ.get(f"{prov.upper()}_API_KEY")
            if not key:
                return False, f"{prov.upper()}_API_KEY not set"
            self.client = OpenAI(api_key=key, base_url=base, timeout=self.timeout_s, max_retries=1)
            self.model = mdl
            return True, ""
        # ollama: probe once so we do not pay a connection timeout on every incident
        try:
            import httpx
            r = httpx.get(base.rstrip("/") + "/models", timeout=1.5)
            if r.status_code != 200:
                return False, f"server at {base} answered HTTP {r.status_code}"
        except Exception as e:
            return False, f"not reachable at {base} ({type(e).__name__})"
        self.client = OpenAI(api_key="ollama", base_url=base, timeout=max(self.timeout_s, 60.0), max_retries=0)
        self.model = mdl
        return True, ""

    def _setup_instructor(self) -> None:
        import instructor
        self._instructor = instructor.from_openai(self.client, mode=instructor.Mode.JSON)
        self._instructor.on("completion:response", self._record_usage)

    def _record_usage(self, resp: Any) -> None:
        """Snapshot token counts per HTTP response. (Instructor later writes the
        running total into the final response's usage object, so references
        must not be kept.)"""
        u = getattr(resp, "usage", None)
        self._usage.append(None if u is None else (
            int(getattr(u, "prompt_tokens", 0) or 0),
            int(getattr(u, "completion_tokens", 0) or 0),
            int(getattr(u, "total_tokens", 0) or 0),
        ))

    @property
    def uses_llm(self) -> bool:
        return self.provider != "heuristic"

    # Back-compat accessors used by older code/tests
    @property
    def last_provider(self) -> str:
        return self.last_call.provider

    @property
    def last_latency_ms(self) -> float:
        return self.last_call.latency_ms

    @property
    def last_tokens(self) -> int:
        return self.last_call.total_tokens

    # ---------------------------------------------------------------- choose
    def choose(
        self,
        incident: IncidentRecord,
        violation: Optional[str] = None,
        rejected: Optional[List[dict]] = None,
        previous: Optional[Directive] = None,
    ) -> Directive:
        start = time.perf_counter()
        replaying = self.response_store is not None and self.response_store.mode == "replay"
        if self.provider != "heuristic" and (replaying or (self._instructor is not None and self.client is not None)):
            self._usage = []
            call = LLMCall(provider=self.provider, model=self.model)
            try:
                directive = self._ask_llm(incident, violation, rejected, previous, call)
                call.directive = directive
                self.total_calls += 1
            except ResponseNotRecorded:
                raise  # replaying a run must never silently change a decision
            except Exception as e:  # network error, invalid output after re-asks, ...
                self.llm_errors += 1
                call.error = f"{type(e).__name__}: {str(e)[:200]}"
                logger.warning("LLM (%s/%s) failed for %s: %s -> using heuristic for this decision",
                               self.provider, self.model, incident.id, call.error)
                directive = None
            if not call.replayed:
                usage = [u for u in self._usage if u is not None]
                call.api_requests = len(self._usage)
                call.prompt_tokens = sum(u[0] for u in usage)
                call.completion_tokens = sum(u[1] for u in usage)
                call.total_tokens = sum(u[2] for u in usage)
            self.total_api_requests += call.api_requests
            self.total_tokens += call.total_tokens
            if directive is not None:
                if not call.replayed:
                    call.latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
                    self._record(call)
                self.last_call = call
                return directive
            # Per-decision degradation: record the failed LLM attempt, decide heuristically
            directive = self._heuristic_choice(incident, violation)
            call.latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
            call.directive = directive
            call.provider = "heuristic"
            self.heuristic_decisions += 1
            self.last_call = call
            return directive

        directive = self._heuristic_choice(incident, violation)
        self.heuristic_decisions += 1
        self.last_call = LLMCall(provider="heuristic", model=None,
                                 latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
                                 directive=directive)
        return directive

    def _ask_llm(self, incident, violation, rejected, previous, call: LLMCall) -> Directive:
        response_model = grounded_directive(
            incident.candidate_robots or incident.plannable_affected(), incident.must_replan)
        user_msg = format_user_prompt(
            incident, violation=violation, rejected=rejected,
            previous_directive=previous.model_dump() if previous is not None else None)
        self._pending_key = None
        if self.response_store is not None:
            key = request_key(self.provider, self.model, self.temperature, SYSTEM_PROMPT, user_msg)
            rec = self.response_store.get(key)  # raises in replay mode when missing
            if rec is not None:
                # The recorded answer, with the latency and tokens it cost when it was recorded
                call.replayed = True
                call.latency_ms = float(rec["latency_ms"])
                call.prompt_tokens = int(rec.get("prompt_tokens", 0))
                call.completion_tokens = int(rec.get("completion_tokens", 0))
                call.total_tokens = int(rec.get("total_tokens", 0))
                call.api_requests = int(rec.get("api_requests", 1))
                call.returned_model = rec.get("returned_model")
                self.replayed_calls += 1
                return Directive.model_validate(rec["directive"])
            self._pending_key = key
        result = self._instructor.chat.completions.create_with_completion(
            model=self.model,
            response_model=response_model,
            max_retries=self.schema_retries,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_msg}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        parsed, completion = (result[0], result[1]) if isinstance(result, tuple) else (result, None)
        call.returned_model = getattr(completion, "model", None)
        return Directive.model_validate(parsed.model_dump())

    def _record(self, call: LLMCall) -> None:
        key = getattr(self, "_pending_key", None)
        if self.response_store is None or key is None or call.directive is None:
            return
        self.response_store.put(key, {
            "provider": call.provider, "model": call.model, "returned_model": call.returned_model,
            "temperature": self.temperature, "directive": call.directive.model_dump(),
            "latency_ms": call.latency_ms, "prompt_tokens": call.prompt_tokens,
            "completion_tokens": call.completion_tokens, "total_tokens": call.total_tokens,
            "api_requests": call.api_requests,
        })
        self._pending_key = None

    # ------------------------------------------------------------ heuristic
    def _heuristic_choice(self, incident: IncidentRecord, violation: Optional[str] = None) -> Directive:
        """Deterministic, rule-based strategy (NOT an LLM): used for the
        heuristic ablation and as a per-decision safety net."""
        robots = sorted(set(incident.plannable_affected()) | set(incident.must_replan))
        if not robots:
            robots = list(incident.candidate_robots[:1]) or list(incident.affected_robots[:1]) or [0]
        if violation:
            return Directive(robots=robots, solver="windowed_pbs", priority=list(reversed(robots)), horizon=30,
                             rationale=f"Retry after '{violation[:60]}': PBS with longer horizon.")
        if incident.interacts_with or len(robots) >= 4:
            return Directive(robots=robots, solver="windowed_pbs", priority=robots, horizon=25,
                             rationale="Interacting / multi-robot disruption: windowed PBS.")
        if incident.congestion.level == "high" or len(robots) >= 2:
            return Directive(robots=robots, solver="lns", priority=robots, horizon=20,
                             rationale="Several robots or congested area: LNS.")
        return Directive(robots=robots, solver="pibt", priority=robots, horizon=15,
                         rationale="Single isolated disruption: PIBT for minimal latency.")
