# Integration review (November 2026)

Each stage of the synopsis pipeline was checked against what the code actually did, by
running the closed loop and auditing its output rather than relying on the unit tests (all
32 original tests passed while most of the problems below were present). This document lists
what was broken, the evidence, the fix, and the test that now guards it.

Test suite: **88 tests pass** (32 original, a few updated where they asserted old internals,
plus `tests/test_integration.py` and `tests/test_overlap_scenarios.py`).

## 1. The LLM was never called

**Evidence.** The committed benchmark reported `LLMCalls=1, LLMTokens=0` at ~2.3 s latency for the
"Proposed Framework". In a diagnostic run all strategist decisions came from `heuristic`, while
`metrics.llm_calls` counted them as LLM calls. The viewer filled missing values with hardcoded
"850 tokens / 1120 ms".

**Causes.**
* No API key resolved; the default Groq model `qwen/qwen3.8-27b` does not appear in Groq's model list.
* `OPENAI_API_KEY` was sent to Groq's base URL.
* An Ollama client was tried on every call, costing ~1.4 s of connection retries, then the
  heuristic answered silently.
* Instructor was a dependency but was never used; the retry call's telemetry was discarded.

**Fix** (`llm/strategist.py`, `llm/schema.py`, `llm/prompts.py`):
* Explicit providers (`groq`, `openai`, `ollama`, `heuristic`, `auto`); Ollama is probed once.
* Missing LLM is an error with `require_llm` / `--require-llm`, otherwise a loud warning.
* Calls go through **Instructor** (JSON mode) with a per-incident **grounded** Pydantic model:
  robot IDs outside `candidate_robots`, or omitting a `must_replan` robot, raise a validation
  error that Instructor sends back to the LLM as a re-ask.
* Per call the replay records provider, model, latency, tokens (summed over re-asks; Instructor
  writes a running total into the last response, so usage is snapshotted per request) and errors.
* Heuristic decisions are labelled and counted separately (`heuristic_decisions`), never as `llm_calls`.

**Tests:** `test_llm_is_really_called_through_instructor`, `test_hallucinated_robot_id_is_rejected_and_reasked`,
`test_llm_failure_is_not_counted_as_llm_call`, `test_heuristic_is_labelled_and_never_counted_as_llm`,
`test_require_llm_fails_loudly_without_provider`. These use `tests/mock_llm.py`, a fake
OpenAI-compatible server plugged into the real `openai` SDK, so the full SDK → Instructor →
Pydantic → controller path runs without network access.

## 2. Robots did not follow a MAPF algorithm

**Evidence.** `World.step()` planned each robot with single-agent A* ignoring all other robots.
One run (60 robots, 200 steps) executed **79 vertex collisions and 35 edge swaps**.

**Fix** (`sim/planning.py`, `sim/lifelong.py`). Rolling-horizon cooperative space-time A*
(windowed cooperative A*, Silver 2005, reference [1]):
* Every robot follows a committed, time-indexed plan that is collision-free with respect to all
  other plans (vertex, swap and parking reservations).
* Batches use prioritized planning with restarts.
* A consistency check runs before every move, and an execution guard counts any residual
  intervention.
* Each robot has its own seeded task stream, so all modes see identical tasks.

**Result.** 0 collisions, 0 swaps, 0 guard interventions across seeds, 100–200 robots and all modes.

**Tests:** `test_robots_execute_their_committed_mapf_plans` (every move equals the next cell of the
committed plan), `test_closed_loop_is_collision_free_in_every_mode`, `test_same_seed_gives_same_tasks_across_modes`.

## 3. Disruption semantics were broken by the repair step

**Evidence.**
* A robot that failed at t=40 was **resurrected**: the failed robot was in the directive, the
  solver gave it a path, and the controller set its state to "replanning".
* **Every delayed robot kept moving** during its delay (the repair overwrote `state`).
* Robot failures and delays used the robot's position from schedule generation (t=0).
* Blocked cells could appear under a robot.

**Fix.**
* Failed robots are never plannable; the controller sanitises every directive.
* A delay prefixes k waits to the plan; all solvers and the validator enforce the wait.
* Robot disruptions resolve the robot and cell when they fire; blockages never land under a
  robot; robots whose goal becomes blocked get a new goal.

**Tests:** `test_failed_robot_is_never_resurrected_by_repair`, `test_delayed_robot_waits_for_k_steps_even_when_repaired`,
`test_validator_rejects_moving_a_delayed_robot`, `test_solvers_respect_delays_and_fleet`.

## 4. Validator only checked repaired robots against each other

**Evidence.** A repaired path could drive into a robot that was not being replanned and still
pass. Checks were indexed one step off (the path excludes the start cell). Affected robots left
out of the directive were never checked.

**Fix** (`validate/validator.py`). Candidate plans are validated against **every** other robot's
committed plan. The checks are: move continuity, obstacles, failed robots, delay waits,
vertex/edge conflicts, parking safety, `required` affected robots that were not replanned, and
deadlock. Violation messages carry absolute times.

**Tests:** `test_validator_checks_against_robots_that_are_not_replanned`,
`test_validator_flags_unrepaired_required_robot`, rewritten `test_validator_catches_all_violations`.

## 5. Solvers

**Problems.**
* Reservations of non-replanned robots were off by one time step (PIBT, LNS).
* PIBT ignored swaps with non-replanned robots.
* "Windowed PBS" only tried the given order and one swapped order.
* LNS used an unseeded RNG.

**Fix.** Faithful PIBT (priority inheritance and backtracking), MAPF-LNS (prioritized-planning
initial solution, destroy/repair neighbourhoods, sum-of-costs acceptance) and PBS (depth-first
search over priority constraints, explores the directive's order first). All share
`sim/planning.py`, use true distances, and use deterministic RNGs that do not consume the world RNG.

## 6. Overlapping (simultaneous / interacting) disruptions were effectively missing

**Evidence** (3 seeds, 100 robots, 300 steps):

| Mode | Overlap pairs | Interacting incidents | Multi-disruption incidents |
| --- | --- | --- | --- |
| `same_aisle` | 40 | 2 | 0 |
| `same_robot` | 38 | 0 | 0 |
| simultaneous | not generated | – | – |

**Causes.**
* Second disruptions landed on unused cells (always a blocked cell for `same_aisle`).
* `same_robot` delayed robots that had already failed.
* Interaction tracking only remembered incidents that produced a repair.

**Fix.**
* New modes `simultaneous`, `cascade` and `mixed`; follow-ups can be any type.
* Follow-ups are resolved when they fire so that they interact with the first disruption
  (affected robots or path regions overlap).
* The sentinel tracks every active disruption and links incidents across time.
* Each incident is labelled `isolated | simultaneous | sequential | compound` with its
  `related_disruptions`, which the LLM prompt also receives.
* Metrics are split per overlap kind.
* A controlled scenario suite (`experiments/scenarios.py`) covers all 16 combinations: each type
  alone, every pair of types simultaneous and sequential, and the three-type compound.

**Result** (same setup): `same_aisle` 16 interacting incidents, `same_robot` 12, `simultaneous`
21 merged incidents, `cascade` 16 compound incidents with all three types. Every scenario of the
suite produces detected overlaps with 0 collisions. At 150 robots sequential pairs reach 13–14/15
detected repetitions; the rest are follow-ups that affected no robot.

**Tests:** `tests/test_overlap_scenarios.py` (generator structure, simultaneous merge, sequential
linking, prompt content, every overlap mode end-to-end, every scenario of the suite, coverage of
all type pairs).

## 7. Strategy cache and negative memory

**Problems.**
* Negative memory compared exact robot-ID lists, so it never matched a new incident (dead code).
* Cosine similarity on mostly one-hot features scored ~0.87 between an isolated single-robot
  incident and an interacting two-robot one, so the 0.70–0.95 threshold sweep could not
  discriminate.
* Incident IDs restart every run, so new runs overwrote earlier cache entries.
* All benchmark conditions shared one on-disk cache, which was also committed to the repo.

**Fix.**
* Strategy signatures `(solver, horizon)` per category; a signature is cleared when it later validates.
* Hard filter now includes isolated/interacting; RBF similarity on continuous incident features.
* Namespaced entry IDs; failing entries are evicted.
* Per-run in-memory cache by default (`cache_persist`); stale cache files removed from git.

**Tests:** `test_cache_does_not_mix_isolated_and_interacting_incidents`, `test_cache_threshold_discriminates`,
`test_negative_memory_uses_strategy_signature_and_clears_on_success`, `test_cache_entries_are_unique_across_runs`,
`test_runs_do_not_touch_repo_cache_by_default`.

## 8. Controller, ablations and metrics

**Problems.**
* `validator_enabled=False` had no effect in LLM mode.
* `cost_delta_pct` was `mean(cost_delta) * 0.8`; `recovery_steps` was the repair-path length.
* Cache insertions were summed cumulatively per incident.
* The exported map listed failed robots and blocked cells as shelves.

**Fix.**
* A real no-validator ablation: unvalidated plans are executed, counted in
  `executed_safety_violations`, and left to the execution guard.
* Measured recovery (steps until net progress resumes, censored) and path quality (detour over
  the single-agent shortest path).
* Throughput, first-attempt success and a post-hoc frame audit of executed collisions.
* Baselines get the same validator and fallback, so comparisons isolate strategy selection.
* A rule-based selector control condition was added.

**Test:** `test_validator_gate_vs_no_validator_ablation`.

## 9. Viewer

**Problems.**
* Hardcoded token/latency fallbacks.
* The incident diagnostics panel had no markup in `index.html`, so all its updates went to `null`.
* No `.hidden` CSS class.
* The Scenario dropdown pointed at files that do not exist.
* The PIBT / LNS / PBS modes all showed the same baseline file.
* The incident close button hid the incident list.
* The default demo replay was synthetic.

**Fix.**
* Added a repair-pipeline card: sentinel with all disruptions and overlap kind, cache, dry run,
  strategist provider/latency/tokens/re-asks, executed solver and validity, validator attempts.
* Per-mode replay files; honest labels for rule-based runs.
* `scripts/make_viewer_replays.py` generates real replays for five scenarios × four modes;
  legacy replays are labelled.
* `tsc` and `vite build` pass. The UI has not been checked in a browser here.

## Still open

* **No real-LLM results yet.** All results in this review were produced with the rule-based
  selector (no API key in the review environment), so the proposed framework and
  `RuleBased_Selector` are identical in them. Run the benchmark and the scenario suite with
  `--provider groq` (or OpenAI / Ollama) before drawing conclusions about LLM-guided selection.
* `viewer/public/replays/*` were generated without an LLM; regenerate them with a provider for demos.
* `CLAUDE.md` asks for `three/webgpu` + `WebGPURenderer`; the viewer uses `WebGLRenderer`. Not changed here.
* Results in `runs/legacy_invalid/` were produced before these fixes and should not be cited.
