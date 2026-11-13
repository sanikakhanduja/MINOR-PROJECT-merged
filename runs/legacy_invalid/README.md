# Legacy results — do not cite

These files were produced by the pipeline **before** the integration fixes
(see `INTEGRATION_REVIEW.md` in the repository root). They are kept only for
reference and are **not valid research results**:

* No LLM was ever called: every "LLM" decision was the deterministic heuristic
  (`LLMTokens = 0`, ~2 s latency = the Ollama connection timing out), yet it was
  counted as an LLM call.
* The simulator let robots walk independent A* paths, so executed trajectories
  contained vertex collisions and edge swaps.
* Failed robots could be "resurrected" by a repair and delays had no effect.
* `cost_delta_pct` was `mean(cost_delta) * 0.8` and `recovery_steps` was the
  repair-path length, not measured quantities.
* All benchmark conditions shared one persistent cache, so results depended on
  run order and on earlier sessions.

Regenerate results with `python -m amr_repair.experiments.benchmark` and
`python -m amr_repair.experiments.scenarios` (see the main README).
