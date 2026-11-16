# How to Open & Run This Project

## Project Overview

**Lifelong MAPF Simulator with LLM-Guided Path Repair & 3D Replay Viewer**

This project has two main components:

| Component | Tech Stack | Purpose |
|-----------|-----------|---------|
| **Python Pipeline** | Python 3.11, NumPy, NetworkX, OpenAI, Pydantic | MAPF simulation, disruption injection, LLM-guided repair, benchmarks |
| **3D Replay Viewer** | TypeScript, Three.js, Vite | Interactive 3D warehouse visualization of replay JSON files |

---

## Prerequisites

| Tool | Required Version | Check Command |
|------|-----------------|---------------|
| **Python** | ≥ 3.11 | `python3 --version` |
| **Node.js** | ≥ 18.x | `node --version` |
| **npm** | ≥ 9.x | `npm --version` |
| **uv** (Python package manager) | Latest | `uv --version` |

### Install `uv` (if not already installed)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip3 install uv
```

---

## Step 1: Clone / Open the Project

```bash
cd /path/to/MINOR-PROJECT-main
```

---

## Step 2: Install Python Dependencies

From the project root directory:

```bash
uv sync
```

This will:
- Create a `.venv` virtual environment with Python 3.11
- Install all dependencies listed in `pyproject.toml` (NumPy, NetworkX, OpenAI, Pydantic, scikit-learn, Trimesh, Instructor, etc.)

---

## Step 3: Run the Python Tests

Verify everything is working:

```bash
uv run pytest -q
```

Expected output: **all tests pass** (91 at the time of writing, ~2 minutes). ✅

---

## Step 3b: Configure the LLM

```bash
cp .env.example .env     # then fill in ONE provider (Groq, OpenAI or a local Ollama)
```

`.env` is git-ignored: never commit it. Without a provider the pipeline still runs, but every
decision is made by the **rule-based selector** and is labelled `llm_provider = "heuristic"`
(no LLM calls are counted). Use `--require-llm` for experiments so a missing key is an error.

---

## Step 4: Install Viewer Dependencies

```bash
cd viewer
npm install
```

---

## Step 5: Start the 3D Replay Viewer

```bash
cd viewer
npm run dev
```

This starts a Vite dev server. Open your browser to:

- **3D Digital Twin Viewer**: `http://localhost:5173/`
- **Research & Benchmark Dashboard**: `http://localhost:5173/dashboard/` (or click the "Analytics Dashboard" button in the top navigation bar)

The 3D viewer includes:
- Animated AMR (Autonomous Mobile Robot) fleet
- Interactive timeline with incident markers
- Incident diagnostics panel (Sentinel → Cache → Dry-Run → LLM → Solver → Validator)
- Camera presets and playback controls

The Analytics Dashboard includes:
- Headline KPIs (Repair Success Rate, Cache Hit Rate, Recovery Time, Throughput)
- Core Comparison: LLM-Guided vs Fixed Strategies (PIBT, LNS, Windowed PBS, Random, Oracle)
- Cost & Compute Analysis (Token savings, Decision Latency breakdown, Deterministic Validator Audit)
- Incident Audit History Table with direct "View 3D" links to jump to any incident in the 3D twin

---

## Quick Start (All-in-One)

Run everything from the project root in two terminal tabs:

**Terminal 1 — Python pipeline (tests / experiments):**
```bash
cd /path/to/MINOR-PROJECT-main
uv sync
uv run pytest -q
```

**Terminal 2 — 3D Viewer:**
```bash
cd /path/to/MINOR-PROJECT-main/viewer
npm install
npm run dev
```

Then open **http://localhost:5173/** in your browser.

---

## Running Experiments & Simulations

### One run

```bash
uv run python -m amr_repair.experiments.run --provider openai --require-llm --robots 100 --steps 300 --overlap mixed
```

### Benchmark: proposed framework vs fixed solvers, rule-based selector and ablations

```bash
uv run python -m amr_repair.experiments.benchmark --provider openai --seeds 1 2 3
# quick smoke test without an LLM:
uv run python -m amr_repair.experiments.benchmark --quick --allow-heuristic
```

### Controlled overlapping-disruption scenarios (all 16 combinations)

```bash
uv run python -m amr_repair.experiments.scenarios --provider openai --seeds 1 2 3
```

### Regenerate the viewer's replays from the pipeline

```bash
uv run python scripts/make_viewer_replays.py --provider openai
```

> [!NOTE]
> Provider settings come from `.env` (see `.env.example`): `GROQ_API_KEY` / `OPENAI_API_KEY` /
> `OLLAMA_BASE_URL`, plus optional `*_MODEL`. `--provider heuristic` (or `--allow-heuristic`)
> runs without an LLM; those results are labelled rule-based everywhere.

### Generated output
All simulation runs are saved as JSON replay files in the `runs/` directory, following the replay schema documented in [README.md](README.md). Benchmarks write `runs/benchmark_summary.{csv,md,tex}` and `runs/benchmark_by_overlap_kind.csv`; scenarios write `runs/scenarios/scenario_summary.csv`. Results produced before the integration fixes are in `runs/legacy_invalid/` and must not be cited.

---

## Keyboard Shortcuts (3D Viewer)

| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `←` / `→` | Step backward / forward |
| `1` | Full warehouse camera |
| `2` | Incident camera |
| `3` | Robot follow camera |
| `N` / `P` | Next / Previous incident |
| `F` | Toggle fullscreen |
| `R` | Reset camera |
| `Esc` | Close panels |

---

## Project Structure

```
├── amr_repair/           # Python simulation & repair pipeline
│   ├── sim/              # Lifelong MAPF simulator (grid, agents)
│   ├── disrupt/          # Disruption injector
│   ├── sentinel/         # Incident detection & feature extraction
│   ├── cache/            # Strategy cache & negative memory
│   ├── llm/              # LLM strategist (Pydantic + Instructor)
│   ├── solvers/          # PIBT, LNS, Windowed PBS solvers
│   ├── validate/         # Deterministic plan validator
│   ├── loop/             # Closed-loop controller & logger
│   ├── experiments/      # Baselines, ablations, experiment runner
│   └── replay/           # Replay JSON exporter
├── viewer/               # 3D Three.js Replay Viewer (Vite + TypeScript)
│   ├── src/main.ts       # Main viewer application
│   ├── public/replays/   # Replay JSON files served to the viewer
│   └── public/models/    # 3D model assets (.glb)
├── tests/                # Pytest suite (incl. a fake OpenAI-compatible LLM for end-to-end tests)
├── maps/                 # MovingAI warehouse .map files
├── runs/                 # Simulation output replays
├── pyproject.toml        # Python project config & dependencies
└── uv.lock               # Locked Python dependency versions
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `uv: command not found` | Install uv: `pip3 install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Python version mismatch | `uv sync` auto-downloads Python 3.11 — no manual install needed |
| `npm install` fails | Ensure Node.js ≥ 18: `node --version` |
| Viewer shows black screen | Check browser console for errors; ensure `viewer/public/replays/` has `.json` files |
| OpenAI API errors | Put the key in `.env` (see `.env.example`); check `llm_provider` / `llm_errors` in the run output |
| Viewer: "Replay not available" | Run `uv run python scripts/make_viewer_replays.py` to generate the scenario/mode files |

