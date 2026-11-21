import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';
import type { Incident, Replay } from './types';
import { gridFromObstacles, parseMovingAiMap, type Grid } from './world/grid';
import { createFloor } from './world/floor';
import { createRacks } from './world/racks';
import { RobotFleet } from './world/robots';
import { DisruptionLayer } from './world/disruptions';
import { IncidentOverlay } from './world/incidents';
import { environmentBrightness } from './world/focus';
import { applySky, createPipeline, createRenderer, createSun, MAX_PIXEL_RATIO } from './render/stage';

const STEPS_PER_SECOND = 3.5;
const DASH_TEXT = '—';
const OVERVIEW_PITCH = THREE.MathUtils.degToRad(58);
const OVERVIEW_YAW = THREE.MathUtils.degToRad(-14);
const INCIDENT_PITCH = THREE.MathUtils.degToRad(52);
const FOCUS_DIM = 0.45; // environment brightness while an incident is selected
const DISRUPTION_LABELS = { blocked_cell: 'Aisle blockage', robot_failure: 'Robot failure', robot_delay: 'Robot delay' } as const;

// --- App State ---
class ViewerApp {
  private container!: HTMLElement;
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGPURenderer;
  private pipeline!: THREE.RenderPipeline;
  private controls!: OrbitControls;


  private baselineReplay: Replay | null = null;
  private activeReplay: Replay | null = null;
  private isBaselineMode = false;
  private currentScenario = 'overlap_mixed';
  private currentMode = 'llm';
  private readonly replayCache = new Map<string, Replay>();

  // World (rebuilt only when the replay names a different map)
  private mapWidth = 161;
  private mapHeight = 63;
  private worldMapName: string | null = null;
  private world: THREE.Group | null = null;
  private concreteTexture: THREE.Texture | null = null;
  private readonly fleet = new RobotFleet();
  private readonly disruptions = new DisruptionLayer();
  private readonly incidentOverlay = new IncidentOverlay();
  // Animation & Clock
  private clockTime = 0.0;
  private isPlaying = true;
  private playSpeed = 1.0;
  private lastTimestamp = 0;
  private selectedIncident: Incident | null = null;
  private selectedRobotId: number | null = null;
  private affectedFor: Incident | null = null;
  private affectedSet: Set<number> | null = null;

  // Camera animation
  private activePreset = 1;
  private isCameraTransitioning = false;
  private cameraStartPos = new THREE.Vector3();
  private cameraStartTarget = new THREE.Vector3();
  private cameraEndPos = new THREE.Vector3();
  private cameraEndTarget = new THREE.Vector3();
  private cameraAnimProgress = 1.0;
  private cameraAnimDuration = 1.0;
  private keysPressed: Record<string, boolean> = {};
  private readonly followTarget = new THREE.Vector3();
  private readonly calloutAnchor = new THREE.Vector3();
  private calloutRobotKey = '';

  // Performance tracking
  private frameCount = 0;
  private lastFpsUpdate = 0;
  private fpsDisplay: HTMLElement | null = null;

  async init() {
    this.container = document.getElementById('canvas-container')!;
    this.fpsDisplay = document.getElementById('fps-counter');
    const scenarioSelect = document.getElementById('scenario-select') as HTMLSelectElement | null;
    // Deep link from the dashboard: /?experiment=<replay key>&step=<t>
    const params = new URLSearchParams(window.location.search);
    const linked = params.get('experiment');
    const linkedStep = Number(params.get('step'));
    if (scenarioSelect && linked && Array.from(scenarioSelect.options).some(o => o.value === linked)) {
      scenarioSelect.value = linked;
    }
    if (scenarioSelect?.value) this.currentScenario = scenarioSelect.value;
    if (Number.isFinite(linkedStep) && linkedStep > 0) this.clockTime = linkedStep;
    const modeSelect = document.getElementById('mode-select') as HTMLSelectElement | null;
    if (modeSelect?.value) this.currentMode = modeSelect.value;

    this.scene = new THREE.Scene();
    applySky(this.scene);
    this.scene.add(this.fleet.group, this.disruptions.group, this.incidentOverlay.group);

    this.renderer = await createRenderer(this.container);
    this.setupCamera();
    this.pipeline = createPipeline(this.renderer, this.scene, this.camera);

    await this.loadEnvironment();
    await this.loadScenario(this.currentScenario, this.currentMode);

    this.setupUI();
    this.setupEventListeners();

    new ResizeObserver(() => this.onWindowResize()).observe(this.container);
    this.startIntroDolly();
    this.renderer.setAnimationLoop(this.renderLoop.bind(this));
  }

  private setupCamera() {
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;
    this.camera = new THREE.PerspectiveCamera(45, w / h, 1.0, 1500);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    this.controls.rotateSpeed = 0.85;
    this.controls.zoomSpeed = 1.2;
    this.controls.panSpeed = 1.1;
    this.controls.minDistance = 3;
    this.controls.maxDistance = 500;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.05;

    this.overviewPose(this.camera.position, this.controls.target);
    this.controls.update();
  }

  private async loadEnvironment() {
    const [hdr, concrete] = await Promise.all([
      new HDRLoader().loadAsync('/textures/warehouse.hdr').catch((e) => { console.warn('Could not load warehouse.hdr', e); return null; }),
      new THREE.TextureLoader().loadAsync('/textures/concrete_floor.jpg').catch(() => null),
    ]);
    if (hdr) {
      hdr.mapping = THREE.EquirectangularReflectionMapping;
      this.scene.environment = hdr;
    }
    this.concreteTexture = concrete;
  }

  /** Build floor, racks, walls and sun for the replay's map (from /maps/<map_name>.map). */
  private async ensureWorld(replay: Replay) {
    const mapName = replay.meta.map_name;
    if (this.worldMapName === mapName) return;
    let grid: Grid;
    try {
      const res = await fetch(`/maps/${mapName}.map`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      grid = parseMovingAiMap(await res.text());
    } catch (e) {
      console.warn(`/maps/${mapName}.map unavailable, building the layout from replay.map instead`, e);
      grid = gridFromObstacles(replay.map.width, replay.map.height, replay.map.obstacles);
    }

    if (this.world) this.scene.remove(this.world);
    this.mapWidth = grid.width;
    this.mapHeight = grid.height;
    const sun = createSun(grid.width, grid.height);
    this.world = new THREE.Group();
    this.world.add(createFloor(grid, this.concreteTexture), createRacks(grid), sun, sun.target);
    this.scene.add(this.world);
    this.worldMapName = mapName;
  }

  private async fetchReplay(url: string): Promise<Replay> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    return res.json();
  }

  // Replay file per scenario and mode: llm -> <key>.json, baseline_x -> <key>_baseline_x.json
  // (legacy files only have <key>_baseline.json, which is PIBT).
  private replayUrls(key: string, mode: string): string[] {
    if (mode === 'llm') return [`/replays/${key}.json`];
    const urls = [`/replays/${key}_${mode}.json`];
    if (mode === 'baseline_pibt') urls.push(`/replays/${key}_baseline.json`);
    return urls;
  }

  private async fetchModeReplay(key: string, mode: string): Promise<Replay | null> {
    const cacheKey = `${key}|${mode}`;
    const hit = this.replayCache.get(cacheKey);
    if (hit) return hit;
    for (const url of this.replayUrls(key, mode)) {
      try {
        const replay = await this.fetchReplay(url);
        this.replayCache.set(cacheKey, replay);
        return replay;
      } catch {
        // try the next candidate file
      }
    }
    return null;
  }

  private async loadScenario(scenarioKey: string, mode: string = this.currentMode) {
    try {
      const main = await this.fetchModeReplay(scenarioKey, 'llm');
      const active = mode === 'llm' ? main : await this.fetchModeReplay(scenarioKey, mode);
      if (!main || !active) {
        const missing = !main ? `${scenarioKey}.json` : this.replayUrls(scenarioKey, mode)[0].replace('/replays/', '');
        this.showToast('Replay not available',
          `public/replays/${missing} is missing. Generate it with: python scripts/make_viewer_replays.py`);
        return;
      }
      const base = mode === 'llm' ? await this.fetchModeReplay(scenarioKey, 'baseline_pibt') : active;
      await this.ensureWorld(active);
      const scenarioChanged = scenarioKey !== this.currentScenario || !this.activeReplay;
      this.currentScenario = scenarioKey;
      this.currentMode = mode;
      this.isBaselineMode = mode !== 'llm';
      this.baselineReplay = base;
      this.activeReplay = active;
      this.disruptions.setReplay(main);
      if (base) this.disruptions.setReplay(base);
      this.disruptions.setReplay(active);

      if (scenarioChanged) {
        this.selectedIncident = null;
        this.selectedRobotId = null;
        document.getElementById('incident-panel')?.classList.add('hidden');
      }
      this.updatePathsVisualization();
      this.clockTime = Math.min(this.clockTime, this.activeReplay.frames.length - 1);

      const slider = document.getElementById('step-slider') as HTMLInputElement;
      if (slider) slider.max = (this.activeReplay.frames.length - 1).toString();

      this.updateMetricsUI();
      this.buildTimelineTicks();
      this.reselectIncident();
      const prov = active.meta.llm_provider;
      const provStr = mode === 'llm'
        ? (prov === 'heuristic' ? ' • rule-based, NO LLM' : prov ? ` • LLM: ${prov}` : '')
        : '';
      this.showToast('Scenario Loaded', `${scenarioKey} / ${mode} (${active.incidents.length} incidents${provStr})`);
    } catch (e) {
      console.error('Failed to load scenario:', e);
      this.showToast('Scenario not available', `${scenarioKey}: replay file could not be loaded`);
    }
  }

  private updateMetricsUI() {
    if (!this.activeReplay) return;
    const m = this.activeReplay.metrics;
    const bm = this.baselineReplay ? this.baselineReplay.metrics : null;

    const elSuccess = document.getElementById('metric-success') || document.getElementById('metric-success-rate') || document.getElementById('chip-success-rate');
    const elRecovery = document.getElementById('metric-recovery') || document.getElementById('metric-recovery-steps');
    const elCost = document.getElementById('metric-cost');
    const elLlm = document.getElementById('metric-llm') || document.getElementById('metric-llm-calls');

    const baseSuccessStr = bm ? `vs ${(bm.repair_success_rate * 100).toFixed(1)}% base` : '';
    const baseRecStr = bm ? `vs ${bm.mean_recovery_steps.toFixed(1)} base` : '';
    const baseCostStr = bm ? `vs +${bm.cost_delta_pct.toFixed(1)}% base` : '';
    const baseLlmStr = bm ? `vs ${bm.llm_calls} base` : '';

    if (elSuccess) elSuccess.innerHTML = `${(m.repair_success_rate * 100).toFixed(1)}%`;
    const baseSuccEl = document.getElementById('base-success');
    if (baseSuccEl) baseSuccEl.innerText = baseSuccessStr;

    if (elRecovery) elRecovery.innerHTML = `${m.mean_recovery_steps.toFixed(1)} <span class="unit">steps</span>`;
    const baseRecEl = document.getElementById('base-recovery');
    if (baseRecEl) baseRecEl.innerText = baseRecStr;

    if (elCost) elCost.innerHTML = `+${m.cost_delta_pct.toFixed(1)}%`;
    const baseCostEl = document.getElementById('base-cost');
    if (baseCostEl) baseCostEl.innerText = baseCostStr;

    const noLlm = m.llm_provider === 'heuristic';
    if (elLlm) {
      elLlm.innerHTML = this.isBaselineMode ? 'n/a <span class="unit">baseline</span>'
        : `${m.llm_calls} <span class="unit">${noLlm ? 'calls (rule-based, no LLM)' : 'calls'}</span>`;
      elLlm.title = noLlm ? `${m.heuristic_decisions ?? 0} decisions by the rule-based selector; no LLM was called` : '';
    }
    const baseLlmEl = document.getElementById('base-llm');
    if (baseLlmEl) baseLlmEl.innerText = `${m.cache_hits} cache hits ${baseLlmStr ? '(' + baseLlmStr + ')' : ''}`;

    const deltaSuccess = bm ? (m.repair_success_rate - bm.repair_success_rate) * 100 : 0;
    const badgeSuccess = document.getElementById('badge-success');
    if (badgeSuccess) {
      badgeSuccess.innerText = `${deltaSuccess >= 0 ? '+' : ''}${deltaSuccess.toFixed(1)}%`;
    }

    const deltaRec = bm ? m.mean_recovery_steps - bm.mean_recovery_steps : 0;
    const badgeRec = document.getElementById('badge-recovery');
    if (badgeRec) {
      badgeRec.innerText = `${deltaRec >= 0 ? '+' : ''}${deltaRec.toFixed(1)} steps`;
    }

    // Header chips
    const chipRobots = document.getElementById('chip-robots-count');
    if (chipRobots && this.activeReplay.frames[0]) {
      chipRobots.innerText = `${this.activeReplay.frames[0].pos.length}`;
    }
    const chipSteps = document.getElementById('chip-steps-count');
    if (chipSteps) chipSteps.innerText = `${this.activeReplay.meta.steps}`;
    const chipInc = document.getElementById('chip-incidents-count');
    if (chipInc) chipInc.innerText = `${this.activeReplay.incidents.length}`;
    const chipCache = document.getElementById('metric-cache-rate');
    if (chipCache) {
      // A hit = a stored strategy that matched AND passed its dry run (older replays
      // counted a match as a hit, so count from the incidents instead of the metric).
      const incs = this.activeReplay.incidents;
      const lookups = incs.filter(i => i.cache != null).length;
      const hits = incs.filter(i => i.cache?.hit && i.dry_run?.ran && i.dry_run.valid).length;
      if (this.isBaselineMode) {
        chipCache.innerHTML = 'n/a <span class="unit">baseline</span>';
      } else {
        const rate = lookups > 0 ? `${((hits / lookups) * 100).toFixed(1)}%` : DASH_TEXT;
        chipCache.innerHTML = `${hits} <span class="unit">/ ${lookups} (${rate})</span>`;
        chipCache.title = `${hits} cached strategies executed after passing the dry run, in ${lookups} lookups`;
      }
    }
    const elSafety = document.getElementById('metric-safety-violations');
    if (elSafety) {
      const executed = (m.executed_vertex_collisions ?? 0) + (m.executed_edge_swaps ?? 0) + (m.executed_safety_violations ?? 0);
      elSafety.innerText = m.safety_violations_intercepted != null ? `${executed}` : 'n/a';
      elSafety.title = m.safety_violations_intercepted != null
        ? `${executed} executed (collisions + unvalidated plans) · ${m.safety_violations_intercepted} invalid candidate plans caught by the validator`
        : 'not recorded in this replay';
    }
  }

  private buildTimelineTicks() {
    if (!this.activeReplay) return;
    const container = document.getElementById('timeline-ticks') || document.getElementById('track-incidents-viewport');
    if (!container) return;
    container.innerHTML = '';
    const totalSteps = this.activeReplay.meta.steps;

    // Dynamically build axis ticks from totalSteps
    const axisContainer = document.getElementById('timeline-axis-ticks');
    if (axisContainer) {
      axisContainer.innerHTML = '';
      const numTicks = 6;
      for (let i = 0; i <= numTicks; i++) {
        const val = Math.round((totalSteps * i) / numTicks);
        const span = document.createElement('span');
        span.innerText = `${val}`;
        axisContainer.appendChild(span);
      }
    }

    const tooltip = document.getElementById('incident-tooltip');

    this.activeReplay.incidents.forEach((inc) => {
      const tick = document.createElement('div');
      const type = inc.disruptions[0]?.type || 'blocked_cell';
      let typeClass = 'blocked';
      if (type === 'robot_failure') typeClass = 'failure';
      if (type === 'robot_delay') typeClass = 'delay';

      const kind = inc.overlap_kind || (inc.interacts_with.length > 0 ? 'sequential' : 'isolated');
      tick.className = `incident-tick ${typeClass} ${kind !== 'isolated' ? 'overlap' : ''}`;
      tick.dataset.incident = inc.id;
      tick.title = `${inc.id} · t=${inc.t_detected} · ${inc.disruptions.map(d => d.type.replace('_', ' ')).join(' + ')} · ${kind}`;
      const pct = (inc.t_detected / totalSteps) * 100;
      tick.style.left = `${pct}%`;

      // Hover Tooltip
      if (tooltip) {
        tick.addEventListener('mouseenter', () => {
          const rect = tick.getBoundingClientRect();
          tooltip.style.left = `${rect.left + rect.width / 2}px`;
          tooltip.style.bottom = `95px`;
          const ttHeader = document.getElementById('tt-header');
          if (ttHeader) ttHeader.innerText = `Incident ${inc.id} (t=${inc.t_detected})`;
          const ttBody = document.getElementById('tt-body');
          if (ttBody) ttBody.innerText = `${type.replace('_', ' ').toUpperCase()} • ${inc.affected_robots.length} AMRs affected`;
          tooltip.classList.remove('hidden');
        });

        tick.addEventListener('mouseleave', () => {
          tooltip.classList.add('hidden');
        });
      }

      tick.addEventListener('click', (e) => {
        e.stopPropagation();
        tooltip?.classList.add('hidden');
        this.selectIncident(inc);
      });

      container.appendChild(tick);
    });

    const incChip = document.getElementById('incident-counter-chip') || document.getElementById('incidents-badge-count');
    if (incChip) {
      incChip.innerText = `${this.activeReplay.incidents.length}`;
    }
    this.renderIncidentList();
  }

  /** Incident list (bottom-left card): one row per incident with its disruptions and overlap kind. */
  private renderIncidentList() {
    const list = document.getElementById('incidents-list-container');
    if (!list || !this.activeReplay) return;
    list.innerHTML = '';
    if (this.activeReplay.incidents.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'inc-entry-type';
      empty.innerText = 'No incidents in this replay.';
      list.appendChild(empty);
      return;
    }
    for (const inc of this.activeReplay.incidents) {
      const kind = inc.overlap_kind || (inc.interacts_with.length > 0 ? 'interacting' : 'isolated');
      const ok = inc.executed_valid ?? inc.validator.some(v => v.valid);
      const row = document.createElement('div');
      row.className = `incident-entry-item${this.selectedIncident?.id === inc.id ? ' active' : ''}`;
      row.dataset.incident = inc.id;
      row.innerHTML = `
        <div class="inc-entry-details">
          <div class="inc-entry-id"><span>${inc.id}</span><span class="inc-entry-time">t=${inc.t_detected}</span></div>
          <div class="inc-entry-type">${inc.disruptions.map(d => d.type.replace('_', ' ')).join(' + ')}</div>
          <div class="inc-entry-tags">
            <span class="overlap-tag ${kind}">${kind}</span>
            <span class="overlap-tag ${ok ? 'ok' : 'bad'}">${inc.decision_source ?? (ok ? 'repaired' : 'failed')}</span>
          </div>
        </div>`;
      row.addEventListener('click', () => this.selectIncident(inc));
      list.appendChild(row);
    }
  }

  private highlightIncident(id: string | null) {
    document.querySelectorAll<HTMLElement>('#incidents-list-container .incident-entry-item').forEach((el) => {
      el.classList.toggle('active', el.dataset.incident === id);
    });
    document.querySelectorAll<HTMLElement>('.incident-tick').forEach((el) => {
      el.classList.toggle('active', el.dataset.incident === id);
    });
  }

  private selectIncident(inc: Incident) {
    this.selectedIncident = inc;
    this.clockTime = inc.t_detected;
    this.updateSliderUI();
    this.showIncidentPanel(inc);
    this.updatePathsVisualization();
    this.setCameraPreset(2);
    this.highlightIncident(inc.id);
    this.showToast(`Incident ${inc.id}`,
      `t=${inc.t_detected} · ${inc.disruptions.map(d => d.type.replace('_', ' ')).join(' + ')} · ${inc.overlap_kind ?? 'isolated'}`);
  }

  private showIncidentPanel(inc: Incident) {
    const panel = document.getElementById('incident-panel') || document.getElementById('incidents-panel');
    if (panel) panel.classList.remove('hidden');

    const titleEl = document.getElementById('incident-panel-title');
    if (titleEl) titleEl.innerText = `Incident ${inc.id}`;
    const types = inc.disruptions.map(d => d.type.replace('_', ' ')).join(' + ');
    const kind = inc.overlap_kind || (inc.interacts_with.length > 0 ? 'interacting' : 'isolated');
    const subEl = document.getElementById('incident-panel-subtitle');
    if (subEl) subEl.innerText = `Step t=${inc.t_detected} • ${types.toUpperCase()} • ${kind.toUpperCase()}`;

    // 1. Sentinel
    const others = inc.interacts_with.filter(id => id !== inc.id);
    const interactsStr = others.length > 0 ? ` | Interacts with: [${others.join(', ')}]` : '';
    const related = (inc.related_disruptions ?? [])
      .map(r => `${r.type.replace('_', ' ')} @t=${r.t} (${r.incident_id})`).join('; ');
    const relatedStr = related ? ` | Overlaps earlier: ${related}` : '';
    const mustStr = inc.must_replan && inc.must_replan.length ? ` | Must replan: [${inc.must_replan.join(', ')}]` : '';
    const densityStr = inc.congestion.density != null ? `${(inc.congestion.density * 100).toFixed(1)}%` : 'n/a';
    const sentEl = document.getElementById('inc-sentinel-text');
    if (sentEl) {
      sentEl.innerText = `Disruptions (${inc.disruptions.length}): ${types} • Overlap: ${kind} • Congestion: ${inc.congestion.level.toUpperCase()} (density: ${densityStr})${mustStr}${interactsStr}${relatedStr}`;
    }

    const chipsContainer = document.getElementById('inc-affected-chips');
    if (chipsContainer) {
      chipsContainer.innerHTML = '';
      inc.affected_robots.forEach((rId) => {
        const chip = document.createElement('span');
        chip.className = 'robot-chip';
        chip.innerText = `AMR #${rId}`;
        chip.title = `Click to inspect AMR #${rId}`;
        chip.addEventListener('click', () => {
          this.selectedRobotId = rId;
          this.showRobotCard(rId);
          this.setCameraPreset(3);
        });
        chipsContainer.appendChild(chip);
      });
    }

    // 2. Cache
    const cacheHit = inc.cache?.hit;
    const simPct = Math.round((inc.cache?.similarity || 0) * 100);
    const cacheBadge = document.getElementById('inc-cache-badge');
    if (cacheBadge) {
      const verified = cacheHit && inc.dry_run?.valid;
      cacheBadge.className = `diag-badge ${verified ? 'green' : cacheHit ? 'amber' : 'purple'}`;
      cacheBadge.innerText = verified ? 'CACHE HIT' : cacheHit ? 'MATCH, REJECTED' : 'MISS';
    }

    const cacheHitText = !inc.cache
      ? 'Cache not used for this incident (cache disabled or baseline mode).'
      : cacheHit
        ? `Matched stored strategy ${inc.cache?.matched} with ${simPct}% similarity (RBF on incident features).`
          + (inc.dry_run?.valid ? ' Dry run passed, so it was executed: strategist skipped.' : ' Dry run failed, so it counts as a miss: fell through to the strategist.')
        : 'MISS: No verified strategy in the same category above the similarity threshold.';
    const cTextEl = document.getElementById('inc-cache-text');
    if (cTextEl) cTextEl.innerText = cacheHitText;
    const cSimVal = document.getElementById('inc-cache-sim-val');
    if (cSimVal) cSimVal.innerText = `${simPct}%`;
    const cSimFill = document.getElementById('inc-cache-sim-fill');
    if (cSimFill) cSimFill.style.width = `${simPct}%`;

    // 3. Dry Run
    const dryRun = inc.dry_run;
    const dryBadge = document.getElementById('inc-dryrun-badge');
    if (dryBadge) {
      if (dryRun && dryRun.ran) {
        dryBadge.className = `diag-badge ${dryRun.valid ? 'green' : 'amber'}`;
        dryBadge.innerText = dryRun.valid ? 'PASSED ✓' : 'FAILED ✗';
      } else {
        dryBadge.className = 'diag-badge';
        dryBadge.innerText = 'SKIPPED';
      }
    }
    const dryText = document.getElementById('inc-dryrun-text');
    if (dryText) {
      if (dryRun && dryRun.ran) {
        dryText.innerText = dryRun.valid
          ? 'VALID: Cached strategy directive verified clean against live obstacle state.'
          : `INVALID: ${dryRun.violation || 'conflict detected'}. Discarded and promoted to negative memory.`;
      } else {
        dryText.innerText = 'Skipped (Cache miss or first observation).';
      }
    }

    // 4. LLM Strategist
    const llmBadge = document.getElementById('inc-llm-badge');
    const llmText = document.getElementById('inc-llm-text');
    const llmLat = document.getElementById('inc-llm-latency');
    const llmTok = document.getElementById('inc-llm-tokens');
    const llmSolv = document.getElementById('inc-llm-solver');
    const llmDir = document.getElementById('inc-llm-directive');

    const calls = inc.llm?.calls ?? [];
    const callsStr = calls.map(c =>
      `#${c.attempt} ${c.provider}${c.model ? '/' + c.model : ''}: ${Math.round(c.latency_ms)} ms, ${c.tokens} tok`
      + (c.api_requests > 1 ? `, ${c.api_requests} requests (schema re-ask)` : '')
      + (c.error ? ` - ERROR ${c.error}` : '')).join('\n');
    if (inc.llm?.called) {
      if (llmBadge) { llmBadge.className = 'diag-badge pink'; llmBadge.innerText = `LLM CALLED (${inc.llm.provider ?? 'llm'})`; }
      if (llmText) llmText.innerText = (inc.llm.directive?.rationale || '(no rationale returned)') + (callsStr ? '\n' + callsStr : '');
      if (llmLat) llmLat.innerText = inc.llm.latency_ms != null ? `⚡ ${inc.llm.latency_ms} ms` : '⚡ n/a';
      if (llmTok) llmTok.innerText = inc.llm.tokens != null ? `🎟️ ${inc.llm.tokens} tokens` : '🎟️ n/a';
      if (llmSolv) llmSolv.innerText = `Solver: ${(inc.llm.directive?.solver ?? 'n/a').toUpperCase()}`;
      if (llmDir) llmDir.innerText = JSON.stringify(inc.llm.directive, null, 2);
    } else if (inc.llm) {
      // The strategist ran but no LLM answered (rule-based selector, or the LLM call failed)
      if (llmBadge) { llmBadge.className = 'diag-badge amber'; llmBadge.innerText = 'RULE-BASED (NO LLM)'; }
      if (llmText) llmText.innerText = 'Decided by the deterministic rule-based selector, not by an LLM.' + (callsStr ? '\n' + callsStr : '');
      if (llmLat) llmLat.innerText = `⚡ ${inc.llm.latency_ms ?? 0} ms`;
      if (llmTok) llmTok.innerText = '🎟️ 0 tokens';
      if (llmSolv) llmSolv.innerText = `Solver: ${(inc.llm.directive?.solver ?? 'n/a').toUpperCase()}`;
      if (llmDir) llmDir.innerText = JSON.stringify(inc.llm.directive ?? {}, null, 2);
    } else {
      if (llmBadge) { llmBadge.className = 'diag-badge'; llmBadge.innerText = 'NOT CALLED'; }
      const why = inc.decision_source === 'cache' ? 'Resolved by a verified cache hit.'
        : this.isBaselineMode ? 'Fixed-strategy baseline: no strategist.'
        : 'No strategist call recorded for this incident.';
      if (llmText) llmText.innerText = why;
      if (llmLat) llmLat.innerText = '⚡ 0 ms';
      if (llmTok) llmTok.innerText = '🎟️ 0 tokens';
      if (llmSolv) llmSolv.innerText = `Solver: ${(inc.executed_directive?.solver ?? 'n/a').toUpperCase()}`;
      if (llmDir) llmDir.innerText = JSON.stringify(inc.executed_directive ?? { note: 'no directive recorded' }, null, 2);
    }

    // 5. Solver: what was actually executed, and whether it passed validation
    const exec = inc.executed_directive ?? inc.llm?.directive ?? null;
    const solvText = document.getElementById('inc-solver-text');
    if (solvText) {
      if (inc.decision_source === 'emergency_hold') {
        solvText.innerText = 'All candidate plans (incl. the fallback) were rejected by the validator: robots held in place (emergency hold).';
      } else if (exec) {
        const src = inc.decision_source ? ` [source: ${inc.decision_source}]` : '';
        const validity = inc.executed_valid === false ? ' Executed WITHOUT passing validation (no-validator ablation).'
          : inc.executed_valid ? ' Plan passed the validator before execution.' : '';
        solvText.innerText = `Executed ${exec.solver.toUpperCase()} on robots [${exec.robots.join(', ')}] with horizon ${exec.horizon}.${src}${validity}`;
      } else {
        solvText.innerText = 'No executed directive recorded in this replay.';
      }
    }

    // 6. Validator
    const valAttempts = inc.validator || [];
    const valBadge = document.getElementById('inc-validator-badge');
    const allValid = valAttempts.some(v => v.valid);
    if (valBadge) {
      valBadge.className = `diag-badge ${allValid ? 'green' : 'red'}`;
      valBadge.innerText = allValid ? 'VERIFIED ✓' : 'FAILED ✗';
    }

    const valText = valAttempts.map(v => `Attempt ${v.attempt}: ${v.valid ? 'VALID ✓' : 'INVALID ✗ (' + v.violation + ')'}`).join('\n');
    const valEl = document.getElementById('inc-validator-text');
    if (valEl) valEl.innerText = valText || 'No validator attempts recorded.';
  }

  private updatePathsVisualization() {
    this.incidentOverlay.setIncident(this.selectedIncident, this.activeReplay);
    this.setIncidentCallout(this.selectedIncident);
  }

  /** After switching LLM/baseline, point the selection at the same incident id in the new replay. */
  private reselectIncident() {
    if (!this.selectedIncident) return;
    const matching = this.activeReplay?.incidents.find(i => i.id === this.selectedIncident?.id) ?? null;
    this.selectedIncident = matching;
    if (matching) this.showIncidentPanel(matching);
    else document.getElementById('incident-panel')?.classList.add('hidden');
    this.highlightIncident(matching?.id ?? null);
    this.updatePathsVisualization();
  }

  /** Fill the incident callout for a new selection; its screen position is updated every frame. */
  private setIncidentCallout(inc: Incident | null) {
    const el = document.getElementById('callout-incident');
    if (!el) return;
    if (!inc) {
      el.style.display = 'none';
      return;
    }
    const first = inc.disruptions[0];
    const extra = inc.disruptions.length > 1 ? ` +${inc.disruptions.length - 1}` : '';
    const title = document.getElementById('callout-inc-title');
    if (title) title.innerText = `${first ? DISRUPTION_LABELS[first.type] : 'Incident'}${extra} · ${inc.id}`;
    const sub = document.getElementById('callout-inc-sub');
    if (sub) sub.innerText = `detected t=${inc.t_detected} · executed t=${inc.executed_at} · ${inc.affected_robots.length} robots${inc.fallback_used ? ' · fallback' : ''}`;
    const note = document.getElementById('callout-inc-note');
    if (note) {
      const alignment = this.incidentOverlay.alignment;
      note.innerText = alignment !== null && alignment < 0.8 ? 'Routes in this replay do not match the robots\' recorded motion' : '';
      note.style.display = note.innerText ? '' : 'none';
    }
  }

  /** Project the callout anchors to screen space; hide them when off screen or behind the camera. */
  private updateCallouts() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.updateMatrixWorld(); // controls moved the camera this frame; avoid a one-frame lag
    const place = (el: HTMLElement | null, world: THREE.Vector3 | null) => {
      if (!el) return;
      const v = world?.project(this.camera);
      if (!v || v.z > 1 || Math.abs(v.x) > 1.05 || Math.abs(v.y) > 1.05) {
        el.style.display = 'none';
        return;
      }
      el.style.display = '';
      el.style.left = `${(v.x * 0.5 + 0.5) * w}px`;
      el.style.top = `${(-v.y * 0.5 + 0.5) * h}px`;
    };

    const incidentEl = document.getElementById('callout-incident');
    place(incidentEl, this.selectedIncident ? this.incidentOverlay.anchor(this.calloutAnchor) : null);
    const incidentX = incidentEl && incidentEl.style.display !== 'none' ? parseFloat(incidentEl.style.left) : null;

    const replay = this.activeReplay;
    const r = this.selectedRobotId;
    const pose = r !== null ? this.fleet.poses[r] : undefined;
    if (!replay || r === null || !pose || r >= replay.frames[0].pos.length) {
      place(document.getElementById('callout-robot'), null);
      return;
    }
    const step = Math.max(0, Math.min(Math.floor(this.clockTime), replay.frames.length - 1));
    const key = `${r}:${step}:${replay.meta.mode}`;
    if (key !== this.calloutRobotKey) {
      this.calloutRobotKey = key;
      const frame = replay.frames[step];
      const next = replay.frames[step + 1]?.pos[r];
      const [x, y] = frame.pos[r];
      const goal = frame.goals?.[r];
      const set = (id: string, text: string) => { const el = document.getElementById(id); if (el) el.innerText = text; };
      set('callout-robot-id', `R${r}`);
      set('callout-robot-pos', `(${x}, ${y})`);
      set('callout-robot-dest', goal ? `(${goal[0]}, ${goal[1]})` : 'not recorded');
      set('callout-robot-next', !next ? '—' : next[0] === x && next[1] === y ? 'wait' : `(${next[0]}, ${next[1]})`);
      const stateEl = document.getElementById('callout-robot-state');
      if (stateEl) {
        stateEl.innerText = frame.state[r];
        stateEl.className = `state-badge ${frame.state[r]}`;
      }
    }
    const robotEl = document.getElementById('callout-robot');
    place(robotEl, this.calloutAnchor.set(pose.x, 0.95, pose.z));
    // Open the card away from the incident badge so the two never cover each other.
    if (robotEl) robotEl.classList.toggle('flip-left', incidentX !== null && incidentX > parseFloat(robotEl.style.left));
  }

  private showRobotCard(rId: number) {
    this.selectedRobotId = rId;
    const card = document.getElementById('robot-card') || document.getElementById('robot-details-card');
    if (card) card.classList.remove('hidden');

    const s0 = Math.floor(this.clockTime);
    const frame = this.activeReplay?.frames[s0];
    const pos = frame?.pos[rId] || [0, 0];
    const state = frame?.state[rId] || 'moving';

    const cardId = document.getElementById('robot-card-id') || document.getElementById('detail-robot-id');
    if (cardId) cardId.innerText = `AMR #${rId}`;
    const stateEl = document.getElementById('robot-card-state') || document.getElementById('detail-state');
    if (stateEl) stateEl.innerText = state.toUpperCase();
    const posEl = document.getElementById('robot-card-pos') || document.getElementById('detail-pos');
    if (posEl) posEl.innerText = `(${pos[0]}, ${pos[1]})`;
    const goal = frame?.goals?.[rId];
    const goalEl = document.getElementById('robot-card-goal') || document.getElementById('detail-dest');
    if (goalEl) goalEl.innerText = goal ? `(${goal[0]}, ${goal[1]})` : 'not recorded';

    const dot = document.getElementById('robot-card-status-dot');
    if (dot) {
      if (state === 'failed') dot.style.background = '#ef4444';
      else if (state === 'delayed') dot.style.background = '#f59e0b';
      else if (state === 'idle') dot.style.background = '#10b981';
      else if (state === 'replanning') dot.style.background = '#d946ef';
      else dot.style.background = '#38bdf8';
    }

    const totalSteps = this.activeReplay ? this.activeReplay.frames.length - 1 : 0;
    const timeEl = document.getElementById('detail-timestep');
    if (timeEl) timeEl.innerText = `${s0} / ${totalSteps}`;

    // Find any incident involving this robot
    const inc = this.activeReplay?.incidents.find(i => i.affected_robots.includes(rId));
    const incEl = document.getElementById('detail-incident');
    if (incEl) {
      if (inc) {
        incEl.innerText = `Incident #${inc.id}`;
        incEl.style.cursor = 'pointer';
        incEl.onclick = () => this.selectIncident(inc);
      } else {
        incEl.innerText = 'None';
        incEl.style.cursor = 'default';
        incEl.onclick = null;
      }
    }

    const routeStatusEl = document.getElementById('detail-route-status');
    if (routeStatusEl) {
      if (state === 'failed') routeStatusEl.innerText = 'Failed';
      else if (state === 'delayed') routeStatusEl.innerText = 'Delayed';
      else if (inc) routeStatusEl.innerText = inc.executed_valid ? 'Repaired' : 'Violated';
      else routeStatusEl.innerText = 'Normal';
    }

    const origEl = document.getElementById('detail-orig-steps');
    if (origEl) {
      origEl.innerText = goal ? `${Math.abs(goal[0] - pos[0]) + Math.abs(goal[1] - pos[1])} steps (dist)` : '--';
    }

    const invalEl = document.getElementById('detail-inval-steps');
    if (invalEl) {
      const invalPath = inc?.paths?.invalidated?.[rId.toString()];
      invalEl.innerText = invalPath ? `${invalPath.length} steps` : '0 steps';
    }

    const repEl = document.getElementById('detail-repaired-steps');
    if (repEl) {
      const repPath = inc?.paths?.repaired?.[rId.toString()];
      repEl.innerText = repPath ? `${repPath.length} steps` : '0 steps';
    }
  }

  private showToast(title: string, desc: string) {
    const toast = document.getElementById('toast-notification');
    if (!toast) return;
    const titleEl = document.getElementById('toast-title');
    if (titleEl) titleEl.innerText = title;
    const descEl = document.getElementById('toast-desc');
    if (descEl) descEl.innerText = desc;
    toast.classList.remove('hidden');

    setTimeout(() => {
      toast.classList.add('hidden');
    }, 3200);
  }

  private setupUI() {
    // Play/Pause button
    const playPauseBtn = document.getElementById('play-pause-btn');
    const playIcon = document.getElementById('play-icon');
    const playIconSvg = document.getElementById('play-icon-svg');
    const playText = document.getElementById('play-text');
    playPauseBtn?.addEventListener('click', () => {
      this.isPlaying = !this.isPlaying;
      if (playIcon) playIcon.innerHTML = this.isPlaying ? '&#10074;&#10074;' : '&#9654;';
      if (playIconSvg) {
        playIconSvg.innerHTML = this.isPlaying
          ? '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>'
          : '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
      }
      if (playText) playText.innerText = this.isPlaying ? 'Pause' : 'Play';
    });

    // Jump to Start & End buttons
    document.getElementById('jump-start-btn')?.addEventListener('click', () => {
      this.clockTime = 0.0;
      this.updateSliderUI();
    });
    document.getElementById('jump-end-btn')?.addEventListener('click', () => {
      const maxSteps = (this.activeReplay?.meta.steps || 60) - 1;
      this.clockTime = maxSteps;
      this.updateSliderUI();
    });

    // Step forward & back frame buttons
    document.getElementById('step-back-btn')?.addEventListener('click', () => {
      this.isPlaying = false;
      this.clockTime = Math.max(0, this.clockTime - 1);
      this.updateSliderUI();
    });
    document.getElementById('step-fwd-btn')?.addEventListener('click', () => {
      this.isPlaying = false;
      const maxSteps = (this.activeReplay?.meta.steps || 300) - 1;
      this.clockTime = Math.min(maxSteps, this.clockTime + 1);
      this.updateSliderUI();
    });

    // Speed: the redesigned page uses a <select>; pill buttons kept for older layouts
    const speedSelect = document.getElementById('speed-select') as HTMLSelectElement | null;
    speedSelect?.addEventListener('change', () => {
      this.playSpeed = parseFloat(speedSelect.value) || 1;
    });
    const speedButtons = document.querySelectorAll('.speed-pill-group .speed-btn');
    speedButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        speedButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.playSpeed = parseFloat(btn.getAttribute('data-speed') || '1');
      });
    });

    // Slider
    const slider = document.getElementById('step-slider') as HTMLInputElement | null;
    slider?.addEventListener('input', () => {
      this.clockTime = parseFloat(slider.value);
      this.updateStepIndicator();
    });

    // Camera preset buttons
    const camButtons = document.querySelectorAll('.camera-preset-bar .cam-btn');
    camButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        camButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const p = parseInt(btn.getAttribute('data-cam') || '1', 10);
        this.setCameraPreset(p);
      });
    });

    // In-viewport camera preset buttons
    document.getElementById('btn-cam-full')?.addEventListener('click', () => this.setCameraPreset(1));
    document.getElementById('btn-cam-robot')?.addEventListener('click', () => this.setCameraPreset(3));
    document.getElementById('btn-cam-incident')?.addEventListener('click', () => this.setCameraPreset(2));
    document.getElementById('btn-cam-reset')?.addEventListener('click', () => this.setCameraPreset(1));

    // Scenario Select (replays generated by scripts/make_viewer_replays.py)
    const scenarioSelect = document.getElementById('scenario-select') as HTMLSelectElement | null;
    scenarioSelect?.addEventListener('change', () => {
      this.loadScenario(scenarioSelect.value, this.currentMode);
    });

    // Mode Toggle Buttons
    const modeLlmBtn = document.getElementById('mode-llm-btn');
    const modeBaseBtn = document.getElementById('mode-base-btn');
    if (modeLlmBtn && modeBaseBtn) {
      const switchMode = async (mode: string) => {
        modeLlmBtn.classList.toggle('active', mode === 'llm');
        modeBaseBtn.classList.toggle('active', mode !== 'llm');
        await this.loadScenario(this.currentScenario, mode);
      };
      modeLlmBtn.addEventListener('click', () => switchMode('llm'));
      modeBaseBtn.addEventListener('click', () => switchMode('baseline_pibt'));
    }

    // Mode Select Dropdown
    const modeSelect = document.getElementById('mode-select') as HTMLSelectElement | null;
    modeSelect?.addEventListener('change', () => {
      // each mode is its own run (PIBT, LNS and PBS baselines differ), so load its file
      this.loadScenario(this.currentScenario, modeSelect.value);
    });

    // Incident Drawer close button
    document.getElementById('incident-close-btn')?.addEventListener('click', () => {
      document.getElementById('incident-panel')?.classList.add('hidden'); // keep the incidents list visible
      this.selectedIncident = null;
      this.highlightIncident(null);
      this.updatePathsVisualization();
    });

    // Collapse the incidents list (CSS for .collapsed already exists)
    document.getElementById('incidents-collapse-btn')?.addEventListener('click', () => {
      document.getElementById('incidents-panel')?.classList.toggle('collapsed');
    });

    // Robot Card close & follow buttons
    document.getElementById('robot-card-close')?.addEventListener('click', () => {
      document.getElementById('robot-card')?.classList.add('hidden');
      this.selectedRobotId = null;
    });

    document.getElementById('detail-close-btn')?.addEventListener('click', () => {
      document.getElementById('robot-card')?.classList.add('hidden');
      document.getElementById('robot-details-card')?.classList.add('hidden');
      this.selectedRobotId = null;
    });

    document.getElementById('robot-follow-btn')?.addEventListener('click', () => {
      this.setCameraPreset(3);
    });

    // Next / Prev incident buttons
    document.getElementById('next-inc-btn')?.addEventListener('click', () => this.jumpIncident(1));
    document.getElementById('prev-inc-btn')?.addEventListener('click', () => this.jumpIncident(-1));

    // Fullscreen button
    document.getElementById('btn-fullscreen')?.addEventListener('click', () => {
      this.toggleFullscreen();
    });

    // Copy Directive JSON
    document.getElementById('copy-directive-btn')?.addEventListener('click', () => {
      const code = document.getElementById('inc-llm-directive')?.innerText || '';
      navigator.clipboard.writeText(code);
      this.showToast('Copied to Clipboard', 'Directive JSON copied');
    });

    // Help Modal
    const helpBtn = document.getElementById('help-btn');
    const helpModal = document.getElementById('help-modal');
    const helpClose = document.getElementById('help-modal-close');
    helpBtn?.addEventListener('click', () => helpModal?.classList.remove('hidden'));
    helpClose?.addEventListener('click', () => helpModal?.classList.add('hidden'));
    helpModal?.addEventListener('click', (e) => {
      if (e.target === helpModal) helpModal?.classList.add('hidden');
    });
  }

  private toggleFullscreen() {
    const card = document.getElementById('viewport-card') || this.container;
    if (!document.fullscreenElement) {
      if (card.requestFullscreen) {
        card.requestFullscreen().catch(() => {});
      } else if ((card as any).webkitRequestFullscreen) {
        (card as any).webkitRequestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen().catch(() => {});
      } else if ((document as any).webkitExitFullscreen) {
        (document as any).webkitExitFullscreen();
      }
    }
  }

  private setupEventListeners() {
    window.addEventListener('keydown', (e) => {
      this.keysPressed[e.key.toLowerCase()] = true;
      if (e.key === '1') this.setCameraPreset(1);
      if (e.key === '2') this.setCameraPreset(2);
      if (e.key === '3') this.setCameraPreset(3);
      if (e.key === '0' || e.key.toLowerCase() === 'r') this.setCameraPreset(1);
      if (e.key.toLowerCase() === 'f') this.toggleFullscreen();
      if (e.key.toLowerCase() === 'n') this.jumpIncident(1);
      if (e.key.toLowerCase() === 'p') this.jumpIncident(-1);
      if (e.key === 'ArrowLeft') {
        this.clockTime = Math.max(0, this.clockTime - 1);
        this.updateSliderUI();
      }
      if (e.key === 'ArrowRight') {
        const maxSteps = (this.activeReplay?.meta.steps || 300) - 1;
        this.clockTime = Math.min(maxSteps, this.clockTime + 1);
        this.updateSliderUI();
      }
      if (e.key === 'Escape') {
        if (document.fullscreenElement) {
          document.exitFullscreen().catch(() => {});
        }
        document.getElementById('incident-panel')?.classList.add('hidden');
        document.getElementById('robot-card')?.classList.add('hidden');
        document.getElementById('help-modal')?.classList.add('hidden');
        this.selectedIncident = null;
        this.selectedRobotId = null;
        this.updatePathsVisualization();
      }
      if (e.code === 'Space') {
        e.preventDefault();
        const playBtn = document.getElementById('play-pause-btn');
        playBtn?.click();
      }
    });

    window.addEventListener('keyup', (e) => {
      this.keysPressed[e.key.toLowerCase()] = false;
    });

    document.addEventListener('fullscreenchange', () => {
      this.onWindowResize();
      const btn = document.getElementById('btn-fullscreen');
      if (btn) {
        if (document.fullscreenElement) {
          btn.classList.add('active');
          btn.title = 'Exit Fullscreen (Key F or Esc)';
        } else {
          btn.classList.remove('active');
          btn.title = 'Toggle Fullscreen (Key F)';
        }
      }
    });

    // Click (not an orbit drag) on a robot selects it
    let downX = 0;
    let downY = 0;
    this.renderer.domElement.addEventListener('pointerdown', (e) => {
      downX = e.clientX;
      downY = e.clientY;
    });
    this.renderer.domElement.addEventListener('pointerup', (e) => {
      if (Math.hypot(e.clientX - downX, e.clientY - downY) > 5) return;
      const rect = this.renderer.domElement.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      );

      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(mouse, this.camera);
      const rId = this.fleet.pick(raycaster);
      if (rId !== null) this.showRobotCard(rId);
    });
  }

  private jumpIncident(direction: number) {
    if (!this.activeReplay) return;
    const incs = this.activeReplay.incidents;
    if (incs.length === 0) return;

    if (!this.selectedIncident) {
      this.selectIncident(direction > 0 ? incs[0] : incs[incs.length - 1]);
      return;
    }

    const currIdx = incs.findIndex(i => i.id === this.selectedIncident?.id);
    let nextIdx = currIdx + direction;
    if (nextIdx >= incs.length) nextIdx = 0;
    if (nextIdx < 0) nextIdx = incs.length - 1;

    this.selectIncident(incs[nextIdx]);
  }

  /** Closest camera pose, looking from (pitch, yaw), at which the whole floor rectangle is on screen. */
  private framePose(minX: number, minZ: number, maxX: number, maxZ: number, pitch: number, pos: THREE.Vector3, target: THREE.Vector3, zoomOut = 1) {
    const dir = new THREE.Vector3(
      Math.sin(OVERVIEW_YAW) * Math.cos(pitch),
      Math.sin(pitch),
      Math.cos(OVERVIEW_YAW) * Math.cos(pitch),
    );
    target.set((minX + maxX) / 2, 0, (minZ + maxZ) / 2);
    const probe = this.camera.clone();
    const corner = new THREE.Vector3();
    const fits = (dist: number) => {
      probe.position.copy(target).addScaledVector(dir, dist);
      probe.lookAt(target);
      probe.updateMatrixWorld();
      for (const x of [minX, maxX]) {
        for (const z of [minZ, maxZ]) {
          corner.set(x, 0, z).project(probe);
          if (Math.abs(corner.x) > 0.96 || Math.abs(corner.y) > 0.92) return false;
        }
      }
      return true;
    };
    let lo = 5;
    let hi = 1000;
    for (let i = 0; i < 30; i++) {
      const mid = (lo + hi) / 2;
      if (fits(mid)) hi = mid; else lo = mid;
    }
    pos.copy(target).addScaledVector(dir, hi * zoomOut);
  }

  /** Whole-map framing at a steep angle so aisles stay visible. */
  private overviewPose(pos: THREE.Vector3, target: THREE.Vector3, zoomOut = 1) {
    this.framePose(0, 0, this.mapWidth, this.mapHeight, OVERVIEW_PITCH, pos, target, zoomOut);
  }

  private setCameraPreset(preset: number) {
    this.activePreset = preset;
    document.querySelectorAll('.cam-preset-btn[data-cam]').forEach(btn => {
      const cam = btn.getAttribute('data-cam');
      btn.classList.toggle('active', (preset === 1 && cam === 'full') || (preset === 2 && cam === 'incident') || (preset === 3 && cam === 'robot'));
    });

    this.cameraStartPos.copy(this.camera.position);
    this.cameraStartTarget.copy(this.controls.target);

    if (preset === 1) {
      this.overviewPose(this.cameraEndPos, this.cameraEndTarget);
    } else if (preset === 2) {
      const b = this.incidentOverlay.focusBounds();
      if (b) {
        const pad = 3;
        this.framePose(b.min.x - pad, b.min.z - pad, b.max.x + pad, b.max.z + pad, INCIDENT_PITCH, this.cameraEndPos, this.cameraEndTarget);
      } else {
        this.overviewPose(this.cameraEndPos, this.cameraEndTarget, 0.5);
      }
    } else if (preset === 3) {
      const pose = this.fleet.poses[this.selectedRobotId ?? 0];
      const x = pose?.x ?? this.mapWidth / 2;
      const z = pose?.z ?? this.mapHeight / 2;
      this.cameraEndPos.set(x - 3, 7, z + 8);
      this.cameraEndTarget.set(x, 0.3, z);
    }

    this.cameraAnimProgress = 0.0;
    this.cameraAnimDuration = 0.8;
    this.isCameraTransitioning = true;
  }

  private startIntroDolly() {
    this.overviewPose(this.cameraStartPos, this.cameraStartTarget, 1.45);
    this.overviewPose(this.cameraEndPos, this.cameraEndTarget);
    this.cameraAnimProgress = 0.0;
    this.cameraAnimDuration = 2.0;
    this.isCameraTransitioning = true;
  }

  private updateCameraAnimation(dt: number) {
    if (!this.isCameraTransitioning) {
      // Follow camera tracks the selected robot's interpolated pose
      if (this.activePreset === 3 && this.selectedRobotId !== null) {
        const pose = this.fleet.poses[this.selectedRobotId];
        if (pose) {
          const before = this.controls.target.clone();
          this.controls.target.lerp(this.followTarget.set(pose.x, 0.3, pose.z), 0.1);
          this.camera.position.add(before.sub(this.controls.target).negate());
        }
      }
      return;
    }

    this.cameraAnimProgress += dt / this.cameraAnimDuration;
    if (this.cameraAnimProgress >= 1.0) {
      this.cameraAnimProgress = 1.0;
      this.isCameraTransitioning = false;
    }

    const t = this.cameraAnimProgress;
    const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

    this.camera.position.lerpVectors(this.cameraStartPos, this.cameraEndPos, ease);
    this.controls.target.lerpVectors(this.cameraStartTarget, this.cameraEndTarget, ease);
    this.controls.update();
  }

  private updateSliderUI() {
    const slider = document.getElementById('step-slider') as HTMLInputElement;
    if (slider) slider.value = Math.floor(this.clockTime).toString();
    this.updateStepIndicator();
  }

  private updateStepIndicator() {
    const cur = Math.floor(this.clockTime);
    const total = this.activeReplay?.meta.steps || 300;
    const indicator = document.getElementById('step-indicator');
    if (indicator) {
      indicator.innerText = `Step ${cur} / ${total}`;
    }
    const pct = Math.round((cur / (total - 1)) * 100);
    const pctEl = document.getElementById('step-pct');
    if (pctEl) pctEl.innerText = `(${pct}%)`;

    const progBar = document.getElementById('slider-progress-bar');
    if (progBar) progBar.style.width = `${pct}%`;

    const readout = document.getElementById('timestep-display');
    if (readout) readout.innerText = `${cur} / ${total - 1}`;

    if (this.selectedRobotId !== null && this.activeReplay) {
      const frame = this.activeReplay.frames[cur];
      const pos = frame?.pos[this.selectedRobotId];
      if (pos) {
        const posEl = document.getElementById('detail-pos');
        if (posEl) posEl.innerText = `(${pos[0]}, ${pos[1]})`;
      }
      const timeEl = document.getElementById('detail-timestep');
      if (timeEl) timeEl.innerText = `${cur} / ${total - 1}`;
    }

    // Playhead sits over the track lanes: 10px container padding + 100px track label on the left, 10px on the right.
    // Uses the same t / steps scale as the incident ticks so the playhead crosses a tick at its t_detected.
    const playhead = document.getElementById('timeline-playhead');
    if (playhead) playhead.style.left = `calc(110px + (100% - 120px) * ${Math.min(1, this.clockTime / total)})`;
  }

  private affectedRobots(): Set<number> | null {
    if (this.affectedFor !== this.selectedIncident) {
      this.affectedFor = this.selectedIncident;
      this.affectedSet = this.selectedIncident ? new Set(this.selectedIncident.affected_robots) : null;
    }
    return this.affectedSet;
  }

  private renderLoop(now: number) {
    if (!this.lastTimestamp) this.lastTimestamp = now;
    const dt = Math.min((now - this.lastTimestamp) / 1000, 0.1);
    this.lastTimestamp = now;

    this.frameCount++;
    if (now - this.lastFpsUpdate > 500) {
      const fps = Math.round((this.frameCount * 1000) / (now - this.lastFpsUpdate));
      if (this.fpsDisplay) this.fpsDisplay.innerText = `${fps} FPS`;
      this.frameCount = 0;
      this.lastFpsUpdate = now;
    }

    const replay = this.activeReplay;
    if (this.isPlaying && replay) {
      const last = replay.frames.length - 1;
      this.clockTime += dt * STEPS_PER_SECOND * this.playSpeed;
      if (this.clockTime >= last) this.clockTime = 0.0;
      this.updateSliderUI();
    }

    if (replay) {
      const timeSec = now / 1000;
      const camDist = this.camera.position.distanceTo(this.controls.target);
      const markerScale = THREE.MathUtils.smoothstep(camDist, 30, 70) * Math.min(camDist * 0.012, 3);
      this.fleet.update(replay.frames, this.clockTime, { selected: this.selectedRobotId, affected: this.affectedRobots() }, timeSec, markerScale);
      this.disruptions.update(replay, this.clockTime, this.fleet.poses, timeSec, Math.max(1, camDist * 0.012));
      this.incidentOverlay.update(this.clockTime, this.fleet.poses, dt, timeSec, THREE.MathUtils.clamp(camDist / 30, 1, 3));
      const focus = this.selectedIncident ? FOCUS_DIM : 1;
      environmentBrightness.value += (focus - environmentBrightness.value) * Math.min(1, dt * 5);
    }
    this.updateCameraAnimation(dt);

    // Smooth keyboard flight navigation (WASD to pan, QE to orbit)
    const focused = document.activeElement?.tagName;
    if (!this.isCameraTransitioning && focused !== 'INPUT' && focused !== 'TEXTAREA' && focused !== 'SELECT') {
      const panSpeed = (this.camera.position.y * 0.75 + 15) * dt;
      const forward = new THREE.Vector3();
      this.camera.getWorldDirection(forward);
      forward.y = 0;
      forward.normalize();
      const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();

      if (this.keysPressed['w']) {
        this.controls.target.addScaledVector(forward, panSpeed);
        this.camera.position.addScaledVector(forward, panSpeed);
      }
      if (this.keysPressed['s']) {
        this.controls.target.addScaledVector(forward, -panSpeed);
        this.camera.position.addScaledVector(forward, -panSpeed);
      }
      if (this.keysPressed['a']) {
        this.controls.target.addScaledVector(right, -panSpeed);
        this.camera.position.addScaledVector(right, -panSpeed);
      }
      if (this.keysPressed['d']) {
        this.controls.target.addScaledVector(right, panSpeed);
        this.camera.position.addScaledVector(right, panSpeed);
      }
      if (this.keysPressed['q']) {
        const offset = this.camera.position.clone().sub(this.controls.target);
        offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), dt * 1.5);
        this.camera.position.copy(this.controls.target).add(offset);
      }
      if (this.keysPressed['e']) {
        const offset = this.camera.position.clone().sub(this.controls.target);
        offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), -dt * 1.5);
        this.camera.position.copy(this.controls.target).add(offset);
      }
    }

    this.controls.update();
    this.updateCallouts();
    this.pipeline.render();
  }

  private onWindowResize() {
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, MAX_PIXEL_RATIO));
    this.renderer.setSize(w, h);
  }
}

// Start application
const app = new ViewerApp();
app.init();
if (import.meta.env.DEV) (window as unknown as { viewer: ViewerApp }).viewer = app; // console / screenshot harness access
