/**
 * Benchmark dashboard.
 *
 * Reads only real pipeline runs (see costComputeData.ts) and never estimates a
 * value: anything a run did not record is shown as "—" with the reason.
 *
 * The method comparison uses the counterfactual branches recorded per incident
 * (amr_repair/loop/counterfactual.py): every method starts from the same world
 * state at detection, so methods are compared on identical incidents.
 */

import {
  FIXED_ARMS,
  type FixedArm,
  type IncidentRow,
  type Run,
  incidentRows,
  loadRuns,
  mean,
  violationKind,
} from './costComputeData';

import {
  renderMethodComparisonChart,
  renderMethodRecoveryBoxPlot,
  renderValidatorCostChart,
} from './dashboardCharts';

const DASH = '—';
const COLORS: Record<string, string> = {
  Ours: '#F0821E',
  PIBT: '#6B9CCC',
  LNS: '#C9A57D',
  'Windowed PBS': '#86B8A2',
  Random: '#7A8189',
  Oracle: '#ECEDEE',
};
const ARM_LABEL: Record<FixedArm, string> = { pibt: 'PIBT', lns: 'LNS', windowed_pbs: 'Windowed PBS' };
const SOURCE_LABEL: Record<string, { label: string; color: string }> = {
  cache: { label: 'Cache', color: '#3CC8FF' },
  llm: { label: 'LLM', color: '#F0821E' },
  llm_retry: { label: 'LLM retry', color: '#FFB020' },
  heuristic: { label: 'Rules', color: '#7A8189' },
  heuristic_retry: { label: 'Rules retry', color: '#B4BAC1' },
  fallback: { label: 'Fallback', color: '#D65CFF' },
  baseline: { label: 'Fixed', color: '#6B9CCC' },
  emergency_hold: { label: 'Hold', color: '#FF5A4E' },
  recorded: { label: 'Recorded', color: '#8F969E' },
};

let runs: Run[] = [];
let selection = 'overlap_mixed';
const compareTypes = new Set(['blocked_cell', 'robot_delay', 'robot_failure']);
let compareMode: 'all' | 'single' | 'overlapping' = 'all';
const costTypes = new Set(['blocked_cell', 'robot_delay', 'robot_failure']);
let costMode: 'all' | 'single' | 'overlapping' = 'all';

const $ = (id: string) => document.getElementById(id);
const setText = (id: string, text: string) => {
  const el = $(id);
  if (el) el.textContent = text;
};
const fmt = (x: number | null, digits = 1, unit = '') => (x === null || Number.isNaN(x) ? DASH : `${x.toFixed(digits)}${unit}`);
const pct = (num: number, den: number) => (den > 0 ? (100 * num) / den : null);

function selectedRuns(): Run[] {
  if (selection === 'all_llm') return runs.filter(r => r.isLLM);
  if (selection === 'all_rule') return runs.filter(r => r.provider === 'heuristic');
  return runs.filter(r => r.key === selection);
}

function allRows(): IncidentRow[] {
  return selectedRuns().flatMap(incidentRows);
}

function filterRows(rows: IncidentRow[], types: Set<string>, mode: string): IncidentRow[] {
  return rows.filter(r => r.types.some(t => types.has(t)) &&
    (mode === 'all' || (mode === 'overlapping') === r.overlapping));
}

const ruleBasedOnly = () => selectedRuns().every(r => !r.isLLM);
const oursName = () => (ruleBasedOnly() ? 'the rule-based selector (no LLM)' : 'the LLM with the verified cache');

// ------------------------------------------------------------------ KPIs

function renderKPIs(rows: IncidentRow[]) {
  const sel = selectedRuns();
  const n = rows.length;
  setText('val-success-rate', fmt(pct(rows.filter(r => r.success).length, n), 1, '%'));

  const withCf = rows.filter(r => r.cf);
  if (withCf.length === n && n > 0) {
    const best = Math.max(...FIXED_ARMS.map(a => pct(withCf.filter(r => r.cf!.arms[a].source !== 'hold').length, n) ?? 0));
    setText('lift-success-rate', `best fixed solver: ${best.toFixed(1)}%`);
  } else {
    setText('lift-success-rate', 'no same-incident comparison recorded');
  }

  // A hit = a stored strategy that matched AND passed its dry run (it was executed)
  const lookups = rows.filter(r => r.cacheLookedUp).length;
  const matches = rows.filter(r => r.cacheMatch).length;
  const hits = rows.filter(r => r.cacheHit).length;
  setText('val-cache-rate', lookups ? fmt(pct(hits, lookups), 1, '%') : DASH);
  setText('val-llm-avoided', !lookups ? 'cache not used in this run'
    : matches ? `${hits} of ${matches} matches passed the dry run (precision ${fmt(pct(hits, matches), 0, '%')})`
    : `no stored strategy matched in ${lookups} lookups`);

  const rec = rows.map(r => r.recoverySteps).filter((x): x is number => x !== null);
  setText('val-mean-recovery', fmt(mean(rec), 1));
  if (withCf.length === n && n > 0) {
    const ours = mean(withCf.map(r => r.cf!.arms.executed.recovery_steps))!;
    const fixed = FIXED_ARMS.map(a => mean(withCf.map(r => r.cf!.arms[a].recovery_steps))!);
    setText('lift-mean-recovery', `same incidents: ${ours.toFixed(1)} vs best fixed ${Math.min(...fixed).toFixed(1)}`);
  } else {
    setText('lift-mean-recovery', 'no same-incident comparison recorded');
  }

  const goals = sel.reduce((s, r) => s + (r.replay.metrics?.throughput ?? 0), 0);
  const steps = sel.reduce((s, r) => s + Math.max(0, (r.replay.meta?.steps ?? 1) - 1), 0);
  setText('val-throughput', steps ? (goals / steps).toFixed(2) : DASH);
}

// ------------------------------------------------- same-incident comparison

interface Outcome { success: number; recovery: number; goals: number; detour: number; shortest: number; latency: number; win: number }

function armOutcome(row: IncidentRow, arm: FixedArm): Outcome {
  const a = row.cf!.arms[arm];
  return {
    success: a.source !== 'hold' ? 1 : 0, recovery: a.recovery_steps, goals: a.goals,
    detour: a.detour, shortest: a.shortest, latency: a.solver_ms, win: row.cf!.oracle.includes(arm) ? 1 : 0,
  };
}

function comparison(rows: IncidentRow[]) {
  const methods: Record<string, Outcome[]> = { Ours: [], PIBT: [], LNS: [], 'Windowed PBS': [], Random: [], Oracle: [] };
  for (const row of rows) {
    const cf = row.cf!;
    const ex = cf.arms.executed;
    methods.Ours.push({
      success: ex.source !== 'hold' ? 1 : 0, recovery: ex.recovery_steps, goals: ex.goals, detour: ex.detour,
      shortest: ex.shortest, latency: row.llmLatencyMs + ex.solver_ms, win: cf.matches_oracle ? 1 : 0,
    });
    const fixed = FIXED_ARMS.map(a => armOutcome(row, a));
    FIXED_ARMS.forEach((a, i) => methods[ARM_LABEL[a]].push(fixed[i]));
    // Random = exact expectation of picking one fixed solver uniformly
    const avg = (k: keyof Outcome) => fixed.reduce((s, o) => s + o[k], 0) / fixed.length;
    methods.Random.push({ success: avg('success'), recovery: avg('recovery'), goals: avg('goals'), detour: avg('detour'),
      shortest: avg('shortest'), latency: avg('latency'), win: avg('win') });
    methods.Oracle.push({ ...armOutcome(row, cf.oracle[0]), win: 1 });
  }
  return Object.entries(methods).map(([method, os]) => ({ method, color: COLORS[method], os }));
}

function updateComparison() {
  const scoped = filterRows(allRows(), compareTypes, compareMode);
  const rows = scoped.filter(r => r.cf);
  const status = $('compare-status');
  const containers = ['chart-compare-recovery', 'chart-compare-cost',
    'chart-compare-throughput', 'chart-compare-latency', 'chart-compare-win-rate'];
  if (!rows.length) {
    const why = scoped.length
      ? 'This run has no counterfactual branches. Add them with scripts/counterfactual_from_replay.py, or rerun with --counterfactual-window.'
      : 'No incidents match the filters.';
    if (status) status.textContent = why;
    containers.forEach(id => { const el = $(id); if (el) el.innerHTML = `<p class="chart-not-measured">${why}</p>`; });
    return;
  }
  const window = rows[0].cf!.window;
  if (status) {
    const missing = scoped.length - rows.length;
    status.textContent = `${rows.length} incidents, each method rolled forward ${window} steps from the same state. ` +
      `Ours = ${oursName()}.${missing ? ` ${missing} incidents without branches are left out.` : ''} ` +
      `With this few incidents, differences are not statistically meaningful.`;
  }
  const data = comparison(rows);
  const sum = (os: Outcome[], k: keyof Outcome) => os.reduce((s, o) => s + o[k], 0);
  renderMethodRecoveryBoxPlot('chart-compare-recovery',
    data.map(d => ({ method: d.method, color: d.color, values: d.os.map(o => o.recovery) })));
  const ratios = data.map(d => {
    const sh = sum(d.os, 'shortest');
    return { method: d.method, color: d.color, value: sh > 0 ? 1 + sum(d.os, 'detour') / sh : 1 };
  });
  renderMethodComparisonChart('chart-compare-cost', ratios, '× original', Math.max(1.3, ...ratios.map(r => r.value * 1.15)), 1);
  renderMethodComparisonChart('chart-compare-throughput',
    data.map(d => ({ method: d.method, color: d.color, value: sum(d.os, 'goals') / d.os.length })), 'goals');
  renderMethodRecoveryBoxPlot('chart-compare-latency',
    data.map(d => ({ method: d.method, color: d.color, values: d.os.map(o => o.latency) })), 'ms');
  renderMethodComparisonChart('chart-compare-win-rate',
    data.map(d => ({ method: d.method, color: d.color, value: 100 * sum(d.os, 'win') / d.os.length })), '%', 100);
  setText('compare-window-label', `${window} steps`);
}

// --------------------------------------------------------- cost & compute

function updateCost() {
  const sel = selectedRuns();
  const rows = filterRows(allRows(), costTypes, costMode);
  const n = rows.length;
  const llmRows = rows.filter(r => r.llmCalls > 0);
  const calls = rows.reduce((s, r) => s + r.llmCalls, 0);
  const tokens = rows.reduce((s, r) => s + r.llmTokens, 0);
  const decidedByCache = rows.filter(r => r.decisionSource === 'cache');

  setText('cc-val-llm-calls', calls.toLocaleString());
  setText('cc-delta-llm-calls', ruleBasedOnly() ? 'rule-based selector: no LLM' : `${decidedByCache.length} incidents decided by cache`);
  setText('cc-val-tokens', tokens ? tokens.toLocaleString() : '0');
  const split = llmRows.length > 0 && llmRows.every(r => r.promptTokens !== null);
  setText('cc-val-dollar-cost', split
    ? `${llmRows.reduce((s, r) => s + r.promptTokens!, 0).toLocaleString()} in / ${llmRows.reduce((s, r) => s + r.completionTokens!, 0).toLocaleString()} out`
    : 'input/output split not recorded');
  const perDecision = llmRows.length ? tokens / llmRows.length : null;
  setText('cc-delta-tokens', perDecision !== null
    ? `${Math.round(perDecision).toLocaleString()} tokens per LLM decision`
    : 'no LLM decisions');

  // Decision latency = LLM time + solver time of the executed plan + validator time.
  const solverMs = (r: IncidentRow) => r.cf ? r.cf.arms.executed.solver_ms : r.run.replay.metrics?.mean_solver_latency_ms ?? 0;
  const validatorMs = (r: IncidentRow) => r.run.replay.metrics?.mean_validator_latency_ms ?? 0;
  const latency = (r: IncidentRow) => r.llmLatencyMs + solverMs(r) + validatorMs(r);
  setText('cc-val-avg-latency', fmt(mean(rows.map(latency)), 1));
  const hitLat = mean(decidedByCache.map(latency));
  const llmLat = mean(llmRows.map(latency));
  setText('cc-delta-latency', `cache hit ${fmt(hitLat, 1, ' ms')} · LLM ${fmt(llmLat, 0, ' ms')}`);

  // How each incident was decided (replaces the former estimated with/without-cache chart)
  const sources: Record<string, number> = {};
  rows.forEach(r => { sources[r.decisionSource] = (sources[r.decisionSource] || 0) + 1; });
  renderMethodComparisonChart('chart-with-without-cache', Object.entries(sources).map(([k, v]) => ({
    method: SOURCE_LABEL[k]?.label || k, color: SOURCE_LABEL[k]?.color || '#8F969E', value: v,
  })), 'count');

  // Validator panel
  const invalid = rows.flatMap(r => r.attempts.filter(a => !a.valid));
  const firstFailed = rows.filter(r => r.attempts[0] && !r.attempts[0].valid && r.decisionSource !== 'cache');
  const retryOk = rows.filter(r => r.decisionSource.endsWith('_retry')).length;
  const valMs = mean(sel.map(r => r.replay.metrics?.mean_validator_latency_ms).filter((x): x is number => typeof x === 'number'));
  setText('val-stat-caught', invalid.length.toString());
  setText('val-stat-retry-rate', firstFailed.length ? fmt(pct(retryOk, firstFailed.length), 0, '%') : DASH);
  setText('val-stat-retry-desc', firstFailed.length ? `${retryOk} of ${firstFailed.length} failed first attempts` : 'no failed first attempts');
  setText('val-stat-fallback-rate', fmt(pct(rows.filter(r => r.fallbackUsed).length, n), 1, '%'));
  setText('val-stat-avg-time', fmt(valMs, 2, ' ms'));
  const collisions = sel.reduce((s, r) => s + (r.replay.metrics?.executed_vertex_collisions ?? 0) + (r.replay.metrics?.executed_edge_swaps ?? 0), 0);
  setText('validator-collisions-chip', `${collisions} collisions executed (audited from frames)`);

  const kinds: Record<string, number> = {};
  invalid.forEach(a => { const k = violationKind(a.violation); kinds[k] = (kinds[k] || 0) + 1; });
  const violEl = $('chart-validator-violations');
  if (!invalid.length) {
    if (violEl) violEl.innerHTML = '<p class="chart-not-measured">The validator rejected no plan in this selection.</p>';
  } else {
    renderMethodComparisonChart('chart-validator-violations',
      Object.entries(kinds).map(([k, v]) => ({ method: k, value: v, color: '#FF5A4E' })), 'count');
  }
  const retryCalls = rows.reduce((s, r) => s + r.retryCalls, 0);
  renderValidatorCostChart('chart-validator-cost', [
    { label: 'Check time', value: fmt(valMs, 2, ' ms'), desc: 'Mean validator time per plan (run telemetry)' },
    { label: 'Retry tokens', value: `${rows.reduce((s, r) => s + r.retryTokens, 0).toLocaleString()}`, desc: `Tokens spent on ${retryCalls} LLM retries after a violation` },
    { label: 'Retry latency', value: `${Math.round(rows.reduce((s, r) => s + r.retryLatencyMs, 0)).toLocaleString()} ms`, desc: 'Total LLM time spent on retries' },
  ]);

  // Summary table
  const dryRuns = rows.filter(r => r.dryRunRan);
  const dryFail = dryRuns.filter(r => r.dryRunValid === false).length;
  const cfRows = rows.filter(r => r.cf);
  const armMs = (a: FixedArm) => (cfRows.length ? fmt(mean(cfRows.map(r => r.cf!.arms[a].solver_ms)), 2, ' ms') : DASH);
  const replanned = rows.map(r => r.robotsReplanned).filter((x): x is number => x !== null);
  const tbody = $('cc-table-summary-body');
  if (tbody) {
    const row = (name: string, value: string, note: string) =>
      `<tr><td>${name}</td><td class="num">${value}</td><td class="note">${note}</td></tr>`;
    tbody.innerHTML = [
      row('Dry-run rejection rate', dryRuns.length ? fmt(pct(dryFail, dryRuns.length), 1, '%') : DASH,
        dryRuns.length ? `${dryFail} of ${dryRuns.length} cache matches failed the dry run` : 'no cache matches'),
      row('PIBT solver time', armMs('pibt'), 'Mean, same-state branches'),
      row('LNS solver time', armMs('lns'), 'Mean, same-state branches'),
      row('Windowed PBS solver time', armMs('windowed_pbs'), 'Mean, same-state branches'),
      row('Robots replanned', fmt(mean(replanned), 1), 'Mean per executed directive'),
    ].join('');
  }
}

// ------------------------------------------------------------ incident log

function validatorText(r: IncidentRow): string {
  if (r.decisionSource === 'emergency_hold') return '<span class="validator-status-tag hold">Emergency hold</span>';
  if (!r.attempts.length) return DASH;
  const n = r.attempts.length;
  if (r.fallbackUsed) return `<span class="validator-status-tag fallback">Passed on fallback (${n} plans)</span>`;
  if (n === 1) return '<span class="validator-status-tag">Passed first try</span>';
  return `<span class="validator-status-tag retry">Passed on try ${n}</span>`;
}

function cacheText(r: IncidentRow): string {
  if (!r.cacheLookedUp) return '<span class="cache-status-tag off">Cache off</span>';
  if (r.cacheMatch) {
    const sim = r.similarity !== null ? ` (sim ${r.similarity.toFixed(2)})` : '';
    return r.cacheHit
      ? `<span class="cache-status-tag hit">Cache hit${sim}</span>`
      : `<span class="cache-status-tag miss">Matched${sim}, failed dry run</span>`;
  }
  return `<span class="cache-status-tag miss">Miss, ${r.llmCalls ? 'LLM called' : 'rule-based'}</span>`;
}

function renderIncidents() {
  const tbody = $('incidents-table-body');
  if (!tbody) return;
  const q = (($('incident-search-input') as HTMLInputElement | null)?.value || '').toLowerCase().trim();
  const type = ($('incident-filter-type') as HTMLSelectElement | null)?.value || 'all';
  const cache = ($('incident-filter-cache') as HTMLSelectElement | null)?.value || 'all';
  const rows = allRows().filter(r => {
    if (type !== 'all' && !r.types.includes(type as any)) return false;
    if (cache === 'hit' && !r.cacheHit) return false;
    if (cache === 'miss' && (r.cacheHit || !r.cacheLookedUp)) return false;
    if (q && !(r.id.toLowerCase().includes(q) || r.types.some(t => t.includes(q)) || r.affected.some(a => `${a}`.includes(q)))) return false;
    return true;
  });
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="table-empty">No incidents match these filters.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(r => {
    const type = r.primaryType || 'disruption';
    const typeClass = type.includes('delay') ? 'delay' : type.includes('block') ? 'blocked' : 'failure';
    const congestion = r.congestionLevel
      ? `${r.congestionLevel}${r.density !== null ? ` (${Math.round(r.density * 100)}%)` : ''}` : DASH;
    const recovery = r.recoverySteps === null ? DASH : `${r.recoverySteps} steps${r.recovered === false ? ' (not recovered by run end)' : ''}`;
    const solver = r.solver ? `<span class="solver-badge ${r.solver}">${r.solver.toUpperCase()}</span>` : DASH;
    return `
      <tr>
        <td class="mono strong">${r.id}${selection.startsWith('all_') ? `<br/><span class="run-key">${r.run.key}</span>` : ''}</td>
        <td class="mono">t = ${r.t}</td>
        <td><span class="type-badge ${typeClass}">${r.types.map(t => t.replace('_', ' ')).join(' + ')}</span></td>
        <td class="mono">[${r.affected.join(', ')}]</td>
        <td class="congestion ${r.congestionLevel || ''}">${congestion}</td>
        <td>${cacheText(r)}</td>
        <td>${solver}</td>
        <td>${validatorText(r)}</td>
        <td class="mono">${recovery}</td>
        <td>
          <a href="/?experiment=${r.run.key}&step=${r.t}" class="btn-table-action" title="Open the 3D replay at step ${r.t}">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
            <span>View 3D</span>
          </a>
        </td>
      </tr>`;
  }).join('');
}

// ----------------------------------------------------------------- wiring

function update() {
  const rows = allRows();
  renderKPIs(rows);
  updateComparison();
  updateCost();
  renderIncidents();
}

function bindPills(attr: string, set: Set<string>, onChange: () => void) {
  document.querySelectorAll<HTMLButtonElement>(`[${attr}]`).forEach(btn => {
    btn.addEventListener('click', () => {
      const v = btn.getAttribute(attr)!;
      if (set.has(v)) {
        if (set.size === 1) return;
        set.delete(v);
      } else {
        set.add(v);
      }
      btn.classList.toggle('active', set.has(v));
      onChange();
    });
  });
}

function bindModes(attr: string, set: (m: any) => void, onChange: () => void) {
  const buttons = document.querySelectorAll<HTMLButtonElement>(`[${attr}]`);
  buttons.forEach(btn => btn.addEventListener('click', () => {
    set(btn.getAttribute(attr));
    buttons.forEach(b => b.classList.toggle('active', b === btn));
    onChange();
  }));
}

document.addEventListener('DOMContentLoaded', async () => {
  runs = await loadRuns();
  const select = $('dashboard-scenario-select') as HTMLSelectElement | null;
  if (select) {
    const llm = runs.filter(r => r.isLLM);
    const rule = runs.filter(r => r.provider === 'heuristic');
    const count = (rs: Run[]) => rs.reduce((s, r) => s + r.replay.incidents.length, 0);
    const opt = (id: string, label: string) => `<option value="${id}">${label}</option>`;
    select.innerHTML =
      (llm.length ? `<optgroup label="LLM runs">${llm.map(r => opt(r.key, r.label)).join('')}${llm.length > 1 ? opt('all_llm', `All LLM runs (${llm.length} runs, ${count(llm)} incidents)`) : ''}</optgroup>` : '') +
      (rule.length ? `<optgroup label="Rule-based runs (no LLM)">${rule.map(r => opt(r.key, r.label)).join('')}${rule.length > 1 ? opt('all_rule', `All rule-based runs (${rule.length} runs, ${count(rule)} incidents)`) : ''}</optgroup>` : '');
    if (!runs.some(r => r.key === selection)) selection = runs[0]?.key ?? selection;
    select.value = selection;
    select.addEventListener('change', () => { selection = select.value; update(); });
  }
  ['incident-search-input', 'incident-filter-type', 'incident-filter-cache'].forEach(id =>
    $(id)?.addEventListener(id.endsWith('input') ? 'input' : 'change', renderIncidents));
  bindPills('data-compare-type', compareTypes, updateComparison);
  bindModes('data-compare-mode', m => { compareMode = m; }, updateComparison);
  bindPills('data-type', costTypes, updateCost);
  bindModes('data-mode', m => { costMode = m; }, updateCost);
  update();
});
