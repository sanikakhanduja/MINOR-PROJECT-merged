/**
 * Data layer for the benchmark dashboard.
 *
 * Every value here is read from a replay written by the Python pipeline
 * (amr_repair/loop/controller.py). Anything a replay does not contain is null
 * and shown as "—" by the dashboard: nothing is estimated, defaulted or filled in.
 *
 * Only the runs in RUNS are shown. The *_baseline_*.json files next to them are
 * the closed-loop fixed-solver runs used by the 3D viewer's Mode menu; results
 * compare methods on identical incidents through the counterfactual branches.
 */

export const RUNS = [
  'overlap_mixed',
  'latest_experiment',
  'isolated',
  'overlap_simultaneous',
  'overlap_sequential',
  'overlap_compound',
] as const;

export type DisruptionType = 'blocked_cell' | 'robot_failure' | 'robot_delay';
export const FIXED_ARMS = ['pibt', 'lns', 'windowed_pbs'] as const;
export type FixedArm = (typeof FIXED_ARMS)[number];

/** One branch of the counterfactual evaluation (amr_repair/loop/counterfactual.py). */
export interface ArmResult {
  solver: string | null;
  horizon: number | null;
  first_valid: boolean;
  source: 'direct' | 'fallback' | 'hold';
  goals: number;
  progress?: number;           // absent in arms recorded before fleet progress was measured
  recovery_steps: number;
  recovered: boolean;
  detour: number;
  shortest: number;
  solver_ms: number;
}

export interface Counterfactual {
  window: number;
  arms: Record<FixedArm | 'executed', ArmResult>;
  oracle: FixedArm[];
  distinct?: boolean;          // the fixed solvers did not all tie on this incident
  decision_delay?: number;
  matches_oracle: boolean;
  goal_regret: number;
  reproduced: boolean;
}

export interface Run {
  key: string;
  replay: any;
  provider: string | null; // provider that answered; "heuristic" = rule-based, null = baseline mode
  model: string | null;
  isLLM: boolean;
  label: string;
}

export interface IncidentRow {
  run: Run;
  id: string;
  t: number;
  types: DisruptionType[];
  primaryType: DisruptionType;
  overlapKind: string;
  overlapping: boolean;
  affected: number[];
  congestionLevel: string | null;
  density: number | null;
  decisionSource: string;
  cacheLookedUp: boolean;
  cacheMatch: boolean;         // a stored strategy matched above the threshold
  cacheHit: boolean;           // the match passed its dry run and was executed
  similarity: number | null;
  dryRunRan: boolean;
  dryRunValid: boolean | null;
  llmCalls: number;            // real LLM calls (rule-based decisions are not counted)
  llmTokens: number;
  promptTokens: number | null; // null when the run did not record the split
  completionTokens: number | null;
  llmLatencyMs: number;
  retryCalls: number;
  retryTokens: number;
  retryLatencyMs: number;
  attempts: Array<{ valid: boolean; violation: string | null }>;
  fallbackUsed: boolean;
  success: boolean;            // executed plan passed validation and was not an emergency hold
  solver: string | null;
  robotsReplanned: number | null;
  recoverySteps: number | null;
  recovered: boolean | null;
  costDelta: number | null;
  shortestCost: number | null;
  cf: Counterfactual | null;
}

async function fetchJson(path: string): Promise<any | null> {
  try {
    const res = await fetch(path);
    if (!res.ok || (res.headers.get('content-type') || '').includes('text/html')) return null;
    return await res.json();
  } catch {
    return null;
  }
}

function runLabel(key: string, replay: any): string {
  const m = replay.meta || {};
  const n = replay.metrics?.incidents_total ?? replay.incidents?.length ?? 0;
  const robots = m.num_robots ? `${m.num_robots} robots` : 'robots not recorded';
  const who = m.llm_provider === 'heuristic' ? 'rule-based, no LLM'
    : m.llm_provider ? `${m.llm_provider}${m.llm_model ? ` ${m.llm_model}` : ''}` : m.mode;
  return `${key} (${who}, ${robots}, ${n} incidents)`;
}

export async function loadRuns(): Promise<Run[]> {
  const loaded = await Promise.all(RUNS.map(async key => ({ key, replay: await fetchJson(`/replays/${key}.json`) })));
  return loaded
    .filter(r => r.replay && Array.isArray(r.replay.incidents))
    .map(({ key, replay }) => {
      const provider = replay.meta?.llm_provider ?? null;
      return {
        key, replay, provider, model: replay.meta?.llm_model ?? null,
        isLLM: provider !== null && provider !== 'heuristic',
        label: runLabel(key, replay),
      };
    });
}

const isRealCall = (c: any) => c && typeof c.provider === 'string' && !c.provider.startsWith('heuristic');

export function incidentRows(run: Run): IncidentRow[] {
  return run.replay.incidents.map((inc: any): IncidentRow => {
    const types = Array.from(new Set<DisruptionType>((inc.disruptions || []).map((d: any) => d.type)));
    const calls: any[] = inc.llm?.calls || [];
    const real = calls.filter(isRealCall);
    const hasSplit = real.length > 0 && real.every(c => typeof c.prompt_tokens === 'number' && c.prompt_tokens + c.completion_tokens > 0);
    const retries = real.filter(c => c.attempt >= 2);
    const overlapKind = inc.overlap_kind || ((inc.interacts_with || []).length ? 'sequential' : 'isolated');
    return {
      run,
      id: inc.id,
      t: inc.t_detected,
      types,
      primaryType: types[0] || inc.disruptions?.[0]?.type,
      overlapKind,
      overlapping: overlapKind !== 'isolated',
      affected: inc.affected_robots || [],
      congestionLevel: inc.congestion?.level ?? null,
      density: typeof inc.congestion?.density === 'number' ? inc.congestion.density : null,
      decisionSource: inc.decision_source || 'unknown',
      cacheLookedUp: inc.cache != null,
      cacheMatch: inc.cache?.hit === true,
      cacheHit: inc.cache?.hit === true && inc.dry_run?.ran === true && inc.dry_run.valid === true,
      similarity: typeof inc.cache?.similarity === 'number' ? inc.cache.similarity : null,
      dryRunRan: inc.dry_run?.ran === true,
      dryRunValid: inc.dry_run?.ran ? inc.dry_run.valid === true : null,
      llmCalls: real.length,
      llmTokens: real.reduce((s, c) => s + (c.tokens || 0), 0),
      promptTokens: hasSplit ? real.reduce((s, c) => s + c.prompt_tokens, 0) : null,
      completionTokens: hasSplit ? real.reduce((s, c) => s + c.completion_tokens, 0) : null,
      llmLatencyMs: real.reduce((s, c) => s + (c.latency_ms || 0), 0),
      retryCalls: retries.length,
      retryTokens: retries.reduce((s, c) => s + (c.tokens || 0), 0),
      retryLatencyMs: retries.reduce((s, c) => s + (c.latency_ms || 0), 0),
      attempts: (inc.validator || []).map((v: any) => ({ valid: v.valid === true, violation: v.violation ?? null })),
      fallbackUsed: inc.fallback_used === true,
      success: inc.executed_valid === true && inc.decision_source !== 'emergency_hold',
      solver: inc.executed_directive?.solver ?? null,
      robotsReplanned: inc.executed_directive ? inc.executed_directive.robots.length : null,
      recoverySteps: typeof inc.outcome?.recovery_steps === 'number' ? inc.outcome.recovery_steps : null,
      recovered: typeof inc.outcome?.recovered === 'boolean' ? inc.outcome.recovered : null,
      costDelta: typeof inc.outcome?.cost_delta === 'number' ? inc.outcome.cost_delta : null,
      shortestCost: typeof inc.outcome?.shortest_cost === 'number' ? inc.outcome.shortest_cost : null,
      cf: inc.counterfactual ?? null,
    };
  });
}

/** Validator violation messages start with a fixed keyword (amr_repair/validate/validator.py). */
export function violationKind(message: string | null): string {
  const m = (message || '').toLowerCase();
  if (m.startsWith('vertex')) return 'Vertex collision';
  if (m.includes('swap') || m.startsWith('edge')) return 'Edge swap';
  if (m.startsWith('blocked_cell')) return 'Blocked cell';
  if (m.startsWith('failed_robot')) return 'Failed robot';
  if (m.startsWith('deadlock')) return 'Deadlock';
  if (m.startsWith('parking')) return 'Unsafe parking';
  if (m.startsWith('delay')) return 'Delay ignored';
  return 'Other';
}

export const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
