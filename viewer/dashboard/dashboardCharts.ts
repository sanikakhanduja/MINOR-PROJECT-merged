/**
 * Zero-dependency SVG Charting Engine for AMR Cost & Compute Analytics
 * Implements Line Charts, Grouped Bars, Stacked Latency Bars, and Interactive Tooltips
 */

// Shared tooltip element
let tooltipEl: HTMLDivElement | null = null;

function getTooltip(): HTMLDivElement {
  if (!tooltipEl) {
    tooltipEl = document.getElementById('chart-tooltip') as HTMLDivElement;
    if (!tooltipEl) {
      tooltipEl = document.createElement('div');
      tooltipEl.id = 'chart-tooltip';
      tooltipEl.className = 'chart-tooltip-box';
      document.body.appendChild(tooltipEl);
    }
  }
  return tooltipEl;
}

export function showTooltip(e: MouseEvent, html: string) {
  const tip = getTooltip();
  tip.innerHTML = html;
  tip.style.display = 'block';
  tip.style.left = `${e.pageX + 12}px`;
  tip.style.top = `${e.pageY - 28}px`;
}

export function hideTooltip() {
  const tip = getTooltip();
  tip.style.display = 'none';
}

function emptyMessage(width: number, height: number, text: string = 'No incidents match active filters'): string {
  return `
    <svg viewBox="0 0 ${width} ${height}" class="chart-svg-root" width="100%" height="100%">
      <text x="${width / 2}" y="${height / 2}" text-anchor="middle" fill="#8F969E" font-family="'Barlow Condensed', sans-serif" font-size="14">
        ${text}
      </text>
    </svg>
  `;
}

// ============================================================================
// 2-3. Validator panel
// (The former "with vs without cache" and "with vs without validator" charts were
// removed: their second series was estimated, never measured. Those comparisons
// need the Ablation_NoCache / Ablation_NoValidator runs.)
// ============================================================================
export function renderValidatorCostChart(
  containerId: string,
  costItems: Array<{ label: string; value: string; desc: string }>
) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = `
    <dl class="measured-list">
      ${costItems.map(item => `
        <div class="measured-row">
          <dt>${item.label}</dt>
          <dd>${item.value}</dd>
          <p>${item.desc}</p>
        </div>`).join('')}
    </dl>
  `;
}

export interface MethodBarDatum {
  method: string;
  value: number;
  color: string;
}

export function renderMethodComparisonChart(
  containerId: string,
  data: MethodBarDatum[],
  unit: string,
  maxValue?: number,
  referenceValue?: number
) {
  const container = document.getElementById(containerId);
  if (!container) return;
  if (!data.length) {
    container.innerHTML = emptyMessage(520, 240);
    return;
  }

  const W = 400;
  const H = 250;
  const pad = { top: 26, right: 12, bottom: 56, left: 40 };
  const innerW = W - pad.left - pad.right;
  const innerH = H - pad.top - pad.bottom;
  const scaleMax = maxValue || Math.max(...data.map(item => item.value)) * 1.18 || 1;
  const groupW = innerW / data.length;
  const barW = Math.min(42, groupW * 0.58);
  const referenceY = referenceValue === undefined ? null : pad.top + innerH - (referenceValue / scaleMax) * innerH;
  const decimals = unit === '× original' ? 2 : unit === 'count' ? 0 : 1;

  const grid = Array.from({ length: 5 }, (_, index) => {
    const value = (scaleMax * index) / 4;
    const y = pad.top + innerH - (value / scaleMax) * innerH;
    const label = unit === '%' ? `${Math.round(value)}%`
      : unit === '× original' ? `${value.toFixed(2)}×`
      : unit === 'count' ? `${Math.round(value)}` : value.toFixed(1);
    return `
      <line x1="${pad.left}" y1="${y.toFixed(1)}" x2="${W - pad.right}" y2="${y.toFixed(1)}" stroke="rgba(255,255,255,0.06)" />
      <text x="${pad.left - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" fill="#8F969E" font-size="14" font-family="'Barlow Condensed', sans-serif">${label}</text>
    `;
  }).join('');

  const bars = data.map((item, index) => {
    const height = Math.max(2, (item.value / scaleMax) * innerH);
    const x = pad.left + index * groupW + (groupW - barW) / 2;
    const y = pad.top + innerH - height;
    return `
      <rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${height.toFixed(1)}" rx="3"
        fill="${item.color}" class="chart-bar-rect"
        data-tooltip="${item.method}<br/><strong>${item.value.toFixed(decimals)} ${unit === 'count' ? '' : unit}</strong>" />
      <text x="${(x + barW / 2).toFixed(1)}" y="${Math.max(14, y - 5).toFixed(1)}" text-anchor="middle"
        fill="${item.color}" font-size="15" font-family="'Barlow Condensed', sans-serif">${item.value.toFixed(decimals)}</text>
      <text y="${H - 34}" text-anchor="middle"
        fill="#B4BAC1" font-size="15" font-family="'Barlow Condensed', sans-serif">${item.method.split(' ').map((word, i) => `<tspan x="${(pad.left + index * groupW + groupW / 2).toFixed(1)}" dy="${i === 0 ? 0 : 15}">${word}</tspan>`).join('')}</text>
    `;
  }).join('');

  container.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" class="chart-svg-root" width="100%" height="100%">
      ${grid}
      ${referenceY === null ? '' : `
        <line x1="${pad.left}" y1="${referenceY.toFixed(1)}" x2="${W - pad.right}" y2="${referenceY.toFixed(1)}" stroke="#FFB020" stroke-dasharray="4,4" opacity="0.7" />
      `}
      ${bars}
    </svg>
  `;
  attachTooltips(container);
}

export interface MethodDistributionDatum {
  method: string;
  values: number[];
  color: string;
}

export function renderMethodRecoveryBoxPlot(containerId: string, data: MethodDistributionDatum[], unit: 'steps' | 'ms' = 'steps') {
  const container = document.getElementById(containerId);
  if (!container) return;
  if (!data.length || data.every(item => item.values.length === 0)) {
    container.innerHTML = emptyMessage(520, 240);
    return;
  }

  const W = 400;
  const H = 250;
  const pad = { top: 22, right: 64, bottom: 32, left: 90 };
  const valueColumnWidth = 62;
  const plotRight = W - pad.right - valueColumnWidth;
  const innerW = plotRight - pad.left;
  const innerH = H - pad.top - pad.bottom;
  const maxValue = Math.max(...data.flatMap(item => item.values), 1) * 1.08;
  const useLogScale = unit === 'ms';
  const scaleValue = (value: number) => useLogScale ? Math.log10(value + 1) : value;
  const maxScaleValue = scaleValue(maxValue);
  const x = (value: number) => pad.left + (scaleValue(value) / maxScaleValue) * innerW;
  const quantile = (values: number[], fraction: number) => {
    const position = (values.length - 1) * fraction;
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    return values[lower] + (values[upper] - values[lower]) * (position - lower);
  };
  const rowH = innerH / data.length;

  const plots = data.map((item, index) => {
    const values = [...item.values].sort((a, b) => a - b);
    if (!values.length) return '';
    const min = values[0];
    const max = values[values.length - 1];
    const q1 = quantile(values, 0.25);
    const median = quantile(values, 0.5);
    const q3 = quantile(values, 0.75);
    const cy = pad.top + rowH * index + rowH / 2;
    const boxTop = cy - Math.min(10, rowH * 0.28);
    const boxHeight = Math.min(20, rowH * 0.56);
    return `
      <text x="${pad.left - 10}" y="${(cy + 3).toFixed(1)}" text-anchor="end" fill="#B4BAC1" font-size="15" font-family="'Barlow Condensed', sans-serif">${item.method}</text>
      <line x1="${x(min).toFixed(1)}" y1="${cy.toFixed(1)}" x2="${x(max).toFixed(1)}" y2="${cy.toFixed(1)}" stroke="${item.color}" stroke-width="1.5" />
      <line x1="${x(min).toFixed(1)}" y1="${(cy - 5).toFixed(1)}" x2="${x(min).toFixed(1)}" y2="${(cy + 5).toFixed(1)}" stroke="${item.color}" />
      <line x1="${x(max).toFixed(1)}" y1="${(cy - 5).toFixed(1)}" x2="${x(max).toFixed(1)}" y2="${(cy + 5).toFixed(1)}" stroke="${item.color}" />
      <rect x="${x(q1).toFixed(1)}" y="${boxTop.toFixed(1)}" width="${Math.max(1, x(q3) - x(q1)).toFixed(1)}" height="${boxHeight.toFixed(1)}"
        fill="${item.color}" fill-opacity="0.24" stroke="${item.color}" class="chart-bar-rect"
        data-tooltip="${item.method}<br/>Median: <strong>${median.toFixed(1)} ${unit}</strong><br/>IQR: ${q1.toFixed(1)}–${q3.toFixed(1)} ${unit}<br/>Range: ${min.toFixed(1)}–${max.toFixed(1)} ${unit}" />
      <line x1="${x(median).toFixed(1)}" y1="${boxTop.toFixed(1)}" x2="${x(median).toFixed(1)}" y2="${(boxTop + boxHeight).toFixed(1)}" stroke="${item.color}" stroke-width="2" />
      <text x="${W - pad.right + 8}" y="${(cy + 3).toFixed(1)}" text-anchor="start" fill="${item.color}" font-size="14" font-family="'Barlow Condensed', sans-serif">${median >= 100 ? median.toFixed(0) : median.toFixed(2)} ${unit}</text>
    `;
  }).join('');

  const tickValues = useLogScale
    ? [0, 1, 10, 100, 1000, 10000, 100000].filter(value => value < maxValue / 3)
    : Array.from({ length: 5 }, (_, index) => (maxValue * index) / 4);
  const ticks = tickValues.map(value => {
    const tickX = x(value);
    return `
      <line x1="${tickX.toFixed(1)}" y1="${pad.top}" x2="${tickX.toFixed(1)}" y2="${H - pad.bottom}" stroke="rgba(255,255,255,0.05)" />
      <text x="${tickX.toFixed(1)}" y="${H - 10}" text-anchor="middle" fill="#8F969E" font-size="14" font-family="'Barlow Condensed', sans-serif">${Math.round(value)}</text>
    `;
  }).join('');

  container.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" class="chart-svg-root" width="100%" height="100%">
      ${ticks}
      ${plots}
    </svg>
  `;
  attachTooltips(container);
}

// Attach tooltip hover handlers to any chart container
function attachTooltips(container: HTMLElement) {
  const elements = container.querySelectorAll('[data-tooltip]');
  elements.forEach(el => {
    el.addEventListener('mouseenter', e => {
      const html = el.getAttribute('data-tooltip') || '';
      showTooltip(e as MouseEvent, html);
    });
    el.addEventListener('mousemove', e => {
      const html = el.getAttribute('data-tooltip') || '';
      showTooltip(e as MouseEvent, html);
    });
    el.addEventListener('mouseleave', () => {
      hideTooltip();
    });
  });
}
