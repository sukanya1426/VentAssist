import type { TabularState } from "../../types/recommendation";

const W = 400;
const H = 56;
const P_MAX = 40; // cmH₂O at the top of the trace

const y = (p: number) => H - (Math.min(p, P_MAX) / P_MAX) * (H - 6) - 3;

/**
 * A schematic airway-pressure trace for the patient: PEEP baseline, one square-ish
 * breath per respiratory cycle, plateau scaled by tidal volume. Decorative — it
 * conveys the shape of the current settings, not measured waveform data.
 */
function pressurePath(state: TabularState): string {
  const peep = state.PEEP;
  const plateau = peep + Math.max(6, Math.min(28, state.TV / 26)); // ~driving pressure
  const breaths = Math.max(3, Math.round(state.RR / 4));
  const cycle = W / breaths;
  const pts: string[] = [`M 0 ${y(peep).toFixed(1)}`];

  for (let i = 0; i < breaths; i++) {
    const x0 = i * cycle;
    const rise = x0 + cycle * 0.12;
    const holdEnd = x0 + cycle * 0.42;
    const fall = x0 + cycle * 0.5;
    pts.push(
      `L ${rise.toFixed(1)} ${y(plateau).toFixed(1)}`,
      `L ${holdEnd.toFixed(1)} ${y(plateau - 1).toFixed(1)}`,
      `L ${fall.toFixed(1)} ${y(peep).toFixed(1)}`,
      `L ${(x0 + cycle).toFixed(1)} ${y(peep).toFixed(1)}`
    );
  }
  return pts.join(" ");
}

export function Waveform({
  state,
  className = "",
  tone = "cyan",
}: {
  state: TabularState;
  className?: string;
  tone?: "cyan" | "amber" | "rose";
}) {
  // Desaturated, barely-tinted greys — the trace is background texture, not a signal.
  const stroke = { cyan: "#94a3b8", amber: "#a8a08e", rose: "#ad97a0" }[tone];
  const d = pressurePath(state);
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      className={className}
      aria-hidden="true"
    >
      <path d={d} fill="none" stroke={stroke} strokeOpacity={0.14} strokeWidth={3} />
      <path
        d={d}
        fill="none"
        stroke={stroke}
        strokeOpacity={0.5}
        strokeWidth={1.1}
        strokeLinejoin="round"
        strokeDasharray="1000"
        className="animate-trace"
      />
    </svg>
  );
}
