import type { TabularState } from "../types/recommendation";

export type VitalLevel = "ok" | "warn" | "crit";

/** [critLow, warnLow, warnHigh, critHigh] — nulls disable that bound. */
type Band = [number | null, number | null, number | null, number | null];

const BANDS: Partial<Record<keyof TabularState, Band>> = {
  PEEP: [null, null, 15, 18],
  TV: [250, 300, 600, 700],
  FiO2: [null, null, 0.6, 0.8],
  SpO2: [88, 92, 99, null],
  PaO2: [55, 70, null, null],
  PaCO2: [28, 32, 48, 55],
  pH: [7.25, 7.32, 7.48, 7.55],
  HR: [45, 55, 110, 130],
  SBP: [80, 95, 160, 185],
  RR: [8, 10, 26, 32],
  Temp: [35.0, 35.8, 38.0, 39.0],
};

export function vitalLevel(key: keyof TabularState, v: number): VitalLevel {
  const b = BANDS[key];
  if (!b || Number.isNaN(v)) return "ok";
  const [critLo, warnLo, warnHi, critHi] = b;
  if ((critLo != null && v < critLo) || (critHi != null && v > critHi)) return "crit";
  if ((warnLo != null && v < warnLo) || (warnHi != null && v > warnHi)) return "warn";
  return "ok";
}

export const LEVEL_TEXT: Record<VitalLevel, string> = {
  ok: "text-slate-900",
  warn: "text-amber-600",
  crit: "text-rose-600",
};

export const LEVEL_DOT: Record<VitalLevel, string> = {
  ok: "bg-emerald-500",
  warn: "bg-amber-500",
  crit: "bg-rose-500",
};

/** Worst level across the whole state — drives the patient card's acuity rail. */
export function worstLevel(state: TabularState): VitalLevel {
  let worst: VitalLevel = "ok";
  for (const k of Object.keys(state) as (keyof TabularState)[]) {
    const l = vitalLevel(k, state[k]);
    if (l === "crit") return "crit";
    if (l === "warn") worst = "warn";
  }
  return worst;
}
