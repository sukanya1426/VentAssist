import { PATIENT_PRESETS } from "./patientPresets";
import type { TabularState, Track } from "../types/recommendation";

/** The 6 waveform features Track B accepts (all optional). */
export interface WaveformFeatures {
  HRV_SDNN?: number;
  Arrhythmia_rate?: number;
  Perfusion_Index?: number;
  RRV?: number;
  Breathing_Regularity?: number;
  Asynchrony_Score?: number;
}

/**
 * A patient on the roster. Preset patients ship with the app; uploaded ones are
 * parsed from a clinician's file (see `patientFile.ts`) and held in the roster
 * store. Both carry everything `/api/recommend` needs.
 */
export interface Patient {
  id: string;          // route param, e.g. "patient-a"
  name: string;        // "Patient A"
  weight: number;      // kg → patient_weight
  age: number;
  sex: "M" | "F";
  bed: string;
  summary: string;
  state: TabularState;
  source: "preset" | "upload";
  hint?: string;                    // what the policy is expected to suggest
  waveform?: WaveformFeatures;      // present → Track B selectable
  ventilationMode?: string;         // "volume_control" | "pressure_control"
  track?: Track;                    // preferred track on open
  uploadedAt?: string;              // ISO timestamp, uploads only
}

const LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"];

const ROSTER_META: Omit<Patient, "id" | "name" | "state" | "source" | "hint">[] = [
  { weight: 74, age: 61, sex: "M", bed: "ICU-01", summary: "Post-op, settings and gases in range, weaning candidate" },
  { weight: 68, age: 55, sex: "F", bed: "ICU-02", summary: "ARDS, hypoxaemic on FiO₂ 0.5, tachypnoeic" },
  { weight: 81, age: 70, sex: "M", bed: "ICU-03", summary: "Oxygen saturation 100% on high FiO₂ — over-oxygenated" },
  { weight: 62, age: 48, sex: "F", bed: "ICU-04", summary: "COPD exacerbation, hypercapnic on low tidal volume" },
  { weight: 90, age: 66, sex: "M", bed: "ICU-05", summary: "High PEEP support, over-distension risk" },
  { weight: 77, age: 52, sex: "M", bed: "ICU-06", summary: "Large tidal volumes delivered — volutrauma risk" },
];

export const PATIENTS: Patient[] = PATIENT_PRESETS.map((preset, i) => ({
  id: `patient-${LETTERS[i].toLowerCase()}`,
  name: `Patient ${LETTERS[i]}`,
  state: { ...preset.state },
  hint: preset.hint,
  source: "preset" as const,
  ...(ROSTER_META[i] ?? {
    weight: 75, age: 60, sex: "M" as const, bed: `ICU-${i + 1}`, summary: preset.hint,
  }),
}));

/** Look up in a roster (preset + uploaded), newest roster passed by the caller. */
export function findPatient(roster: Patient[], id: string | undefined): Patient | undefined {
  return roster.find((p) => p.id === id);
}
