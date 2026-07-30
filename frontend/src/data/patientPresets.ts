import type { TabularState } from "../types/recommendation";

/**
 * Clinically distinct preset patient states, one per patient on the roster
 * (see `patients.ts`). Each is chosen to exercise a different part of the policy so
 * the recommendation differs meaningfully between patients. All fields stay editable
 * on the patient page (editing marks the state as "Edited" until reset).
 */
export interface PatientPreset {
  key: string;
  label: string;
  hint: string;          // what the policy is expected to suggest
  state: TabularState;
  waveform?: boolean;    // whether waveform data is "available" for this patient
}

export const PATIENT_PRESETS: PatientPreset[] = [
  {
    key: "stable",
    label: "Stable / weaning (in-band)",
    hint: "settings & gases in range → expect Hold",
    state: { PEEP: 8, TV: 460, FiO2: 0.4, SpO2: 95, PaO2: 88, PaCO2: 40, pH: 7.40,
             HR: 84, SBP: 120, RR: 16, RASS: -1, Temp: 37.0 },
  },
  {
    key: "hypoxaemia",
    label: "Hypoxaemic (low SpO₂)",
    hint: "SpO₂ 84 on FiO₂ 0.5 → expect ↑PEEP / ↑FiO₂",
    state: { PEEP: 8, TV: 480, FiO2: 0.5, SpO2: 84, PaO2: 55, PaCO2: 44, pH: 7.34,
             HR: 104, SBP: 112, RR: 26, RASS: -2, Temp: 37.6 },
  },
  {
    key: "hyperoxia",
    label: "Hyperoxic on high FiO₂",
    hint: "SpO₂ 100 on FiO₂ 0.7 → expect ↓FiO₂",
    state: { PEEP: 8, TV: 480, FiO2: 0.7, SpO2: 100, PaO2: 90, PaCO2: 44, pH: 7.37,
             HR: 92, SBP: 118, RR: 22, RASS: -2, Temp: 37.2 },
  },
  {
    key: "hypercapnia",
    label: "Hypercapnic / low tidal volume",
    hint: "PaCO₂ 65 with TV 320 → expect ↑TV",
    state: { PEEP: 8, TV: 320, FiO2: 0.5, SpO2: 93, PaO2: 72, PaCO2: 65, pH: 7.28,
             HR: 98, SBP: 118, RR: 28, RASS: -1, Temp: 37.3 },
  },
  {
    key: "high_peep",
    label: "High PEEP (over-distension risk)",
    hint: "PEEP 18 → expect ↓PEEP",
    state: { PEEP: 18, TV: 470, FiO2: 0.5, SpO2: 94, PaO2: 80, PaCO2: 43, pH: 7.38,
             HR: 90, SBP: 105, RR: 20, RASS: -3, Temp: 37.2 },
  },
  {
    key: "volutrauma",
    label: "High tidal volume (volutrauma risk)",
    hint: "TV 760 → expect ↓TV",
    state: { PEEP: 8, TV: 760, FiO2: 0.5, SpO2: 95, PaO2: 90, PaCO2: 38, pH: 7.44,
             HR: 86, SBP: 122, RR: 14, RASS: -2, Temp: 37.0 },
  },
];

export const DEFAULT_PRESET = PATIENT_PRESETS[0];
