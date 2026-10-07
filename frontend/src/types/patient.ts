import type { TabularState, Track } from "./recommendation";

/** The 6 waveform features Track B accepts (all optional). */
/** Why one channel of an uploaded recording did or did not yield its features. */
export interface WaveformChannelReport {
  present: boolean;
  valid_fraction?: number | null;   // fraction finite before gap-filling
  quality_ok?: boolean | null;      // passed flatline / motion / disconnect
  note?: string | null;
}

/**
 * What the server made of an uploaded waveform.
 *
 * `features` may be partly empty — a recording with a disconnected Resp belt
 * still gives usable ECG and Pleth features — so `coverage` is the fraction of
 * the 6 that came out and `warnings` explains the gaps in words.
 */
export interface WaveformExtraction {
  features: WaveformFeatures;
  fs: number;
  n_samples: number;
  duration_s: number;
  coverage: number;
  channels: Record<string, WaveformChannelReport>;
  warnings: string[];
}

export interface WaveformFeatures {
  HRV_SDNN?: number;
  Arrhythmia_rate?: number;
  Perfusion_Index?: number;
  RRV?: number;
  Breathing_Regularity?: number;
  Asynchrony_Score?: number;
}

/**
 * A patient on the roster, as stored in PostgreSQL and returned by `GET /api/patients`.
 * Field names match the API exactly (snake_case for the two multi-word fields) so
 * a patient can be round-tripped without a translation layer.
 */
export interface Patient {
  id: string;                       // route param, e.g. "patient-a"
  name: string;
  weight: number;                   // kg → patient_weight
  age: number;
  sex: "M" | "F";
  bed: string;
  summary: string;
  state: TabularState;
  source: "preset" | "upload";
  hint?: string | null;             // what the policy is expected to suggest
  waveform?: WaveformFeatures | null;
  ventilation_mode?: string | null;
  track?: Track | null;
  created_at?: string | null;
  updated_at?: string | null;
  recommendation_count?: number;    // saved recommendations held for them
}

/** Body of `POST /api/patients` — everything but the server-assigned fields. */
export type PatientCreate = Omit<
  Patient, "id" | "created_at" | "updated_at" | "recommendation_count"
> & { id?: string };

export interface SafetyFlagRecord {
  level: string;
  message: string;
}

/** One saved ask: the settings it was made with and the answer it produced. */
export interface RecommendationRecord {
  id: string;
  patient_id: string;
  patient_name?: string | null;
  clinician?: string | null;        // who was signed in when it was asked
  created_at: string;
  track: string;
  patient_weight: number;
  responsiveness: number;
  ventilation_mode?: string | null;
  state: TabularState;
  waveform?: WaveformFeatures | null;
  delta_PEEP: number;
  delta_TV: number;
  delta_FiO2: number;
  action_text: string;
  confidence?: number | null;
  safety_all_clear: boolean;
  safety_flags: SafetyFlagRecord[];
  latency_ms?: number | null;
}
