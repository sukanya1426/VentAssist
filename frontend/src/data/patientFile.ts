import type { Patient, WaveformFeatures } from "./patients";
import type { TabularState } from "../types/recommendation";

/**
 * Parser for uploaded patient files.
 *
 * Format is `key: value`, one per line, `#` starts a comment — a plain text file
 * a clinician can write by hand or export from a chart. A JSON object with the
 * same keys is also accepted.
 *
 * The 12 required clinical fields and their bounds mirror `TabularState` in
 * backend/api/models.py exactly, so a file that parses here will not be rejected
 * by the API with a 422.
 */

export interface FieldSpec {
  key: keyof TabularState;
  aliases: string[];
  min: number;
  max: number;
  unit: string;
  integer?: boolean;
}

// Bounds copied from backend/api/models.py :: TabularState. Shared with the
// patient page so an edit is rejected with a readable message before it is POSTed.
export const CLINICAL_FIELDS: FieldSpec[] = [
  { key: "PEEP",  aliases: ["peep", "peep_cmh2o", "peep_set"], min: 0, max: 30, unit: "cmH₂O" },
  { key: "TV",    aliases: ["tv", "tidal_volume", "vt", "tidal_vol"], min: 100, max: 1200, unit: "mL" },
  { key: "FiO2",  aliases: ["fio2", "fio_2", "inspired_o2", "fio2_fraction"], min: 0.21, max: 1.0, unit: "fraction" },
  { key: "SpO2",  aliases: ["spo2", "sao2", "o2_sat", "oxygen_saturation"], min: 50, max: 100, unit: "%" },
  { key: "PaO2",  aliases: ["pao2", "pa_o2"], min: 30, max: 700, unit: "mmHg" },
  { key: "PaCO2", aliases: ["paco2", "pa_co2"], min: 10, max: 120, unit: "mmHg" },
  { key: "pH",    aliases: ["ph", "arterial_ph"], min: 6.8, max: 7.8, unit: "" },
  { key: "HR",    aliases: ["hr", "heart_rate", "pulse"], min: 20, max: 250, unit: "bpm" },
  { key: "SBP",   aliases: ["sbp", "systolic_bp", "systolic", "map_systolic"], min: 50, max: 250, unit: "mmHg" },
  { key: "RR",    aliases: ["rr", "respiratory_rate", "resp_rate"], min: 4, max: 60, unit: "/min" },
  { key: "RASS",  aliases: ["rass", "sedation_score"], min: -5, max: 4, unit: "", integer: true },
  { key: "Temp",  aliases: ["temp", "temperature", "temp_c"], min: 30, max: 43, unit: "°C" },
];

const WAVEFORM_FIELDS: { key: keyof WaveformFeatures; aliases: string[]; min: number; max: number }[] = [
  { key: "HRV_SDNN", aliases: ["hrv_sdnn", "hrv", "sdnn"], min: 0, max: 500 },
  { key: "Arrhythmia_rate", aliases: ["arrhythmia_rate", "arrhythmia"], min: 0, max: 1 },
  { key: "Perfusion_Index", aliases: ["perfusion_index", "perfusion", "pi"], min: 0, max: 30 },
  { key: "RRV", aliases: ["rrv", "resp_rate_variability"], min: 0, max: 10 },
  { key: "Breathing_Regularity", aliases: ["breathing_regularity", "breath_regularity"], min: 0, max: 1 },
  { key: "Asynchrony_Score", aliases: ["asynchrony_score", "asynchrony"], min: 0, max: 1 },
];

export const WEIGHT = { aliases: ["weight", "patient_weight", "weight_kg", "body_weight"], min: 30, max: 300 };

export interface ParseResult {
  patient?: Patient;
  errors: string[];
  warnings: string[];
}

const norm = (k: string) => k.trim().toLowerCase().replace(/[\s\-.]+/g, "_");

/** `key: value` lines, `#` comments, or a JSON object → flat map of normalised keys. */
function toMap(text: string): Record<string, string> {
  const trimmed = text.trim();
  const out: Record<string, string> = {};

  if (trimmed.startsWith("{")) {
    const flat = (obj: any, prefix = "") => {
      for (const [k, v] of Object.entries(obj ?? {})) {
        if (v && typeof v === "object" && !Array.isArray(v)) flat(v, prefix);
        else if (v != null) out[norm(k)] = String(v);
      }
    };
    flat(JSON.parse(trimmed));
    return out;
  }

  for (const raw of trimmed.split(/\r?\n/)) {
    const line = raw.split("#")[0].trim();
    if (!line) continue;
    const m = line.match(/^([^:=]+)[:=](.*)$/);
    if (!m) continue;
    const value = m[2].trim().replace(/^["']|["']$/g, "");
    if (value !== "") out[norm(m[1])] = value;
  }
  return out;
}

/** First alias present in the map. */
function pick(map: Record<string, string>, aliases: string[]): string | undefined {
  for (const a of aliases) if (map[a] != null) return map[a];
  return undefined;
}

/** Strip a trailing unit ("0.5 fraction", "37.0 C", "95%") before parsing. */
function toNumber(raw: string): number {
  return parseFloat(raw.replace(/[^0-9eE+\-.]/g, ""));
}

export function parsePatientFile(text: string, fileName = "upload"): ParseResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  let map: Record<string, string>;
  try {
    map = toMap(text);
  } catch {
    return { errors: ["File is not valid JSON and has no `key: value` lines."], warnings };
  }
  if (Object.keys(map).length === 0) {
    return { errors: ["No `key: value` lines found — see the template for the format."], warnings };
  }

  // --- 12 required clinical fields ---
  const state = {} as TabularState;
  for (const f of CLINICAL_FIELDS) {
    const raw = pick(map, f.aliases);
    if (raw == null) {
      errors.push(`Missing required field "${f.key}"${f.unit ? ` (${f.unit})` : ""}`);
      continue;
    }
    const v = toNumber(raw);
    if (Number.isNaN(v)) {
      errors.push(`"${f.key}" is not a number (got "${raw}")`);
    } else if (v < f.min || v > f.max) {
      errors.push(`"${f.key}" = ${v} is outside the accepted range ${f.min}–${f.max}${f.unit ? ` ${f.unit}` : ""}`);
    } else {
      state[f.key] = f.integer ? Math.round(v) : v;
    }
  }

  // --- weight (required: the safety filter needs mL/kg) ---
  const rawWeight = pick(map, WEIGHT.aliases);
  let weight = NaN;
  if (rawWeight == null) {
    errors.push('Missing required field "weight" (kg) — needed for the mL/kg safety check');
  } else {
    weight = toNumber(rawWeight);
    if (Number.isNaN(weight) || weight < WEIGHT.min || weight > WEIGHT.max) {
      errors.push(`"weight" must be a number between ${WEIGHT.min} and ${WEIGHT.max} kg (got "${rawWeight}")`);
    }
  }

  // --- optional waveform block (enables Track B) ---
  const waveform: WaveformFeatures = {};
  for (const f of WAVEFORM_FIELDS) {
    const raw = pick(map, f.aliases);
    if (raw == null) continue;
    const v = toNumber(raw);
    if (Number.isNaN(v) || v < f.min || v > f.max) {
      warnings.push(`Ignored waveform field "${f.key}" — must be between ${f.min} and ${f.max}`);
      continue;
    }
    (waveform as any)[f.key] = v;
  }
  const hasWaveform = Object.keys(waveform).length > 0;

  // --- optional identity / context ---
  const name = pick(map, ["name", "patient_name", "patient", "label"])
    ?? fileName.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ");
  const sexRaw = (pick(map, ["sex", "gender"]) ?? "").toUpperCase();
  const ageRaw = pick(map, ["age", "age_years"]);
  const mode = pick(map, ["ventilation_mode", "vent_mode", "mode"]);
  const trackRaw = pick(map, ["track", "policy_track"]);

  let ventilationMode: string | undefined;
  if (mode) {
    const m = norm(mode);
    if (["volume_control", "vc", "volume"].includes(m)) ventilationMode = "volume_control";
    else if (["pressure_control", "pc", "pressure"].includes(m)) ventilationMode = "pressure_control";
    else warnings.push(`Unrecognised ventilation_mode "${mode}" — treated as unknown`);
  }

  let track: Patient["track"];
  if (trackRaw) {
    const t = norm(trackRaw);
    if (t.includes("b")) {
      if (hasWaveform) track = "track_b";
      else warnings.push("track_b requested but the file has no waveform values — opening on Track A");
    } else track = "track_a";
  }

  if (errors.length) return { errors, warnings };

  const patient: Patient = {
    id: `upload-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
    name: name.trim(),
    weight,
    age: ageRaw && !Number.isNaN(toNumber(ageRaw)) ? Math.round(toNumber(ageRaw)) : 0,
    sex: sexRaw.startsWith("F") ? "F" : "M",
    bed: pick(map, ["bed", "bed_id", "location", "unit"]) ?? "Uploaded",
    summary: pick(map, ["summary", "notes", "diagnosis", "history"]) ?? `Uploaded from ${fileName}`,
    state,
    source: "upload",
    waveform: hasWaveform ? waveform : undefined,
    ventilationMode,
    track: track ?? (hasWaveform ? "track_b" : "track_a"),
    uploadedAt: new Date().toISOString(),
  };
  return { patient, errors, warnings };
}

/** The blank template offered for download in the upload card. */
export const PATIENT_FILE_TEMPLATE = `# VentAssist patient file
# One "key: value" per line. Lines starting with # are ignored.
# The 12 clinical fields and weight are REQUIRED — everything else is optional.

# ---- identity (optional) ----
name: Patient X
age: 60
sex: M
bed: ICU-07
summary: one-line clinical picture

# ---- required ----
weight: 75            # kg, 30-300 (used for the mL/kg safety check)

PEEP: 8               # cmH2O, 0-30
TV: 460               # mL, 100-1200
FiO2: 0.40            # fraction, 0.21-1.0
SpO2: 95              # %, 50-100
PaO2: 88              # mmHg, 30-700
PaCO2: 40             # mmHg, 10-120
pH: 7.40              # 6.8-7.8
HR: 84                # bpm, 20-250
SBP: 120              # mmHg, 50-250
RR: 16                # breaths/min, 4-60
RASS: -1              # -5 to 4 (integer)
Temp: 37.0            # deg C, 30-43

# ---- optional context ----
ventilation_mode: volume_control   # volume_control | pressure_control
track: track_a                     # track_a (clinical) | track_b (needs waveform below)

# ---- optional waveform block (enables Track B) ----
# HRV_SDNN: 31.2
# Arrhythmia_rate: 0.03
# Perfusion_Index: 2.1
# RRV: 0.19
# Breathing_Regularity: 0.81
# Asynchrony_Score: 0.10
`;

/**
 * Validate an edited state before it is sent to `/api/recommend`, using the same
 * bounds the API enforces. Returns a list of human-readable problems (empty = OK)
 * so a cleared or out-of-range field produces a named error instead of a 422.
 */
export function validateTabular(state: TabularState, weight: number): string[] {
  const problems: string[] = [];
  for (const f of CLINICAL_FIELDS) {
    const v = state[f.key];
    const unit = f.unit ? ` ${f.unit}` : "";
    if (v == null || Number.isNaN(v)) problems.push(`${f.key} is empty`);
    else if (v < f.min || v > f.max) {
      problems.push(`${f.key} ${v} is outside ${f.min}–${f.max}${unit}`);
    }
  }
  if (weight == null || Number.isNaN(weight) || weight < WEIGHT.min || weight > WEIGHT.max) {
    problems.push(`weight must be ${WEIGHT.min}–${WEIGHT.max} kg`);
  }
  return problems;
}

/** Per-field validity, for highlighting the offending input. */
export function fieldIsValid(key: keyof TabularState, v: number): boolean {
  const f = CLINICAL_FIELDS.find((x) => x.key === key);
  if (!f) return true;
  return v != null && !Number.isNaN(v) && v >= f.min && v <= f.max;
}
