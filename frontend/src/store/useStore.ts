import { create } from "zustand";
import { apiErrorMessage, getRecommendation } from "../api/client";
import { DEFAULT_PRESET } from "../data/patientPresets";
import { validateTabular } from "../data/patientFile";
import type { Patient } from "../types/patient";
import type {
  RecommendationRequest, RecommendationResponse, TabularState, Track,
} from "../types/recommendation";

const DEFAULT_STATE: TabularState = { ...DEFAULT_PRESET.state };

/**
 * No waveform data = no waveform numbers. Every field is null until a recording
 * supplies it.
 *
 * This used to hold six plausible-looking constants (HRV 31.2, arrhythmia 0.03,
 * …) that stood in whenever a patient had no recording — including all six
 * presets, which have none. They were then POSTed as `ecg_features` /
 * `pleth_features` / `resp_features`, so the model and the stored clinical
 * record received fabricated measurements as if they had been taken, coverage
 * always read 1.0, and `imputation_used` always read false. The backend has a
 * trained imputer for exactly this case (backend/router/feature_imputer.py) and
 * it could never run, because the client never admitted anything was missing.
 *
 * Nulls are omitted from the request instead, so the router sees real coverage
 * and imputes the rest. The recommendation is unchanged either way — the
 * deployed model's waveform influence is 0.0 — so this costs nothing and stops
 * the UI from showing invented numbers in a clinical tool.
 */
const NO_WAVEFORM = {
  ecgHRV: null, ecgArr: null, pleth: null,
  rrv: null, breathReg: null, asynchrony: null,
};

/** The non-state inputs that also determine a recommendation. */
export interface ResultMeta {
  responsiveness: number;
  ventilationMode: string;
  track: Track;
}

interface AppState {
  patientId: string;          // roster id, e.g. "patient-a"
  patientName: string;
  patientWeight: number;
  tabular: TabularState;
  edited: boolean;            // true once the loaded patient state has been changed
  selectedTrack: Track;
  waveformAvailable: boolean;
  responsiveness: number;
  ventilationMode: string;   // "unknown" | "volume_control" | "pressure_control"
  // null = not measured for this patient. Sent as absent, never as a number.
  ecgHRV: number | null; ecgArr: number | null; pleth: number | null;
  rrv: number | null; breathReg: number | null; asynchrony: number | null;
  result: RecommendationResponse | null;
  resultState: TabularState | null;   // the state `result` was computed from
  // The knobs `result` was computed with. Separate from resultState because they
  // are not patient state, but they change the recommendation just as much — and
  // without them a moved slider leaves a stale panel on screen with no warning,
  // which reads as "the slider does nothing".
  resultMeta: ResultMeta | null;
  loading: boolean;
  error: string | null;
  // Bumped whenever a recommendation is saved, so the history panel
  // knows to refetch without the two stores having to know about each other.
  savedCount: number;
  setField: (k: keyof TabularState, v: number) => void;
  loadPatient: (p: Patient) => void;
  resetPatient: () => void;
  setMeta: (p: Partial<Pick<AppState, "patientWeight" | "selectedTrack" | "waveformAvailable" | "responsiveness" | "ventilationMode" | "ecgHRV" | "ecgArr" | "pleth" | "rrv" | "breathReg" | "asynchrony">>) => void;
  fetch: () => Promise<void>;
}

export const useStore = create<AppState>((set, get) => {
  let current: Patient | null = null;   // the patient the working state came from

  return {
    patientId: "",
    patientName: "",
    patientWeight: 74,
    tabular: { ...DEFAULT_STATE },
    edited: false,
    selectedTrack: "track_a",
    waveformAvailable: false,   // until a patient with a recording is loaded
    responsiveness: 0,
    ventilationMode: "unknown",
    ...NO_WAVEFORM,
    result: null,
    resultState: null,
    resultMeta: null,
    loading: false,
    error: null,
    savedCount: 0,
    // editing any field detaches the working state from the patient's recorded state
    setField: (k, v) => set((s) => ({
      tabular: { ...s.tabular, [k]: v }, edited: true,
    })),
    loadPatient: (p) => {
      current = p;
      const w = p.waveform;
      // Only what the patient actually carries. A feature the recording did not
      // yield stays null rather than borrowing a number from somewhere else, so a
      // partial extraction stays visibly partial.
      const waveform = w ? {
        ecgHRV: w.HRV_SDNN ?? null,
        ecgArr: w.Arrhythmia_rate ?? null,
        pleth: w.Perfusion_Index ?? null,
        rrv: w.RRV ?? null,
        breathReg: w.Breathing_Regularity ?? null,
        asynchrony: w.Asynchrony_Score ?? null,
      } : NO_WAVEFORM;
      // Track B is offered only to a patient that actually has a recording.
      // Presets used to be included here so they could "explore" Track B, but
      // none of the six has a waveform, so the only thing that reached the model
      // was the fabricated constant set — which is not an exploration of Track B,
      // it is Track A plus six invented measurements. The override on the patient
      // page is still there for typing in features by hand.
      const waveformAvailable = !!w;
      set({
        patientId: p.id,
        patientName: p.name,
        patientWeight: p.weight,
        tabular: { ...p.state },
        edited: false,
        waveformAvailable,
        selectedTrack: p.track && waveformAvailable ? p.track : "track_a",
        ventilationMode: p.ventilation_mode ?? "unknown",
        ...waveform,
        result: null,
        resultState: null,
        resultMeta: null,
        error: null,
      });
    },
    // discard local edits and restore the patient's recorded state
    resetPatient: () => { if (current) get().loadPatient(current); },
    setMeta: (p) => set(p),
    fetch: async () => {
      const s = get();
      // Validate locally first: a cleared or out-of-range input would otherwise
      // reach the API as null/NaN and come back as an opaque 422.
      const problems = validateTabular(s.tabular, s.patientWeight);
      if (problems.length) {
        set({ error: `Fix before requesting — ${problems.join("; ")}`, loading: false });
        return;
      }
      set({ loading: true, error: null });
      try {
        const req: RecommendationRequest = {
          patient_id: s.patientId,
          patient_name: s.patientName,
          patient_weight: s.patientWeight,
          track: s.selectedTrack,
          // RASS is an int server-side; the rest are floats.
          tabular_state: { ...s.tabular, RASS: Math.round(s.tabular.RASS) },
          responsiveness: s.responsiveness,
          ventilation_mode: s.ventilationMode === "unknown" ? null : s.ventilationMode,
        };
        if (s.selectedTrack === "track_b") {
          // Send what was measured and nothing else. A null goes over the wire as
          // null, which the router counts as missing — so `waveform_coverage` is
          // the true fraction and the server-side imputer fills the gaps. Writing
          // a stand-in number here instead would report full coverage for data
          // nobody recorded. The three groups are always present even when every
          // value is null, because the API requires at least one group on
          // track_b; it reads their contents, not their presence.
          req.ecg_features = { HRV_SDNN: s.ecgHRV, Arrhythmia_rate: s.ecgArr };
          req.pleth_features = { Perfusion_Index: s.pleth };
          req.resp_features = {
            RRV: s.rrv, Breathing_Regularity: s.breathReg, Asynchrony_Score: s.asynchrony,
          };
        }
        const result = await getRecommendation(req);
        set((st) => ({
          result,
          resultState: { ...req.tabular_state },
          resultMeta: {
            responsiveness: s.responsiveness,
            ventilationMode: s.ventilationMode,
            track: s.selectedTrack,
          },
          loading: false,
          // Only nudge the history panel when the API actually stored the record.
          savedCount: result.record_id ? st.savedCount + 1 : st.savedCount,
        }));
      } catch (e: any) {
        set({ error: apiErrorMessage(e), loading: false });
      }
    },
  };
});
