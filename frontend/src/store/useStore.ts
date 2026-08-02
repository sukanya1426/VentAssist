import { create } from "zustand";
import { apiErrorMessage, getRecommendation } from "../api/client";
import { DEFAULT_PRESET } from "../data/patientPresets";
import { validateTabular } from "../data/patientFile";
import type { Patient } from "../types/patient";
import type {
  RecommendationRequest, RecommendationResponse, TabularState, Track,
} from "../types/recommendation";

const DEFAULT_STATE: TabularState = { ...DEFAULT_PRESET.state };

const DEFAULT_WAVEFORM = {
  ecgHRV: 31.2, ecgArr: 0.03, pleth: 2.1,
  rrv: 0.19, breathReg: 0.81, asynchrony: 0.1,
};

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
  ecgHRV: number; ecgArr: number; pleth: number;
  rrv: number; breathReg: number; asynchrony: number;
  result: RecommendationResponse | null;
  resultState: TabularState | null;   // the state `result` was computed from
  loading: boolean;
  error: string | null;
  // Bumped whenever a recommendation is saved to MongoDB, so the history panel
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
    waveformAvailable: true,
    responsiveness: 0,
    ventilationMode: "unknown",
    ...DEFAULT_WAVEFORM,
    result: null,
    resultState: null,
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
      // Uploads carry their own waveform values; presets fall back to the demo set.
      const waveform = w ? {
        ecgHRV: w.HRV_SDNN ?? DEFAULT_WAVEFORM.ecgHRV,
        ecgArr: w.Arrhythmia_rate ?? DEFAULT_WAVEFORM.ecgArr,
        pleth: w.Perfusion_Index ?? DEFAULT_WAVEFORM.pleth,
        rrv: w.RRV ?? DEFAULT_WAVEFORM.rrv,
        breathReg: w.Breathing_Regularity ?? DEFAULT_WAVEFORM.breathReg,
        asynchrony: w.Asynchrony_Score ?? DEFAULT_WAVEFORM.asynchrony,
      } : DEFAULT_WAVEFORM;
      // A preset patient has no recorded waveform, but the demo values let the
      // clinician explore Track B; an upload only offers it if the file had one.
      const waveformAvailable = p.source === "preset" || !!w;
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
