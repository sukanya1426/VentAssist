import { create } from "zustand";
import { getRecommendation } from "../api/client";
import {
  CUSTOM_LABEL, DEFAULT_PRESET, PATIENT_PRESETS,
} from "../data/patientPresets";
import type {
  RecommendationRequest, RecommendationResponse, TabularState, Track,
} from "../types/recommendation";

const DEFAULT_STATE: TabularState = { ...DEFAULT_PRESET.state };

interface AppState {
  patientId: string;
  patientWeight: number;
  tabular: TabularState;
  presetKey: string;          // key of the active preset, or CUSTOM_LABEL when edited
  selectedTrack: Track;
  waveformAvailable: boolean;
  responsiveness: number;
  ventilationMode: string;   // "unknown" | "volume_control" | "pressure_control"
  ecgHRV: number; ecgArr: number; pleth: number;
  rrv: number; breathReg: number; asynchrony: number;
  result: RecommendationResponse | null;
  loading: boolean;
  error: string | null;
  setField: (k: keyof TabularState, v: number) => void;
  loadPreset: (key: string) => void;
  setMeta: (p: Partial<Pick<AppState, "patientId" | "patientWeight" | "selectedTrack" | "waveformAvailable" | "responsiveness" | "ventilationMode" | "ecgHRV" | "ecgArr" | "pleth" | "rrv" | "breathReg" | "asynchrony">>) => void;
  fetch: () => Promise<void>;
}

export const useStore = create<AppState>((set, get) => ({
  patientId: "demo-001",
  patientWeight: 74,
  tabular: { ...DEFAULT_STATE },
  presetKey: DEFAULT_PRESET.key,
  selectedTrack: "track_a",
  waveformAvailable: true,
  responsiveness: 0,
  ventilationMode: "unknown",
  ecgHRV: 31.2, ecgArr: 0.03, pleth: 2.1,
  rrv: 0.19, breathReg: 0.81, asynchrony: 0.1,
  result: null,
  loading: false,
  error: null,
  // editing any field detaches from the named preset
  setField: (k, v) => set((s) => ({
    tabular: { ...s.tabular, [k]: v }, presetKey: CUSTOM_LABEL,
  })),
  loadPreset: (key) => {
    const p = PATIENT_PRESETS.find((x) => x.key === key);
    if (!p) return;
    set({ tabular: { ...p.state }, presetKey: p.key, result: null });
  },
  setMeta: (p) => set(p),
  fetch: async () => {
    const s = get();
    set({ loading: true, error: null });
    try {
      const req: RecommendationRequest = {
        patient_id: s.patientId,
        patient_weight: s.patientWeight,
        track: s.selectedTrack,
        tabular_state: s.tabular,
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
      set({ result, loading: false });
    } catch (e: any) {
      set({ error: e?.response?.data?.detail?.toString() ?? e.message, loading: false });
    }
  },
}));
