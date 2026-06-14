import { create } from "zustand";
import { getRecommendation } from "../api/client";
import type {
  RecommendationRequest, RecommendationResponse, TabularState, Track,
} from "../types/recommendation";

const DEFAULT_STATE: TabularState = {
  PEEP: 8, TV: 480, FiO2: 0.5, SpO2: 91, PaO2: 68, PaCO2: 44, pH: 7.37,
  HR: 92, SBP: 118, RR: 22, RASS: -2, Temp: 37.2,
};

interface AppState {
  patientId: string;
  patientWeight: number;
  tabular: TabularState;
  selectedTrack: Track;
  waveformAvailable: boolean;
  ecgHRV: number; ecgArr: number; pleth: number;
  rrv: number; breathReg: number; asynchrony: number;
  result: RecommendationResponse | null;
  loading: boolean;
  error: string | null;
  setField: (k: keyof TabularState, v: number) => void;
  setMeta: (p: Partial<Pick<AppState, "patientId" | "patientWeight" | "selectedTrack" | "waveformAvailable" | "ecgHRV" | "ecgArr" | "pleth" | "rrv" | "breathReg" | "asynchrony">>) => void;
  fetch: () => Promise<void>;
}

export const useStore = create<AppState>((set, get) => ({
  patientId: "demo-001",
  patientWeight: 74,
  tabular: { ...DEFAULT_STATE },
  selectedTrack: "track_a",
  waveformAvailable: true,
  ecgHRV: 31.2, ecgArr: 0.03, pleth: 2.1,
  rrv: 0.19, breathReg: 0.81, asynchrony: 0.1,
  result: null,
  loading: false,
  error: null,
  setField: (k, v) => set((s) => ({ tabular: { ...s.tabular, [k]: v } })),
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
