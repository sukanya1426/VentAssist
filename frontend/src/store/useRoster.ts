import { useEffect } from "react";
import { create } from "zustand";
import {
  apiErrorMessage, createPatient, deletePatient, listPatients,
} from "../api/client";
import type { Patient, PatientCreate } from "../types/patient";

/**
 * The roster lives in MongoDB — `GET /api/patients` is the source of truth for
 * every bed, the six seeded presets included. Uploading POSTs a patient, deleting
 * a card DELETEs it (taking that patient's saved recommendations with it), and both
 * update the local copy from the server's response so the grid never drifts from
 * what is actually stored.
 */
interface RosterState {
  patients: Patient[];
  loading: boolean;
  loaded: boolean;               // a load has completed at least once
  error: string | null;
  load: (force?: boolean) => Promise<void>;
  addPatient: (p: PatientCreate) => Promise<Patient>;
  removePatient: (id: string) => Promise<number>;
}

export const useRoster = create<RosterState>()((set, get) => ({
  patients: [],
  loading: false,
  loaded: false,
  error: null,

  load: async (force = false) => {
    // Every card mounts against the same roster; without this the grid would
    // refetch once per card on the first render.
    if (get().loading || (get().loaded && !force)) return;
    set({ loading: true, error: null });
    try {
      set({ patients: await listPatients(), loading: false, loaded: true });
    } catch (e) {
      set({
        error: apiErrorMessage(e, "Could not load the roster"),
        loading: false,
        loaded: true,
      });
    }
  },

  addPatient: async (p) => {
    const saved = await createPatient(p);
    set((s) => ({
      // A re-upload of the same id updates in place rather than duplicating.
      patients: s.patients.some((x) => x.id === saved.id)
        ? s.patients.map((x) => (x.id === saved.id ? saved : x))
        : [...s.patients, saved],
      error: null,
    }));
    return saved;
  },

  removePatient: async (id) => {
    const { recommendations_deleted } = await deletePatient(id);
    set((s) => ({ patients: s.patients.filter((x) => x.id !== id), error: null }));
    return recommendations_deleted;
  },
}));

/** The roster, loading it on first mount. */
export function useAllPatients(): Patient[] {
  const patients = useRoster((s) => s.patients);
  const load = useRoster((s) => s.load);
  useEffect(() => { load(); }, [load]);
  return patients;
}

/**
 * Resolve one patient from the roster, loading it first when the page was opened
 * directly on a patient URL. `undefined` while loading, so callers should check
 * `useRoster.loaded` before showing a "no such patient" message.
 */
export function usePatientById(id: string | undefined): Patient | undefined {
  const patients = useAllPatients();
  return patients.find((p) => p.id === id);
}
