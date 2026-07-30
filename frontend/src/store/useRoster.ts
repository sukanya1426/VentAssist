import { create } from "zustand";
import { persist } from "zustand/middleware";
import { PATIENTS, type Patient } from "../data/patients";

/**
 * The roster = the built-in preset patients plus any the clinician has uploaded.
 * Uploads are persisted to localStorage so a page refresh (or opening a patient
 * URL directly) still resolves them.
 */
interface RosterState {
  uploaded: Patient[];
  addPatient: (p: Patient) => void;
  removePatient: (id: string) => void;
}

export const useRoster = create<RosterState>()(
  persist(
    (set) => ({
      uploaded: [],
      addPatient: (p) => set((s) => ({ uploaded: [...s.uploaded, p] })),
      removePatient: (id) => set((s) => ({ uploaded: s.uploaded.filter((x) => x.id !== id) })),
    }),
    { name: "ventassist-uploaded-patients" }
  )
);

/** Preset patients first, then uploads in the order they were added. */
export function useAllPatients(): Patient[] {
  const uploaded = useRoster((s) => s.uploaded);
  return [...PATIENTS, ...uploaded];
}

export function usePatientById(id: string | undefined): Patient | undefined {
  return useAllPatients().find((p) => p.id === id);
}
