INPUT 3 — TIER B: clinical data + waveform (WFDB record, the dataset's own format)
==================================================================================

Upload ALL THREE files TOGETHER on the upload card (select or drag all at once).

  patient-lowtv.txt       the 12 clinical values + weight (NO waveform block)
  waveform-10min.hea      WFDB header: channel names, gains, sampling rate
  waveform-10min.dat      WFDB signal: the samples themselves (binary)

THIS IS A DIFFERENT PATIENT FROM FOLDERS 1 AND 2, so the recommendation is
different too.

  Patient: SpO2 92 on FiO2 0.60 with a tidal volume of only 300 mL at 86.5 kg.
  Expected: Increase PEEP by 2 cmH2O | Increase tidal volume by 50 mL | Hold FiO2
            confidence 0.550
            no safety flags

WHY THIS FOLDER EXISTS SEPARATELY FROM FOLDER 2
Folder 2 ships the waveform as readable text, which is good for seeing what a
recording contains but involves a transcription step. This folder is the format
MIMIC-IV-WDB actually distributes, so it is the path a clinician handing over a
trimmed record folder would really take. The backend reads it with the same
loader the training cohort was built with.

On upload the backend runs the SAME extractors the training features came from
(backend/waveform/) and derives all 6 Track B features from the record:

    HRV_SDNN 8.68    Arrhythmia_rate 0.00   Perfusion_Index 30.00
    RRV 0.2899       Breathing_Regularity 0.7101   Asynchrony_Score 0.4000
    -> 6/6 features, 100% coverage, 600 s at 62.47 Hz

DO NOT RENAME THE .hea / .dat
The header file stores its own signal filename, so renaming one and not the
other breaks the record. Re-export instead (backend/scripts/export_waveform.py).

SOURCE: MIMIC-IV-WDB subject p17490822, record 85814172, ICU stay 30199016,
hour 9 of that patient's ventilation window. The clinical file and the waveform
are the SAME patient at the SAME hour.

EXPECT THIS: the recommendation and confidence match what Tier A would give for
the same clinical values, and the label reads "waveform recorded, not yet
influencing". That is honest, not a bug -- the trained model assigns the waveform
dimensions zero weight on a 35-patient cohort.
