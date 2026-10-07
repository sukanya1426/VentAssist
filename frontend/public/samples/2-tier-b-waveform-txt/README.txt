INPUT 2 — TIER B: clinical data + waveform (text format)
========================================================

Upload BOTH files TOGETHER on the upload card (select or drag both at once).

  patient-hyperoxic.txt   the 12 clinical values + weight (NO waveform block)
  waveform-10min.txt      10 minutes of real waveform signal, human-readable

THIS IS A DIFFERENT PATIENT FROM FOLDERS 1 AND 3, so the recommendation is
different too. Previously all three folders shipped the same patient file, which
made it look as though the engine returned the same answer no matter the input.

  Patient: FiO2 1.00 with SpO2 99.5 -> over-oxygenated; PEEP only 5.
  Expected: Increase PEEP by 2 cmH2O | Hold tidal volume | Decrease FiO2 by 0.10
            confidence 0.635
            WARNING: projected FiO2 0.90 exceeds the oxygen-toxicity threshold
            (no arrhythmia flag -- the extracted Arrhythmia_rate is 0.00)

The waveform file is plain text you can open and read:

    fs: 62.4725
    channels: II,Pleth,Resp
    signal:
    ... 37,483 rows

On upload the backend runs the SAME extractors the training features came from
(backend/waveform/) and derives all 6 Track B features from the samples:

    HRV_SDNN 36.25   Arrhythmia_rate 0.00   Perfusion_Index 30.00
    RRV 0.1632       Breathing_Regularity 0.8368   Asynchrony_Score 0.0000
    -> 6/6 features, 100% coverage, 600 s at 62.47 Hz

SOURCE: MIMIC-IV-WDB subject p15342703, record 86380383, ICU stay 31066662,
hour 0 of that patient's ventilation window. The clinical file and the waveform
are the SAME patient at the SAME hour.

EXPECT THIS: the recommendation and confidence match what Tier A would give for
the same clinical values, and the label reads "waveform recorded, not yet
influencing". That is honest, not a bug -- the trained model assigns the waveform
dimensions zero weight on a 35-patient cohort.
