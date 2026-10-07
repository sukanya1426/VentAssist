INPUT 1 — TIER A: clinical data only
====================================

Upload ONE of these files on the roster's upload card.

Each is a real MIMIC-IV patient state: 12 clinical values + weight, no waveform.
Track A is the full-strength policy (trained on 992,100 transitions), so these
are the recommendations the system is actually validated on.

  patient-volutrauma.txt    tidal volume 789 mL at 80.1 kg = 9.85 mL/kg
                            -> Hold PEEP | Decrease TV by 50 mL | Hold FiO2
                               confidence 0.605, CRITICAL safety flag
                               (resulting TV 739 mL still exceeds 8 mL/kg)

  patient-hypoxaemic.txt    SpO2 91 on FiO2 0.50, PEEP 5
                            -> Increase PEEP by 2 cmH2O | Hold TV | Hold FiO2
                               confidence 0.612, no safety flags

  patient-hypercapnic.txt   PaCO2 66 on a low tidal volume of 330 mL
                            -> Hold PEEP | Increase TV by 50 mL | Hold FiO2
                               confidence 0.448

WHY ONLY ONE SETTING MOVES
One recommendation is a TRIPLE (dPEEP, dTV, dFiO2) chosen from 125 combinations.
"Hold" is a real choice, not a missing answer. In each case above only one
setting is clinically wrong, so only one moves. Patients with two problems do
get two settings changed at once.
