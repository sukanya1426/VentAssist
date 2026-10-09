INPUT 4 — TIER A: nine behaviours, one file each
================================================

Upload ONE of these on the roster's upload card. All nine are SYNTHETIC: the
twelve clinical values were written by hand to exercise one specific policy
behaviour each. No MIMIC-IV record is reproduced in this folder, so unlike
folders 1-3 these files carry no credentialed-data restriction and can be shown
in a slide deck, a demo or a viva without any data-use concern.

Every recommendation below is the MEASURED output of the deployed model
(backend/models/policy_track_a.pt) at responsiveness 0.0, not an expectation.

  E1  patient-stable-hold.txt
      Everything at target, 6.1 mL/kg
      -> Hold PEEP | Hold TV | Hold FiO2            confidence 0.567, no flags
      The one to open FIRST. "Hold all three" is one of the 125 actions, not a
      failure to answer.

  E2  patient-high-peep.txt
      PEEP 18 with SpO2 96 — pressure costing more than it buys
      -> Decrease PEEP by 2 cmH2O                   confidence 0.645
         CRITICAL: resulting PEEP 16 still exceeds 15
      The hardest reflex to learn, and correct AND still flagged: one step does
      not fix it, and the system does not pretend otherwise.

  E3  patient-hyperoxic.txt
      SpO2 100 on FiO2 0.70 — oxygen to wean
      -> Decrease FiO2 by 0.10                      confidence 0.481, no flags

  E4  patient-hypocapnic.txt
      Over-ventilated: PaCO2 28, pH 7.52, TV exactly at the 8 mL/kg ceiling
      -> Decrease TV by 50 mL                       confidence 0.602, no flags
      No rule is violated, so no flag fires — the policy still backs off, for a
      reason no safety rule encodes.

  E5  patient-low-peep-high-fio2.txt
      PEEP 5 on FiO2 0.85 — buying oxygenation with oxygen, not recruitment
      -> Increase PEEP by 2 cmH2O                   confidence 0.466
         WARNING: projected FiO2 0.85 exceeds the 0.80 toxicity threshold
      Policy and safety filter agree from two independent directions.

  E6 + E7  THE MODE-MASKING PAIR — open both and compare
      patient-hypercapnic-volume-control.txt   and
      patient-hypercapnic-pressure-control.txt
      share the SAME twelve clinical values. Only ventilation_mode differs.
      E6 (volume control)   -> Increase TV by 25 mL      confidence 0.487
      E7 (pressure control) -> Hold all three            confidence 0.514,
                               tidal-volume actions MASKED
      In pressure control, tidal volume is a RESULT of the set pressure, not a
      setting, so a "+25 mL" order cannot be carried out at the bedside. The
      system masks those actions and recommends from what is left. This is the
      clearest two-file demonstration in the whole sample set.

  E8  patient-two-problems.txt
      Hypoxaemic (SpO2 88) AND severely hypercapnic (PaCO2 68, pH 7.20)
      -> Increase TV by 50 mL | Increase FiO2 by 0.10   confidence 0.432
      TWO settings move at once. The single-problem files move one setting each
      because only one setting is wrong, not because the policy cannot do more.

  E9  patient-volutrauma-low-peep.txt
      10.0 mL/kg delivered on a PEEP of 3 — how ventilators cause injury
      -> Increase PEEP by 2 cmH2O | Decrease TV by 50 mL  confidence 0.496
         CRITICAL: resulting TV 670 mL still exceeds 8 mL/kg (576 mL)
      Two settings move, and the flag correctly stays on.

SUGGESTED DEMO ORDER
  E1 (it can hold) -> E2 (safety reflex + honest flag) -> E6/E7 (masking)
  -> E8 or E9 (two settings at once). Four uploads, four distinct capabilities.

WHY CONFIDENCE IS AROUND 0.5
Confidence is a DECISION confidence derived from the Q-value margin between the
chosen action and the runners-up — not a probability that the patient does well.
Among 125 actions, several of which are clinically reasonable, a margin-based
score near 0.5 is the honest reading. E2's 0.645 is high because "lower PEEP" is
clearly separated here; E8's 0.432 is lower because several reasonable
combinations compete.
