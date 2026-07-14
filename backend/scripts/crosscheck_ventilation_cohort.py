"""Cross-check the VentAssist cohort against the multi-signal ventilation view
(SYSTEM_SUMMARY §16 item 5).

VentAssist derives ventilation episodes from a *single* procedure itemid
(``procedureevents`` 225792, "Invasive Ventilation"). IntelliLung instead uses
``mimiciv_derived.ventilation`` — a peer-reviewed *multi-signal* view that infers
invasive ventilation from the presence of an endotracheal tube / invasive
ventilator mode in ``ventilator_setting`` + ``oxygen_delivery`` (see the shipped
``sql/base/ventilation.sql``). That derived view is a BigQuery/Postgres object and
is **not** on disk here, so this script reconstructs its *InvasiveVent* signal
directly from the raw chartevents itemids the view is built on:

  * 226732  O2 Delivery Device(s)      → 'Endotracheal tube'  ⇒ InvasiveVent
                                          'Tracheostomy tube' / 'Trach mask' ⇒ Tracheostomy
  * 223849  Ventilator Mode            → any invasive mode    ⇒ InvasiveVent
  * 229314  Ventilator Mode (Hamilton) → any invasive Hamilton mode ⇒ InvasiveVent

It then compares three stay sets and writes a JSON report:

  A = stays with a raw procedureevents 225792 row      (our cohort source, pre-filter)
  B = stays with reconstructed InvasiveVent evidence   (the multi-signal view)
  C = the final filtered cohort (cohort.csv)

and reports the overlap (A∩B), the two disagreement sets (A\B "procedure but no
setting evidence" and B\A "setting evidence but no procedure"), plus how well the
*final* cohort C is corroborated by the settings signal. This is a read-only audit
— it changes no model or dataset.

Run:  python -m backend.scripts.crosscheck_ventilation_cohort [--max-rows N]
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("vent_crosscheck")

DEVICE_ITEMID, MODE_ITEMID, MODE_HAMILTON_ITEMID = 226732, 223849, 229314
VENT_ITEMIDS = {DEVICE_ITEMID, MODE_ITEMID, MODE_HAMILTON_ITEMID}

INVASIVE_DEVICES = {"Endotracheal tube"}
TRACH_DEVICES = {"Tracheostomy tube", "Trach mask"}      # invasive, distinct category
# Invasive ventilator modes, verbatim from ventilation.sql (stripped).
INVASIVE_MODES = {
    "(S) CMV", "APRV", "APRV/Biphasic+ApnPress", "APRV/Biphasic+ApnVol", "APV (cmv)",
    "Ambient", "Apnea Ventilation", "CMV", "CMV/ASSIST", "CMV/ASSIST/AutoFlow",
    "CMV/AutoFlow", "CPAP/PPS", "CPAP/PSV", "CPAP/PSV+Apn TCPL", "CPAP/PSV+ApnPres",
    "CPAP/PSV+ApnVol", "MMV", "MMV/AutoFlow", "MMV/PSV", "MMV/PSV/AutoFlow", "P-CMV",
    "PCV+", "PCV+/PSV", "PCV+Assist", "PRES/AC", "PRVC/AC", "PRVC/SIMV", "PSV/SBT",
    "SIMV", "SIMV/AutoFlow", "SIMV/PRES", "SIMV/PSV", "SIMV/PSV/AutoFlow", "SIMV/VOL",
    "SYNCHRON MASTER", "SYNCHRON SLAVE", "VOL/AC",
}
HAMILTON_INVASIVE_MODES = {
    "APRV", "APV (cmv)", "Ambient", "(S) CMV", "P-CMV", "SIMV", "APV (simv)",
    "P-SIMV", "VS", "ASV",
}


def _procedure_vent_stays() -> set[int]:
    """Stays with a raw procedureevents 225792 (Invasive Ventilation) row."""
    stays: set[int] = set()
    for ch in pd.read_csv(config.PROCEDUREEVENTS, usecols=["stay_id", "itemid"],
                          chunksize=1_000_000):
        stays.update(int(s) for s in ch.loc[ch.itemid == 225792, "stay_id"].dropna().unique())
    return stays


def _scan_settings(max_rows: int | None) -> tuple[set[int], set[int], int]:
    """Chunk-scan chartevents for the 3 ventilation itemids; classify InvasiveVent.

    Returns (invasive_vent_stays, tracheostomy_stays, n_rows_scanned)."""
    inv: set[int] = set()
    trach: set[int] = set()
    scanned = 0
    t0 = time.time()
    reader = pd.read_csv(config.CHARTEVENTS,
                         usecols=["stay_id", "itemid", "value"],
                         chunksize=2_000_000)
    for i, ch in enumerate(reader):
        scanned += len(ch)
        ch = ch[ch.itemid.isin(VENT_ITEMIDS)]
        if len(ch):
            val = ch["value"].astype("string").str.strip()
            dev = ch.itemid == DEVICE_ITEMID
            inv.update(int(s) for s in ch.loc[dev & val.isin(INVASIVE_DEVICES), "stay_id"].dropna().unique())
            trach.update(int(s) for s in ch.loc[dev & val.isin(TRACH_DEVICES), "stay_id"].dropna().unique())
            m1 = (ch.itemid == MODE_ITEMID) & val.isin(INVASIVE_MODES)
            m2 = (ch.itemid == MODE_HAMILTON_ITEMID) & val.isin(HAMILTON_INVASIVE_MODES)
            inv.update(int(s) for s in ch.loc[m1 | m2, "stay_id"].dropna().unique())
        if (i + 1) % 10 == 0:
            log.info("  scanned %d chunks (%d rows, %.0fs) | invasive=%d trach=%d",
                     i + 1, scanned, time.time() - t0, len(inv), len(trach))
        if max_rows is not None and scanned >= max_rows:
            log.info("  stopping early at max_rows=%d", max_rows)
            break
    return inv, trach, scanned


def cross_check(max_rows: int | None = None) -> dict:
    log.info("Loading procedureevents 225792 vent stays (set A)…")
    A = _procedure_vent_stays()
    log.info("  |A| procedure-vent stays = %d", len(A))

    log.info("Scanning chartevents for ventilator settings/device (set B)…")
    B, trach, scanned = _scan_settings(max_rows)
    log.info("  |B| InvasiveVent-by-settings stays = %d (+%d tracheostomy)", len(B), len(trach))

    cohort_path = config.PROCESSED_PATH / "cohort.csv"
    C = set()
    if cohort_path.exists():
        C = set(int(s) for s in pd.read_csv(cohort_path, usecols=["stay_id"])["stay_id"].unique())

    inter = A & B
    a_not_b = A - B                       # procedure row but no invasive settings evidence
    b_not_a = B - A                       # settings evidence but no procedure row
    B_all = B | trach                     # any invasive (incl. tracheostomy)

    def pct(n, d):
        return round(100.0 * n / d, 2) if d else None

    report = {
        "partial_scan": max_rows is not None,
        "chartevents_rows_scanned": int(scanned),
        "A_procedure_225792_stays": len(A),
        "B_invasive_by_settings_stays": len(B),
        "tracheostomy_by_settings_stays": len(trach),
        "A_and_B": len(inter),
        "A_not_B_procedure_only": len(a_not_b),
        "B_not_A_settings_only": len(b_not_a),
        "jaccard_A_B": round(len(inter) / len(A | B), 4) if (A | B) else None,
        "pct_of_A_corroborated_by_settings": pct(len(inter), len(A)),
        "pct_of_A_corroborated_incl_trach": pct(len(A & B_all), len(A)),
        "final_cohort_stays": len(C),
        "pct_final_cohort_with_settings_evidence": pct(len(C & B_all), len(C)) if C else None,
    }
    (config.LOGS_PATH / "ventilation_crosscheck.json").write_text(json.dumps(report, indent=2))
    log.info("Cross-check report → logs/ventilation_crosscheck.json")
    log.info("  A∩B=%d | A-only=%d | B-only=%d | Jaccard=%.3f",
             report["A_and_B"], report["A_not_B_procedure_only"],
             report["B_not_A_settings_only"], report["jaccard_A_B"] or 0.0)
    log.info("  %% of procedure-vent stays corroborated by settings (incl. trach) = %s%%",
             report["pct_of_A_corroborated_incl_trach"])
    if C:
        log.info("  %% of FINAL cohort corroborated by settings (incl. trach) = %s%%",
                 report["pct_final_cohort_with_settings_evidence"])
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap chartevents rows scanned (for a quick partial run)")
    args = ap.parse_args()
    cross_check(max_rows=args.max_rows)


if __name__ == "__main__":
    main()
