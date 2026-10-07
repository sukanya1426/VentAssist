"""Stratified, frozen train/val/test split — and an audit of the one we ship.

IntelliLung Rule 3 (BENCHMARK_PLAN Phase 1 item 3): *"test-set difficulty must
match."* A head-to-head comparison is only meaningful if every method is scored on
a test set of the same clinical difficulty. Our production split
(``backend/mdp/dataset.py``) is a plain hash of ``stay_id``, which is unbiased in
expectation but says nothing about the realised balance of any one draw — and the
realised draw is the only one anybody reports.

This module does two separate jobs, deliberately kept apart:

**1. AUDIT the shipped split.** Stratify every stay by *episode-length quartile ×
mortality* (8 strata) and report how the shipped split's test fold deviates from
the population rate in each. This is diagnostic only. If the shipped test fold
already matches the population, the single-draw objection is answered with
evidence rather than with an appeal to the hash — which is the cheaper outcome and
the one we want to be able to state.

**2. EMIT a frozen stratified split** for benchmark use, as persisted CSVs. Within
each stratum, stays are allocated to train/val/test at the shipped split's own
proportions, so the only thing that changes is the *balance*, not the sizes. The
CSV is the contract: once written it is never regenerated silently, because a
split that moves between runs invalidates every number computed against it.

Why stratify on these two axes:
  * **episode length** — a long ventilation course is a harder, sicker, more
    heavily-intervened trajectory. Length quartile is the single best cheap proxy
    for case difficulty we can compute without the outcome.
  * **mortality** — the outcome the terminal reward encodes. A test fold with an
    unrepresentative death rate makes outcome-weighted value estimates
    incomparable across methods.

This module NEVER modifies the production split or the deployed model. The
shipped split stays authoritative for the deployed policy; this one exists so a
benchmark run can state its split was stratified and frozen.

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.split_compat --track a
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("bench_split")

RESULTS = Path(__file__).resolve().parent / "results"

# Fixed so the stratified allocation is reproducible from the stay ids alone. A
# benchmark split must be derivable by a reviewer with the same data, not just
# readable from our CSV.
SEED = 20260715
SPLITS = ("train", "val", "test")


def _per_stay(track: str) -> pd.DataFrame:
    """One row per stay: episode length, mortality, and its shipped split.

    Length is counted in MDP transitions (decision points), which is what the
    policy actually sees — not raw charted hours, which differ by gaps.
    """
    mdp = config.PROCESSED_PATH / f"mdp_track_{track}.parquet"
    if not mdp.exists():
        raise FileNotFoundError(f"{mdp} not present — build the MDP first.")
    df = pd.read_parquet(mdp, columns=["stay_id", "split"])
    g = (df.groupby("stay_id")
           .agg(n_transitions=("split", "size"), shipped_split=("split", "first"))
           .reset_index())

    out_path = config.PROCESSED_PATH / "outcomes.csv"
    if out_path.exists():
        oc = pd.read_csv(out_path)
        if "outcome_label_known" in oc.columns:
            oc = oc[oc["outcome_label_known"] == 1]
        g = g.merge(oc[["stay_id", "died_horizon"]], on="stay_id", how="left")
    else:
        g["died_horizon"] = np.nan
    return g


def _assign_strata(g: pd.DataFrame) -> pd.DataFrame:
    """Add ``died``, ``length_quartile``, ``outcome_class`` and ``stratum`` columns.

    Accepts either a ``died`` or a raw ``died_horizon`` column, and coerces it
    here rather than at the I/O boundary so the coercion is unit-testable — the
    dtype bug below was found only after it had silently halved the stratum count
    on the real cohort.
    """
    g = g.copy()
    # Coerced numerically rather than mapped through {True: 1, False: 0}: the
    # column arrives as int64 from outcomes.csv but as bool from a DataFrame built
    # in memory, and a dict with bool keys silently matches neither an int64 nor a
    # float column — it mapped every row to NaN, which read as "no outcome labels
    # available" and collapsed the 8 strata to 4.
    # astype(float) is load-bearing: to_numeric leaves a bool column as bool, and
    # mapping {1: ..., 0: ...} over a bool Series yields NaN for every row.
    src = "died" if "died" in g.columns else "died_horizon"
    g["died"] = pd.to_numeric(g[src], errors="coerce").astype(float).clip(0, 1)

    # qcut on ranks, so ties at a quartile edge cannot collapse a bin to empty.
    q = pd.qcut(g["n_transitions"].rank(method="first"), 4,
                labels=["len_q1", "len_q2", "len_q3", "len_q4"])
    # A stay with no reliable outcome label is its own class rather than being
    # dropped or silently folded in with the survivors — it contributes no
    # terminal reward either, so pretending it survived would bias the balance.
    died = g["died"].map({1: "died", 0: "survived"}).fillna("outcome_unknown")
    g["length_quartile"] = q.astype(str)
    g["outcome_class"] = died.astype(str)
    g["stratum"] = g["length_quartile"] + "|" + g["outcome_class"]
    return g


def _target_proportions(g: pd.DataFrame) -> dict[str, float]:
    """The shipped split's own stay-level proportions, so sizes are preserved."""
    counts = Counter(g["shipped_split"])
    total = sum(counts[s] for s in SPLITS)
    if total == 0:
        raise ValueError("shipped split holds none of train/val/test")
    return {s: counts[s] / total for s in SPLITS}


def _stable_order(stay_ids: np.ndarray) -> np.ndarray:
    """Deterministic, data-independent ordering of stays within a stratum.

    Hashing the stay id (rather than shuffling with a seeded RNG over whatever
    order pandas produced) means the allocation depends only on the set of stay
    ids — so a reviewer rebuilding the cohort gets the same split even if row
    order differs.
    """
    keys = [hashlib.blake2b(f"{SEED}:{int(s)}".encode(), digest_size=8).hexdigest()
            for s in stay_ids]
    return np.argsort(keys)


def _stratified_split(g: pd.DataFrame, prop: dict[str, float]) -> pd.Series:
    """Allocate each stratum across train/val/test at ``prop``, largest-remainder.

    Largest-remainder rather than rounding each independently: with 8 strata,
    independent rounding drifts the totals by enough to matter on the small ones.
    """
    assigned = pd.Series(index=g.index, dtype=object)
    for stratum, block in g.groupby("stratum", sort=True):
        ids = block["stay_id"].to_numpy()
        order = _stable_order(ids)
        n = len(ids)
        exact = {s: n * prop[s] for s in SPLITS}
        base = {s: int(np.floor(exact[s])) for s in SPLITS}
        # hand out the leftover seats to the largest fractional parts
        leftover = n - sum(base.values())
        for s in sorted(SPLITS, key=lambda s: exact[s] - base[s], reverse=True)[:leftover]:
            base[s] += 1
        labels: list[str] = []
        for s in SPLITS:
            labels += [s] * base[s]
        assigned.loc[block.index[order]] = labels
    return assigned


def _balance_table(g: pd.DataFrame, split_col: str) -> dict:
    """Per-stratum share of the test fold vs the population share."""
    pop = g["stratum"].value_counts(normalize=True)
    test = g.loc[g[split_col] == "test", "stratum"].value_counts(normalize=True)
    rows = {}
    for stratum in sorted(pop.index):
        p, t = float(pop[stratum]), float(test.get(stratum, 0.0))
        rows[stratum] = {"population_share": round(p, 5),
                         "test_share": round(t, 5),
                         "abs_deviation": round(abs(t - p), 5)}
    return rows


def _summarise(g: pd.DataFrame, split_col: str) -> dict:
    """Headline difficulty markers for a split's test fold."""
    test = g[g[split_col] == "test"]
    died = test["died"].dropna()
    return {
        "n_stays": int(len(test)),
        "n_transitions": int(test["n_transitions"].sum()),
        "mean_episode_length": round(float(test["n_transitions"].mean()), 3),
        "median_episode_length": float(test["n_transitions"].median()),
        "mortality_rate": (round(float(died.mean()), 5) if len(died) else None),
        "max_abs_stratum_deviation": round(
            max(r["abs_deviation"] for r in _balance_table(g, split_col).values()), 5),
    }


def build(track: str = "a", write: bool = True) -> dict:
    """Audit the shipped split and emit a frozen stratified one.

    ``write=False`` computes and returns everything without touching disk — the
    same guard the OPE entry points carry, so a test can exercise this without
    overwriting the frozen CSV that published numbers refer to.
    """
    g = _assign_strata(_per_stay(track))
    prop = _target_proportions(g)
    g["bench_split"] = _stratified_split(g, prop)

    population_mortality = (float(g["died"].dropna().mean())
                            if g["died"].notna().any() else None)
    result = {
        "track": track,
        "n_stays": int(len(g)),
        "n_transitions": int(g["n_transitions"].sum()),
        "n_strata": int(g["stratum"].nunique()),
        "seed": SEED,
        "target_proportions": {k: round(v, 5) for k, v in prop.items()},
        "population_mortality_rate": (round(population_mortality, 5)
                                      if population_mortality is not None else None),
        "shipped": {
            "summary": _summarise(g, "shipped_split"),
            "stratum_balance": _balance_table(g, "shipped_split"),
        },
        "stratified": {
            "summary": _summarise(g, "bench_split"),
            "stratum_balance": _balance_table(g, "bench_split"),
        },
        "notes": {
            "purpose": "benchmark split only — the shipped split remains authoritative "
                       "for the deployed policy",
            "stratification": "episode-length quartile x mortality (unknown outcome "
                              "kept as its own class, not folded into survivors)",
            "allocation": "largest-remainder within each stratum, ordered by a hash "
                          "of the stay id so the split is reproducible from ids alone",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    shipped_dev = result["shipped"]["summary"]["max_abs_stratum_deviation"]
    strat_dev = result["stratified"]["summary"]["max_abs_stratum_deviation"]
    result["verdict"] = (
        "shipped split is already balanced — stratification is not needed to defend it"
        if shipped_dev <= 0.01 else
        "shipped split deviates from the population by more than 1pp in at least one "
        "stratum — report benchmark numbers on the stratified split")

    if write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        cols = ["stay_id", "n_transitions", "length_quartile", "outcome_class",
                "stratum", "shipped_split", "bench_split"]
        csv = RESULTS / f"split_compat_track_{track}.csv"
        g[cols].sort_values("stay_id").to_csv(csv, index=False)
        (RESULTS / f"split_compat_track_{track}.json").write_text(
            json.dumps(result, indent=2))
        log.info("Frozen stratified split -> %s", csv)

    log.info("Split audit track %s: %d stays, %d strata, population mortality %s",
             track, len(g), g["stratum"].nunique(),
             f"{population_mortality:.4f}" if population_mortality is not None else "n/a")
    for name in ("shipped", "stratified"):
        s = result[name]["summary"]
        log.info("  %-11s test: %5d stays  %7d transitions  mean_len %6.2f  "
                 "mortality %s  max stratum dev %.4f",
                 name, s["n_stays"], s["n_transitions"], s["mean_episode_length"],
                 f"{s['mortality_rate']:.4f}" if s["mortality_rate"] is not None else "n/a",
                 s["max_abs_stratum_deviation"])
    log.info("  verdict: %s", result["verdict"])
    return result


def load(track: str = "a") -> dict[str, set[int]]:
    """Read the frozen stratified split back as ``{split: {stay_id}}``.

    Raises if it was never written: a benchmark run must not silently fall back to
    the shipped split and then claim it was stratified.
    """
    csv = RESULTS / f"split_compat_track_{track}.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not present — run `python -m benchmark.split_compat --track {track}` "
            "once and commit the result; the split must be frozen before it is used.")
    df = pd.read_csv(csv)
    return {s: set(df.loc[df["bench_split"] == s, "stay_id"].astype(int))
            for s in SPLITS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="a", choices=["a", "b"])
    ap.add_argument("--no-write", action="store_true",
                    help="compute and print without touching the frozen CSV")
    args = ap.parse_args()
    build(args.track, write=not args.no_write)


if __name__ == "__main__":
    main()
