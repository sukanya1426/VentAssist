"""Tests for the Track B cohort — the waveform ∩ Track-A-episode intersection.

These pin the 2026-10-02 fix. Track B used to re-derive its own ventilation
windows by merging sparse ``chartevents`` PEEP/TV events with a 2 h gap; because
settings are charted roughly every 4 h, that split a continuous ventilation
course into ~4 h fragments and the >= 4 h filter then discarded nearly all of
them (15 surviving episodes out of 111 records with overlap). The stage now
intersects the recording windows with the Track A episodes, which come from
``procedureevents`` itemid 225792 and carry explicit start/end times.

The regression these guard against is silent: a fragmenting cohort builder still
produces a *valid-looking* cohort, just a tiny one. So the assertions are about
the SHAPE of the result — contiguity, uniqueness, and containment in a Track A
episode — plus a floor on the data volume.

Run:  pytest backend/tests/test_cohort_track_b.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.pipeline import config
from backend.pipeline import cohort_track_b as CB


class Skip(Exception):
    """Raised when an artifact this test needs has not been built."""


def _cohort() -> pd.DataFrame:
    path = config.PROCESSED_PATH / "cohort_track_b.csv"
    if not path.exists():
        raise Skip(f"{path.name} not built — run python -m backend.pipeline.cohort_track_b")
    return pd.read_csv(path, parse_dates=["vent_start", "vent_end",
                                          "rec_start", "rec_end"])


def _track_a() -> pd.DataFrame:
    path = config.PROCESSED_PATH / "cohort.csv"
    if not path.exists():
        raise Skip("cohort.csv not built — run python -m backend.pipeline.cohort")
    return pd.read_csv(path, parse_dates=["vent_start", "vent_end"])


def _load_or_skip(fn):
    try:
        return fn()
    except Skip as e:
        pytest.skip(str(e))


# --------------------------------------------------------------------------- #
# Shape of the intersection
# --------------------------------------------------------------------------- #
def test_one_episode_per_stay():
    """(stay_id, hour) must be unique downstream, so a stay may appear once.

    A subject can own several recordings and several ventilation episodes, so the
    raw pair join is many-to-many; build() keeps the longest overlap per stay. If
    that dedupe is lost, the aggregator emits duplicate (stay_id, hour) rows and
    the MDP silently double-counts those transitions.
    """
    c = _load_or_skip(_cohort)
    assert c["stay_id"].is_unique, "a stay appears more than once in the Track B cohort"


def test_every_episode_is_contiguous_and_at_least_the_minimum():
    """No fragments: each episode is one unbroken window of >= the 4 h floor.

    This is the direct regression for the old chartevents re-derivation, which
    produced 4.0 h stubs sitting exactly on the threshold.
    """
    c = _load_or_skip(_cohort)
    span = (c["vent_end"] - c["vent_start"]) / pd.Timedelta(hours=1)
    assert (span > 0).all(), "an episode has a non-positive duration"
    assert (c["overlap_hours"] >= CB._MIN_OVERLAP_H - 1e-6).all(), \
        f"an episode is shorter than the {CB._MIN_OVERLAP_H} h floor"
    # The window must lie inside the recording it was cut from.
    assert (c["vent_start"] >= c["rec_start"] - pd.Timedelta(seconds=1)).all()
    assert (c["vent_end"] <= c["rec_end"] + pd.Timedelta(seconds=1)).all()


def test_episode_lies_inside_a_track_a_ventilation_episode():
    """Each Track B window must be contained in a real procedureevents episode.

    This is what makes the waveform hours genuinely *ventilated* hours. If the
    stage ever reverts to inventing its own windows, containment breaks.
    """
    c = _load_or_skip(_cohort)
    a = _load_or_skip(_track_a)
    by_stay = {int(r.stay_id): r for r in a.itertuples()}
    for row in c.itertuples():
        ep = by_stay.get(int(row.stay_id))
        assert ep is not None, f"stay {row.stay_id} is not a Track A episode"
        tol = pd.Timedelta(seconds=1)
        assert row.vent_start >= ep.vent_start - tol, \
            f"stay {row.stay_id} starts before its Track A episode"
        assert row.vent_end <= ep.vent_end + tol, \
            f"stay {row.stay_id} ends after its Track A episode"


def test_episode_respects_the_length_cap():
    c = _load_or_skip(_cohort)
    span = (c["vent_end"] - c["vent_start"]) / pd.Timedelta(hours=1)
    assert (span <= config.MAX_EPISODE_HOURS + 1e-6).all(), \
        f"an episode exceeds MAX_EPISODE_HOURS ({config.MAX_EPISODE_HOURS})"


def test_resp_channel_is_required():
    """The Resp belt drives 3 of the 6 waveform features, so it is mandatory."""
    c = _load_or_skip(_cohort)
    assert c["has_resp"].all(), "an episode without a Resp channel was admitted"


# --------------------------------------------------------------------------- #
# Volume floor — the point of the fix
# --------------------------------------------------------------------------- #
def test_cohort_is_not_starved_by_fragmentation():
    """Guard the ~12x gain: the fragmenting builder yielded 15 episodes / ~83 h.

    The floors are set well below the measured 37 episodes / 1223 overlap-hours
    so that ordinary data-scope changes do not trip them, while a regression to
    per-charting-interval fragments (which cannot clear 20 episodes) does.
    """
    c = _load_or_skip(_cohort)
    assert len(c) >= 20, (
        f"only {len(c)} Track B episodes — the fragmenting chartevents "
        "re-derivation yielded 15; expected ~37 from the Track A intersection")
    assert c["overlap_hours"].sum() >= 600, (
        f"only {c['overlap_hours'].sum():.0f} overlap-hours — expected ~1223; "
        "episodes are probably being fragmented again")
    assert c["overlap_hours"].median() > CB._MIN_OVERLAP_H + 1.0, (
        "the median episode sits on the minimum-overlap floor, which is the "
        "signature of fragmentation rather than real short recordings")


def test_patient_attributes_come_from_the_track_a_cohort():
    """Age/weight are inherited, not stamped with population defaults.

    The old builder skipped the weight scan and wrote ``weight_kg = 80.0`` for
    every row, and imputed the median age whenever ``patients`` lookup failed —
    both of which fed the 12-dim state and the volutrauma safety term.
    """
    c = _load_or_skip(_cohort)
    assert c["weight_kg"].nunique() > 1, \
        "every episode carries an identical weight — a population default is back"
    assert c["age"].nunique() > 1, "every episode carries an identical age"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except (Skip, pytest.skip.Exception) as e:
                print(f"SKIP {name}: {e}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print("ALL TESTS PASSED" if failures == 0 else f"{failures} TEST(S) FAILED")
    raise SystemExit(1 if failures else 0)
