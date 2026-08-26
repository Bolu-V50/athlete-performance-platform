"""Tests for the multi-quality analytics layer.

The thing most likely to be silently wrong here is direction. Seven of the
twenty-two catalogued metrics improve by getting smaller, and every trend,
z-score and "most improved" statement inverts for those. These tests exist to
make that impossible to break unnoticed.
"""

from __future__ import annotations

import os

import pytest

needs_db = pytest.mark.skipif(not os.getenv("SUPABASE_DB_URL"), reason="no database configured")
pytestmark = needs_db


@pytest.fixture(scope="module")
def catalog():
    from src.analytics.queries import _df

    return _df("select * from metric_catalog")


def test_every_lower_is_better_metric_is_declared(catalog):
    """If one of these ever flips to higher_is_better the dashboard will call a
    slower sprint an improvement, and nothing else in the system would notice."""
    lower = set(catalog[~catalog["higher_is_better"]]["metric_name"])
    assert {
        "sprint_10m_s", "sprint_30m_s", "agility_505_s", "cod_deficit_s",
        "wingate_fatigue_index_pct", "sum7_skinfolds_mm", "contraction_time_s",
    } <= lower


def test_every_quality_has_at_least_one_headline_metric(catalog):
    """A quality with no headline metric is silently absent from every capability
    profile -- which is exactly how body composition went missing.

    More than one headline per quality is deliberate: power carries both jump
    height and RSI-mod, and speed carries a 10 m sprint for land athletes and a
    100 m freestyle for swimmers. An athlete only ever has data for the tests
    their sport runs, so the profile still shows one row per measured quality.
    """
    headline = catalog[catalog["is_headline"]]
    assert set(catalog["quality"]) == set(headline["quality"]), (
        f"no headline metric for: {set(catalog['quality']) - set(headline['quality'])}"
    )


def test_every_catalogued_metric_has_an_acceptance_range(catalog):
    """The pipeline validates against these; a NULL range means that metric is
    ingested unchecked."""
    missing = catalog[catalog["typical_min"].isna() | catalog["typical_max"].isna()]
    assert missing.empty, f"no acceptance range for: {list(missing['metric_name'])}"


def test_improvement_sign_is_flipped_for_lower_is_better_metrics():
    """A sprint time that falls is an improvement. pct_change is the raw
    arithmetic; pct_improvement must invert it for these metrics."""
    from src.analytics.queries import _df

    df = _df(
        "select metric_name, higher_is_better, pct_change, pct_improvement "
        "from v_metric_trend where pct_change is not null and pct_change <> 0"
    )
    assert not df.empty
    lower = df[~df["higher_is_better"]]
    assert not lower.empty, "no lower-is-better metrics in the data to check"
    for r in lower.itertuples(index=False):
        assert float(r.pct_change) == pytest.approx(-float(r.pct_improvement), abs=0.11), (
            f"{r.metric_name}: raw {r.pct_change} vs improvement {r.pct_improvement}"
        )
    higher = df[df["higher_is_better"]]
    for r in higher.itertuples(index=False):
        assert float(r.pct_change) == pytest.approx(float(r.pct_improvement), abs=0.11)


def test_direction_uses_the_fitted_slope_not_the_endpoints():
    """Endpoint comparison inherits the test-retest error of two single days and
    can invert a real trend. Direction must follow the fitted change."""
    from src.analytics.queries import _df

    df = _df(
        "select athlete_code, metric_name, direction, fitted_change_in_sd, change_in_sd "
        "from v_quality_profile where direction <> 'insufficient_data' "
        "and fitted_change_in_sd is not null"
    )
    assert not df.empty
    for r in df.itertuples(index=False):
        sd = float(r.fitted_change_in_sd)
        expected = "improving" if sd >= 1.0 else "declining" if sd <= -1.0 else "stable"
        assert r.direction == expected, f"{r.athlete_code}/{r.metric_name}"

    # and there should genuinely be cases where the two estimators disagree,
    # otherwise this distinction is untested by the data
    disagree = df[
        ((df["fitted_change_in_sd"].astype(float) >= 1.0) & (df["change_in_sd"].astype(float) < 1.0))
        | ((df["fitted_change_in_sd"].astype(float) <= -1.0) & (df["change_in_sd"].astype(float) > -1.0))
    ]
    assert not disagree.empty, "no case where fitted and endpoint estimates differ"


def test_test_day_z_scores_are_polarity_corrected():
    """A faster-than-usual sprint must read as a positive z, not a negative one."""
    from src.analytics.queries import _df

    df = _df(
        "select value, athlete_mean, z_vs_own_mean, higher_is_better "
        "from v_test_day where z_vs_own_mean is not null and metric_name = 'sprint_10m_s' "
        "and value < athlete_mean limit 20"
    )
    assert not df.empty, "no faster-than-average sprints in the data"
    assert (df["z_vs_own_mean"].astype(float) > 0).all(), (
        "a sprint faster than the athlete's own mean is scored as below average"
    )


# Qualities every sport in the programme measures, whatever else it does.
UNIVERSAL_QUALITIES = {"power", "max_strength", "anaerobic", "body_comp"}


def test_every_athlete_has_the_universally_measured_qualities():
    from src.analytics.queries import _df

    df = _df(
        "select athlete_code, quality from v_quality_profile"
    )
    assert not df.empty
    for code, grp in df.groupby("athlete_code"):
        missing = UNIVERSAL_QUALITIES - set(grp["quality"])
        assert not missing, f"{code} is missing {missing}"


def test_a_sport_only_carries_qualities_its_battery_measures():
    """Swimmers do not run a 505, so they must have no change-of-direction row.
    A profile that invented one would mean the pipeline is attributing another
    sport's test to them."""
    from src.analytics.queries import _df

    df = _df(
        "select a.sport, p.quality, count(*) n from v_quality_profile p "
        "join athletes a using (athlete_id) group by a.sport, p.quality"
    )
    swim = set(df[df["sport"] == "Swimming"]["quality"])
    assert "cod" not in swim, "swimmers have change-of-direction data they never tested for"
    assert "speed" in swim, "swimmers should have speed from the pool time-trial"
    assert "aerobic" in swim, "swimmers should have aerobic from critical swim speed"

    football = set(df[df["sport"] == "Football"]["quality"])
    assert {"cod", "aerobic", "speed"} <= football
