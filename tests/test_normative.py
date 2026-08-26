"""Tests for the published normative reference layer.

Two failure modes matter here and neither announces itself.

**A fabricated citation.** Every value must trace to a study carrying a DOI or
PMID and the URL it was verified from. A comparison a coach cannot follow back
to a paper is not evidence, and an invented reference in a document a coach acts
on is worse than no reference at all.

**A spread type used as if it were something else.** Papers report standard
deviations, 95%% confidence intervals and plain ranges. A CI is uncertainty about
the mean; an SD is the spread of athletes. Dividing by a CI makes an athlete look
several standard deviations from normal when they are a fraction of one.
"""

from __future__ import annotations

import os

import pytest

needs_db = pytest.mark.skipif(not os.getenv("SUPABASE_DB_URL"), reason="no database configured")
pytestmark = needs_db


@pytest.fixture(scope="module")
def studies():
    from src.analytics.queries import reference_studies

    return reference_studies()


@pytest.fixture(scope="module")
def values():
    from src.analytics.queries import _df

    return _df("select * from normative_values")


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def test_every_study_is_identifiable_and_traceable(studies):
    assert not studies.empty
    for r in studies.itertuples(index=False):
        assert r.doi or r.pmid, f"{r.study_key} has neither a DOI nor a PMID"
        assert r.source_url, f"{r.study_key} records no URL it was verified from"
        assert r.verified_on is not None
        assert len(r.citation) > 60, f"{r.study_key} citation looks truncated"


def test_every_normative_value_belongs_to_a_recorded_study(values, studies):
    assert not values.empty
    known = set(studies["study_key"])
    assert set(values["study_key"]) <= known


def test_no_study_is_loaded_without_being_used(studies):
    """A reference in the library that nothing cites is either a leftover or a
    value that failed verification and was half-removed."""
    unused = studies[studies["n_values"] == 0]
    assert unused.empty, f"unused reference studies: {list(unused['study_key'])}"


# ---------------------------------------------------------------------------
# spread semantics
# ---------------------------------------------------------------------------
def test_spread_fields_match_the_declared_spread_type(values):
    for r in values.itertuples(index=False):
        if r.spread_type == "sd":
            assert r.sd_value is not None, f"{r.study_key}/{r.metric_name}: sd declared, none given"
        elif r.spread_type in ("ci95", "range"):
            assert r.low_value is not None and r.high_value is not None
            assert float(r.low_value) < float(r.high_value)
            assert r.sd_value is None, (
                f"{r.study_key}/{r.metric_name}: a {r.spread_type} must not carry an SD"
            )


def test_a_z_score_is_only_computed_from_a_standard_deviation():
    """The core statistical guard of this layer."""
    from src.analytics.queries import _df

    df = _df(
        "select metric_name, population, spread_type, z_vs_reference "
        "from v_normative_comparison"
    )
    assert not df.empty
    non_sd = df[df["spread_type"] != "sd"]
    assert not non_sd.empty, "no non-SD reference present to test the guard"
    assert non_sd["z_vs_reference"].isna().all(), (
        "a z-score was computed against a confidence interval or a range"
    )
    sd_rows = df[df["spread_type"] == "sd"]
    assert sd_rows["z_vs_reference"].notna().all()


# ---------------------------------------------------------------------------
# units and matching
# ---------------------------------------------------------------------------
def test_reference_units_match_the_metric_catalogue(values):
    """A reference in centimetres against a metric stored in metres is wrong by a
    factor of a hundred and nothing on screen would look odd."""
    from src.analytics.queries import _df

    catalog = _df("select metric_name, unit from metric_catalog").set_index("metric_name")["unit"]
    for r in values.itertuples(index=False):
        assert r.unit == catalog[r.metric_name], (
            f"{r.study_key}/{r.metric_name}: reference in {r.unit}, catalogue says "
            f"{catalog[r.metric_name]}"
        )


def test_comparisons_never_cross_sex():
    """Comparing a woman against a male reference band is not a subtle error and
    it is easy to introduce with a careless join."""
    from src.analytics.queries import _df

    df = _df(
        "select v.athlete_code, a.sex as athlete_sex, n.sex as reference_sex "
        "from v_normative_comparison v "
        "join athletes a on a.athlete_code = v.athlete_code "
        "join normative_values n on n.study_key = v.study_key "
        "  and n.metric_name = v.metric_name and n.population = v.population"
    )
    assert not df.empty
    mismatched = df[df["reference_sex"].notna() & (df["reference_sex"] != df["athlete_sex"])]
    assert mismatched.empty, f"cross-sex comparisons: {mismatched.to_dict('records')[:3]}"


def test_comparisons_never_cross_sport():
    from src.analytics.queries import _df

    df = _df(
        "select distinct v.sport as athlete_sport, n.sport as reference_sport "
        "from v_normative_comparison v "
        "join normative_values n on n.study_key = v.study_key "
        "  and n.metric_name = v.metric_name and n.population = v.population"
    )
    assert (df["athlete_sport"] == df["reference_sport"]).all()


def test_the_direction_of_the_comparison_is_polarity_corrected():
    """A 10 m sprint faster than the reference must read as positive."""
    from src.analytics.queries import _df

    df = _df(
        "select athlete_value, reference_mean, pct_vs_reference, higher_is_better "
        "from v_normative_comparison where metric_name = 'sprint_10m_s' "
        "and athlete_value < reference_mean"
    )
    assert not df.empty, "no athlete faster than the sprint reference to check"
    assert (df["pct_vs_reference"].astype(float) > 0).all(), (
        "an athlete faster than the published mean is scored as worse than it"
    )


# ---------------------------------------------------------------------------
# absence must look like absence
# ---------------------------------------------------------------------------
def test_a_sport_with_no_reference_gets_no_comparison():
    """Swimming and sprint athletics are not represented in the library. They
    must return nothing rather than borrow another sport's band."""
    from src.analytics.queries import _df

    df = _df(
        "select distinct sport from v_normative_comparison"
    )
    covered = set(df["sport"])
    assert "Swimming" not in covered
    assert covered <= {"Football", "Basketball"}


def test_metrics_without_a_reference_are_reported_explicitly():
    from src.analytics.queries import metrics_without_reference

    missing = metrics_without_reference("ATH-009")
    assert not missing.empty, "this athlete should have some unreferenced metrics"
    assert "display_name" in missing.columns
