from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from smart_data_studio import analysis
from smart_data_studio.config import (
    MAX_COMPARISON_GROUPS,
    MAX_LLM_PAYLOAD_CHARS,
)
from smart_data_studio.dataset import CsvSource, Dataset
from smart_data_studio.tools import AnalysisTools


def two_groups(shift: float, size: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "segment": ["A"] * size + ["B"] * size,
            "value": np.concatenate([rng.normal(100, 10, size), rng.normal(100 + shift, 10, size)]),
        }
    )


def test_a_real_difference_and_a_trivial_one_are_told_apart() -> None:
    """Both are significant at this size; only the effect size separates them."""
    real = analysis.compare_groups(two_groups(shift=12.0), "segment", "value")["test"]
    trivial = analysis.compare_groups(two_groups(shift=0.15), "segment", "value")["test"]

    assert real["mann_whitney_p_value"] < 0.01
    assert real["effect"] in {"large", "medium"}

    # The trivial difference may well be "significant" — the effect size is what saves us.
    assert trivial["effect"] == "negligible"
    assert abs(trivial["cliffs_delta"]) < abs(real["cliffs_delta"])


def test_group_summaries_cover_every_row_even_when_the_test_samples() -> None:
    frame = two_groups(shift=5.0, size=60_000)
    result = analysis.compare_groups(frame, "segment", "value")
    assert "sampling" in result
    assert [group["rows"] for group in result["groups"]] == [60_000, 60_000]


def test_comparison_rejects_unusable_columns() -> None:
    with pytest.raises(analysis.NotAnalysable, match="not found"):
        analysis.compare_groups(two_groups(1.0), "nope", "value")
    single = pd.DataFrame({"segment": ["A"] * 5, "value": [1.0] * 5})
    with pytest.raises(analysis.NotAnalysable, match="fewer than two groups"):
        analysis.compare_groups(single, "segment", "value")


def test_driver_sweep_finds_the_dimension_that_moved_not_the_one_asked_about() -> None:
    """The point of sweeping: the mover is geo, while tier is flat and would mislead."""
    frame = pd.DataFrame(
        {
            "period": ["before"] * 4 + ["after"] * 4,
            "tier": ["gold", "gold", "silver", "silver"] * 2,
            "geo": ["local", "national"] * 4,
            "revenue": [100.0, 100.0, 100.0, 100.0, 40.0, 160.0, 40.0, 160.0],
        }
    )
    result = analysis.rank_drivers(frame, "revenue", "period")

    assert result["total_change"] == 0.0  # the total hides the movement underneath
    assert result["drivers"][0]["dimension"] == "geo"
    moves = {item["level"]: item["change"] for item in result["drivers"][0]["movers"]}
    assert moves["local"] == -120.0 and moves["national"] == 120.0


def test_driver_sweep_needs_exactly_two_sides() -> None:
    frame = pd.DataFrame({"period": ["a", "b", "c"], "revenue": [1.0, 2.0, 3.0]})
    with pytest.raises(analysis.NotAnalysable, match="exactly two values"):
        analysis.rank_drivers(frame, "revenue", "period")


def test_numeric_and_categorical_associations_are_ranked_apart() -> None:
    """One ranked list said the two scores share a scale. They do not.

    rare_extreme is five rows in three thousand, shifted far enough to stand out
    by eye and explaining nothing anyone could act on — and its eta squared is
    0.81. strong_numeric, which really does drive the target, scores 0.99. The
    single list therefore happened to order these two correctly, and would have
    inverted for any weaker driver, because eta squared and a Spearman rho are
    different quantities that both run 0 to 1.
    """
    rng = np.random.default_rng(1)
    size = 3000
    driver = rng.normal(0, 1, size)
    frame = pd.DataFrame(
        {
            "strong_numeric": driver,
            "noise": rng.normal(0, 1, size),
            "rare_extreme": ["normal"] * (size - 5) + ["rare"] * 5,
            "target": driver * 10 + rng.normal(0, 1, size),
        }
    )
    frame.loc[frame["rare_extreme"] == "rare", "target"] += 500

    found = analysis.relate(frame, "target")
    assert "associations" not in found, "the merged ranking is what claimed one scale"

    numeric = [item["column"] for item in found["numeric_associations"]]
    assert numeric == ["strong_numeric", "noise"], "within one measure the order is meaningful"

    categorical = {item["column"]: item["strength"] for item in found["categorical_associations"]}
    assert categorical["rare_extreme"] > 0.5, (
        "the hazard this test exists for: a column of five interesting rows scores high, "
        "and on a shared list would outrank a numeric column that explains far more"
    )


def test_tools_return_readable_json_and_record_evidence() -> None:
    rows = ["segment,geo,value"]
    rng = np.random.default_rng(2)
    for index in range(600):
        segment = "A" if index % 2 else "B"
        rows.append(f"{segment},{'local' if index % 3 else 'national'},{rng.normal(100, 10):.3f}")
    dataset = Dataset.load([CsvSource.from_upload("g.csv", ("\n".join(rows) + "\n").encode())])
    try:
        tools = AnalysisTools(dataset)
        assert json.loads(tools.compare_groups("segment", "value"))["error"]  # no query yet

        tools.run_sql("SELECT segment, geo, value FROM g")
        compared = json.loads(tools.compare_groups("segment", "value"))
        assert compared["compared"] == ["A", "B"]
        assert "cliffs_delta" in compared["test"]

        related = json.loads(tools.relate("value"))
        assert related["target"] == "value"

        assert "not found" in json.loads(tools.compare_groups("missing", "value"))["error"]
        # Every analysis is captured so the UI can show it as evidence.
        assert [item.kind for item in tools.analyses] == ["comparison", "associations"]
    finally:
        dataset.close()


def test_driver_analysis_refuses_to_sample_because_it_sums() -> None:
    """Sampled sums gave a change of $1,091,057 where the truth was $526,870.

    Means and correlations survive a sample; the difference between two near-equal
    totals does not, so this tool must ask for aggregated input instead.
    """
    from smart_data_studio.config import MAX_ANALYSIS_CELLS

    rows = ["period,geo,revenue"]
    wide = MAX_ANALYSIS_CELLS // 3 + 10  # one row past what three columns can hold
    for index in range(min(wide, 60_000)):
        rows.append(f"{'before' if index % 2 else 'after'},{'x' if index % 3 else 'y'},{index}")
    dataset = Dataset.load([CsvSource.from_upload("d.csv", ("\n".join(rows) + "\n").encode())])
    try:
        tools = AnalysisTools(dataset)
        tools.run_sql("SELECT period, geo, revenue FROM d")
        # Well inside the budget here, so it runs on every row rather than refusing.
        assert "drivers" in json.loads(tools.rank_drivers("revenue", "period"))

        # Aggregated input is what the tool asks for, and it stays exact.
        tools.run_sql("SELECT period, geo, sum(revenue) AS revenue FROM d GROUP BY 1, 2")
        result = json.loads(tools.rank_drivers("revenue", "period"))
        assert "sampled_rows" not in result
        assert result["drivers"][0]["dimension"] == "geo"
    finally:
        dataset.close()


def test_a_group_too_small_to_test_is_refused_not_scored() -> None:
    """A one-row group returned nan p-values and called Cliff's delta of -1.0 'large'."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"g": ["A"] * 40 + ["B"], "v": list(rng.normal(10, 2, 40)) + [99.0]})
    with pytest.raises(analysis.NotAnalysable, match="at least"):
        analysis.compare_groups(frame, "g", "v")


def test_constant_groups_are_refused_rather_than_tested() -> None:
    frame = pd.DataFrame({"g": ["A"] * 30 + ["B"] * 30, "v": [5.0] * 30 + [9.0] * 30})
    with pytest.raises(analysis.NotAnalysable, match="constant within both groups"):
        analysis.compare_groups(frame, "g", "v")


def test_no_analysis_payload_can_carry_nan_or_infinity() -> None:
    """Both are invalid JSON, so the guard lives at the serialization boundary."""
    from smart_data_studio.tools import _finite

    cleaned = _finite(
        {"a": float("nan"), "b": float("inf"), "c": [1.0, float("-inf")], "d": {"e": float("nan")}}
    )
    assert cleaned == {"a": None, "b": None, "c": [1.0, None], "d": {"e": None}}
    assert "NaN" not in json.dumps(cleaned) and "Infinity" not in json.dumps(cleaned)


def test_every_analysis_record_carries_a_readable_subject() -> None:
    """The panel rendered 'Comparison ·  by  · ? periods' when these reused the series record."""
    rows = ["seg,val"] + [f"{'A' if i % 2 else 'B'},{i % 50}" for i in range(400)]
    dataset = Dataset.load([CsvSource.from_upload("s.csv", ("\n".join(rows) + "\n").encode())])
    try:
        tools = AnalysisTools(dataset)
        tools.run_sql("SELECT seg, val FROM s")
        tools.compare_groups("seg", "val")
        tools.relate("val")
        subjects = [record.subject for record in tools.analyses]
        assert subjects == ["val across seg", "what relates to val"]
        assert all(subject.strip() for subject in subjects)
    finally:
        dataset.close()


def test_a_comparison_over_many_groups_stays_inside_the_prompt_budget() -> None:
    """Every other tool is budgeted — run_sql digests, relate keeps 15, drivers 6.
    This one listed every group, which on 2,000 levels meant 195,000 characters of
    summary for the two it went on to test."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "segment": [f"segment_{index % 2000}" for index in range(60_000)],
            "value": rng.normal(100, 10, 60_000),
        }
    )
    result = analysis.compare_groups(frame, "segment", "value")

    assert len(result["groups"]) == MAX_COMPARISON_GROUPS
    assert "2,000 groups" in result["note"]
    assert f"the {MAX_COMPARISON_GROUPS} largest are listed" in result["note"]
    # Listed largest first, so the pair actually tested is always among them.
    assert {name for name in result["compared"]} <= {group["group"] for group in result["groups"]}
    assert len(json.dumps(result)) < MAX_LLM_PAYLOAD_CHARS


def test_a_comparison_over_a_few_groups_still_lists_them_all() -> None:
    frame = pd.DataFrame(
        {"segment": [f"s{index % 3}" for index in range(300)], "value": range(300)}
    )
    result = analysis.compare_groups(frame, "segment", "value")

    assert len(result["groups"]) == 3
    assert "all of them are listed" in result["note"]


def outlier_population(seed: int = 1) -> pd.DataFrame:
    """A population whose totals span three orders of magnitude, two genuinely
    large members, and one member that is broken rather than big."""
    rng = np.random.default_rng(seed)
    size = 200
    return pd.DataFrame(
        {
            "machine": [f"m{index}" for index in range(size)] + ["huge1", "huge2", "broken"],
            "total": np.concatenate([rng.lognormal(7, 1.2, size), [400_000, 380_000], [900]]),
            "rate": np.concatenate([rng.normal(8, 0.4, size), [8.1, 7.9], [22.0]]),
        }
    )


def test_standing_apart_is_not_the_same_as_being_large() -> None:
    """The failure this exists for: asked which machines were unusual, the model
    ranked by the largest raw gap, so the busiest machine won whatever it was
    doing. On the rate, the two largest are ordinary and the broken one is not."""
    found = analysis.find_outliers(outlier_population(), "machine", "rate")
    flagged = {item["entity"] for item in found["outliers"]}

    assert "broken" in flagged, found["outliers"]
    assert not flagged & {"huge1", "huge2"}, "the biggest entities were flagged for being big"
    assert found["outliers"][0]["entity"] == "broken"
    assert found["outliers"][0]["direction"] == "high"


def test_a_skewed_total_says_so_instead_of_pretending() -> None:
    """Statistics cannot rescue the wrong measure: on a long tail, far from the
    median and large are the same thing. So it says so and names the fix, rather
    than returning the head of the tail as though it were a finding."""
    skewed = analysis.find_outliers(outlier_population(), "machine", "total")
    assert "skew_warning" in skewed
    assert "rate or a ratio" in skewed["skew_warning"]

    # ...and stays quiet on a measure where the flagged list is not the tail.
    clean = analysis.find_outliers(outlier_population(), "machine", "rate")
    assert "skew_warning" not in clean, clean.get("skew_warning")


def test_an_outlier_cannot_hide_behind_the_spread_it_creates() -> None:
    """Mean and standard deviation are the obvious choice and the wrong one.

    An outlier inflates the deviation it is then measured against, and the effect
    has a hard ceiling: a z-score cannot exceed (n-1)/sqrt(n) however extreme the
    value is, so at sixty entities nothing can score past about 7.8 — a million
    against a population of tens reads the same as a mild deviation. The median
    and the MAD are not moved by the point being tested.
    """
    rng = np.random.default_rng(3)
    ordinary = rng.normal(10, 1, 60)
    frame = pd.DataFrame(
        {
            "who": [f"e{index}" for index in range(61)],
            "value": np.append(ordinary, 1_000_000.0),
        }
    )

    found = analysis.find_outliers(frame, "who", "value")
    # Not the only one flagged — at this threshold an ordinary draw lands past it
    # now and then — but first, and by a distance nothing else comes near.
    assert found["outliers"][0]["entity"] == "e60"

    values = frame["value"].to_numpy()
    plain_z = (1_000_000 - values.mean()) / values.std(ddof=1)
    assert plain_z < np.sqrt(len(values)), "a z-score is capped by the sample size"
    assert found["outliers"][0]["score"] > 100 * plain_z


@pytest.mark.parametrize(
    ("label", "frame", "because"),
    [
        (
            "too few to have a population",
            pd.DataFrame({"who": list("abcde"), "value": [1.0, 2, 3, 4, 99]}),
            "at least",
        ),
        (
            "no spread to stand apart from",
            pd.DataFrame({"who": [f"e{i}" for i in range(40)], "value": [5.0] * 40}),
            "no spread",
        ),
    ],
)
def test_find_outliers_refuses_what_it_cannot_answer(label, frame, because) -> None:
    with pytest.raises(analysis.NotAnalysable, match=because):
        analysis.find_outliers(frame, "who", "value")


def test_an_entity_seen_once_is_reported_with_its_own_thinness() -> None:
    """Two machines were flagged at 35 and -29 deviations on a real file, both
    genuinely far from the estate. Both had a single day of data, where one
    jackpot moves the rate entirely — far from the median because it has not
    averaged out, which reads identically to broken hardware unless it is said."""
    rng = np.random.default_rng(5)
    rows = [
        {"machine": f"m{index // 30}", "rate": value}
        for index, value in enumerate(rng.normal(8, 0.4, 30 * 40))
    ]
    rows.append({"machine": "seen_once", "rate": 22.0})
    found = analysis.find_outliers(pd.DataFrame(rows), "machine", "rate")

    flagged = {item["entity"]: item for item in found["outliers"]}
    assert "seen_once" in flagged
    assert flagged["seen_once"]["observations"] == 1
    assert all(item["observations"] >= 1 for item in found["outliers"])
    assert "observations" in found["reading"]


def test_an_already_aggregated_result_says_so_rather_than_blaming_group_size() -> None:
    """The wrong diagnosis costs the whole turn.

    Asked whether LOCAL and NATIONAL players really differ, the model queried the
    average per geoType and handed the two rows to the test. It was told the second
    largest group had one row and to find groups with more data — advice that leads
    nowhere from an aggregate — so it called the tool again, and again, nine times,
    until the round limit ended the turn with no answer at all.
    """
    frame = pd.DataFrame({"geoType": ["LOCAL", "NATIONAL"], "avg_theo_win": [55.04, 134.62]})
    with pytest.raises(analysis.NotAnalysable) as raised:
        analysis.compare_groups(frame, "geoType", "avg_theo_win")

    message = str(raised.value)
    assert "already been aggregated" in message
    assert "without GROUP BY" in message, "the message has to say what to do instead"
    assert "coarser dimension" not in message, "that is the advice that sent it in circles"


def test_a_genuinely_small_group_still_says_so() -> None:
    """The other side of it: a real result with a thin group keeps its own reason."""
    frame = pd.DataFrame(
        {"tier": ["top"] * 40 + ["thin"] * 3, "spend": list(range(40)) + [1.0, 2.0, 3.0]}
    )
    with pytest.raises(analysis.NotAnalysable, match="at least 10 are needed"):
        analysis.compare_groups(frame, "tier", "spend")


def test_every_dimension_accounts_for_the_whole_change_not_a_part_of_it() -> None:
    """Six dimensions come back side by side under one key called "drivers".

    Each is the same total sliced a different way, so their moves are the same
    money seen from two angles — added together they double count, and the
    largest of them locates the change rather than explaining it. Nothing in the
    numbers says so, so the reading has to.
    """
    frame = pd.DataFrame(
        {
            "period": ["before"] * 4 + ["after"] * 4,
            "region": ["north", "north", "south", "south"] * 2,
            "tier": ["gold", "silver"] * 4,
            "revenue": [100.0, 100.0, 100.0, 100.0, 40.0, 100.0, 100.0, 100.0],
        }
    )
    found = analysis.rank_drivers(frame, "revenue", "period")

    assert found["total_change"] == -60.0
    # Both dimensions carry the whole change, which is the point: they are two
    # views of one movement, not two contributions to it.
    for dimension in found["drivers"]:
        moved = sum(item["change"] for item in dimension["movers"])
        assert moved == found["total_change"], (
            f"{dimension['dimension']} sums to {moved}, not the total change"
        )

    reading = found["reading"]
    assert "never add across dimensions" in reading
    assert "locates the change rather than explaining it" in reading


def test_a_rate_is_compared_as_a_rate_not_as_an_amount() -> None:
    """The failure this tool exists for, in the numbers that showed it.

    Readmission of 30.19% against 6.38% is a 4.7-fold risk and 23.8 points.
    compare_groups reports Cliff's delta for it — 0.238, which its own bands call
    "small", because a rank-based measure on a 0/1 column is the difference in
    proportions read against thresholds built for continuous data. Right about
    significance, wrong about importance.
    """
    frame = pd.DataFrame(
        {
            "band": ["80+"] * 106 + ["40-64"] * 47,
            "readmitted": [1] * 32 + [0] * 74 + [1] * 3 + [0] * 44,
        }
    )

    old = analysis.compare_groups(frame, "band", "readmitted")
    assert old["test"]["effect"] == "small", "the fixture no longer reproduces the misreading"

    found = analysis.compare_rates(frame, "band", "readmitted")
    assert [group["group"] for group in found["groups"]] == ["80+", "40-64"]
    assert found["groups"][0]["events"] == 32 and found["groups"][0]["observed"] == 106
    comparison = found["comparison"]
    assert round(comparison["relative_risk"], 1) == 4.7
    assert round(comparison["risk_difference_pct_points"], 1) == 23.8
    assert comparison["p_value"] < 0.01
    # And no band, because banding is what went wrong.
    assert "effect" not in comparison


def test_an_interval_is_reported_and_widens_when_the_count_is_thin() -> None:
    """43.75% from 14 of 32 reads precise and is not: Wilson puts it near 28-61%.

    The model already says "small sample size" in prose. What it cannot do without
    this is say how small, and the answer changes with the width.
    """
    frame = pd.DataFrame(
        {
            "segment": ["subprime"] * 32 + ["prime"] * 68,
            "defaulted": [1] * 14 + [0] * 18 + [1] * 5 + [0] * 63,
        }
    )
    found = analysis.compare_rates(frame, "segment", "defaulted")
    thin = next(group for group in found["groups"] if group["group"] == "subprime")
    low, high = thin["interval_95_pct"]
    assert thin["rate_pct"] == 43.75
    assert low < 30 and high > 58, f"the interval {low}-{high} does not carry the uncertainty"
    # The wide arm is wider than the well-counted one.
    fat = next(group for group in found["groups"] if group["group"] == "prime")
    assert (high - low) > (fat["interval_95_pct"][1] - fat["interval_95_pct"][0])


def test_the_unit_of_analysis_is_named_and_repeated_rows_are_collapsed() -> None:
    """Six encounters from one patient are one patient's outcome. Counted as six,
    both the rate and its precision are overstated, and nothing in the old output
    said which had been counted."""
    frame = pd.DataFrame(
        {
            "band": ["80+"] * 12 + ["40-64"] * 12,
            "patient": [1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9] + list(range(10, 22)),
            "readmitted": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [0] * 12,
        }
    )
    per_row = analysis.compare_rates(frame, "band", "readmitted")
    per_patient = analysis.compare_rates(frame, "band", "readmitted", "patient")

    assert per_row["unit_of_analysis"] == "row"
    assert "independence" in per_row, "counting rows without saying so is the old behaviour"
    assert per_patient["unit_of_analysis"] == "patient"
    assert per_patient["observations_counted"] == 21 < per_row["observations_counted"]
    assert "collapsed" in per_patient
    # One patient readmitted three times is one readmitted patient. Looked up by
    # name, because the groups are ordered by size and collapsing changes which
    # of them is larger — which is itself the point.
    older = next(group for group in per_patient["groups"] if group["group"] == "80+")
    assert older["events"] == 1 and older["observed"] == 9
    counted_as_rows = next(group for group in per_row["groups"] if group["group"] == "80+")
    assert counted_as_rows["events"] == 3 and counted_as_rows["observed"] == 12


def test_a_column_that_already_holds_a_rate_is_refused_with_the_reason() -> None:
    """The same shape of mistake compare_groups makes on an aggregate: handed a
    percentage, the denominator is already gone and no comparison can recover it."""
    frame = pd.DataFrame({"segment": ["a", "b"], "rate": [0.4375, 0.0735]})
    with pytest.raises(analysis.NotAnalysable, match="denominator"):
        analysis.compare_rates(frame, "segment", "rate")
