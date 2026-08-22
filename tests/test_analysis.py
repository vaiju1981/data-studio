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


def test_association_ranks_by_explained_variation_not_by_the_biggest_gap() -> None:
    """A tiny extreme group looks impressive by eye but explains almost nothing."""
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

    ranked = analysis.relate(frame, "target")["associations"]
    order = [item["column"] for item in ranked]
    assert order[0] == "strong_numeric"
    assert order.index("noise") > order.index("strong_numeric")
    strength = {item["column"]: item["strength"] for item in ranked}
    assert strength["strong_numeric"] > strength["rare_extreme"]


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
