"""Non-temporal analysis: group comparison, driver sweeps and association ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from smart_data_studio.config import (
    CROWDED_ABOVE,
    MAX_ASSOCIATIONS,
    MAX_COMPARISON_GROUPS,
    MAX_DRIVER_LEVELS,
    MAX_OUTLIERS_REPORTED,
    MAX_RELATE_SAMPLE,
    MAX_TEST_SAMPLE,
    MIN_ASSOCIATION_ROWS,
    MIN_COMPARISON_ROWS,
    MIN_OUTLIER_ENTITIES,
    OUTLIER_SCORE,
    SKEWED_ABOVE,
)

# Romano's conventions for Cliff's delta. Cohen's 0.2/0.5/0.8 belong to d and
# would call a delta of 0.5 "medium" where it is in fact large.
CLIFF_BANDS = ((0.474, "large"), (0.33, "medium"), (0.147, "small"))
SEED = 0
# What the nulls of a dimension are called once they become a level of their own.
MISSING_LEVEL = "(missing)"


class NotAnalysable(ValueError):
    """Raised when the columns given cannot support the analysis asked for."""


def _require(frame: pd.DataFrame, *columns: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise NotAnalysable(
            f"Column(s) not found: {', '.join(missing)}. Available: {list(frame.columns)}"
        )


def _sample(values: pd.Series, limit: int) -> tuple[pd.Series, bool]:
    if len(values) <= limit:
        return values, False
    return values.sample(limit, random_state=SEED), True


def _describe_effect(magnitude: float) -> str:
    for threshold, word in CLIFF_BANDS:
        if magnitude >= threshold:
            return word
    return "negligible"


def compare_groups(frame: pd.DataFrame, dimension: str, measure: str) -> dict[str, object]:
    """Test whether two groups really differ on a measure, and by how much.

    Effect size leads: at a few million rows every difference is significant, so a
    p-value alone rubber-stamps noise as a finding.
    """
    _require(frame, dimension, measure)
    values = pd.to_numeric(frame[measure], errors="coerce")
    if values.notna().sum() == 0:
        raise NotAnalysable(f"{measure} is not numeric")

    working = pd.DataFrame({dimension: frame[dimension], measure: values}).dropna()
    sizes = working.groupby(dimension)[measure].size().sort_values(ascending=False)
    if len(sizes) < 2:
        raise NotAnalysable(f"{dimension} has fewer than two groups with data")

    summary = (
        working.groupby(dimension)[measure]
        .agg(["size", "mean", "median", "std"])
        .reindex(sizes.index)
    )
    # Capped, and largest first, so the pair actually compared is always inside it.
    # Every tool here is budgeted, or one call fills the prompt on its own.
    shown = summary.head(MAX_COMPARISON_GROUPS)
    result: dict[str, object] = {
        "measure": measure,
        "dimension": dimension,
        "groups": [
            {
                "group": str(name),
                "rows": int(row["size"]),
                "mean": round(float(row["mean"]), 4),
                "median": round(float(row["median"]), 4),
                "std": round(float(row["std"]), 4) if pd.notna(row["std"]) else None,
            }
            for name, row in shown.iterrows()
        ],
    }
    if len(sizes) > 2:
        listed = (
            f"the {len(shown)} largest are listed here"
            if len(sizes) > len(shown)
            else "all of them are listed here"
        )
        result["note"] = (
            f"{dimension} has {len(sizes):,} groups; the two largest are compared and "
            f"{listed}. Filter to the pair you care about for a different comparison."
        )

    first, second = sizes.index[0], sizes.index[1]
    if int(sizes.iloc[0]) <= 1:
        # One row per group is not a small sample, it is a result that has already
        # been grouped, and the advice below sends the model somewhere it cannot
        # get to. Asked whether LOCAL and NATIONAL really differ, it queried the
        # average per geoType, handed the two rows to this test, and was told to
        # find groups with more data — so it tried again, nine times, until the
        # round limit ended the turn with no answer.
        raise NotAnalysable(
            f"This result holds one row per {dimension}, so it has already been "
            f"aggregated and there is nothing left to test — a mean has no spread. "
            f"Re-run the query returning one row per observation, selecting "
            f"{dimension} and {measure} without GROUP BY, then compare that."
        )
    if int(sizes.iloc[1]) < MIN_COMPARISON_ROWS:
        raise NotAnalysable(
            f"The second largest group in {dimension} has {int(sizes.iloc[1])} rows; at least "
            f"{MIN_COMPARISON_ROWS} are needed to tell a difference from noise. Filter to "
            "groups with enough data, or compare a coarser dimension."
        )
    left, sampled_left = _sample(working.loc[working[dimension] == first, measure], MAX_TEST_SAMPLE)
    right, sampled_right = _sample(
        working.loc[working[dimension] == second, measure], MAX_TEST_SAMPLE
    )
    result["compared"] = [str(first), str(second)]
    if sampled_left or sampled_right:
        result["sampling"] = (
            f"Tested on a random {MAX_TEST_SAMPLE:,} rows per group; the group summaries "
            "above cover every row."
        )

    left_values, right_values = left.to_numpy(), right.to_numpy()
    if left_values.std() == 0 and right_values.std() == 0:
        raise NotAnalysable(
            f"{measure} is constant within both groups, so there is nothing to test. The "
            f"values are {left_values[0]:g} and {right_values[0]:g}."
        )
    welch = stats.ttest_ind(left_values, right_values, equal_var=False)
    whitney = stats.mannwhitneyu(left_values, right_values, alternative="two-sided")

    pooled = np.sqrt((left_values.var(ddof=1) + right_values.var(ddof=1)) / 2)
    cohens_d = float((left_values.mean() - right_values.mean()) / pooled) if pooled else 0.0
    # Cliff's delta straight from U: no pairwise loop, and it survives the skew
    # that makes a mean-based effect size misleading.
    cliffs_delta = float(2 * whitney.statistic / (len(left_values) * len(right_values)) - 1)

    result["test"] = {
        "welch_t_p_value": float(welch.pvalue),
        "mann_whitney_p_value": float(whitney.pvalue),
        "cohens_d": round(cohens_d, 4),
        "cliffs_delta": round(cliffs_delta, 4),
        "effect": _describe_effect(abs(cliffs_delta)),
        "reading": (
            "Cliff's delta is the rank-based effect size and is the one to trust on skewed "
            "data. Judge importance by effect size; with large samples the p-value is near "
            "zero for differences too small to act on."
        ),
    }
    return result


# 95%, and stated in the result rather than assumed by whoever reads it.
CONFIDENCE_Z = 1.96


def _wilson(events: int, total: int) -> tuple[float, float]:
    """A confidence interval for a proportion that still works at small counts.

    The textbook p ± z·sqrt(p(1-p)/n) runs past 1 and below 0 near the ends and is
    badly wrong when there are few observations — which is exactly when anyone
    asks how precise a rate is. Wilson does not: 14 of 32 comes back as 28% to
    61%, and "43.75%" alone reads far more certain than that.
    """
    if total <= 0:
        return (0.0, 0.0)
    rate = events / total
    z2 = CONFIDENCE_Z**2
    denominator = 1 + z2 / total
    centre = (rate + z2 / (2 * total)) / denominator
    spread = (
        CONFIDENCE_Z * np.sqrt(rate * (1 - rate) / total + z2 / (4 * total * total)) / denominator
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def compare_rates(
    frame: pd.DataFrame, dimension: str, outcome: str, entity_column: str | None = None
) -> dict[str, object]:
    """Compare a binary outcome across groups, with the denominators in plain sight.

    `compare_groups` can be pointed at a 0/1 column and will run, and what it
    reports is Cliff's delta — a rank-based measure that on binary data collapses
    to the difference in proportions and is then read against thresholds built for
    continuous distributions. Measured on the fixtures: readmission of 30.2%
    against 6.4%, a 4.7-fold risk, came back as effect size "small"; defaults of
    43.8% against 7.3%, six-fold, came back "medium". Both answers were right
    about significance and wrong about importance, which is the more expensive way
    to be wrong.

    A proportion wants proportion measures — a difference in points, a ratio of
    risks, an odds ratio — and an interval, because the rate that prompts the
    question is usually the one resting on the fewest observations.

    `entity_column` is the unit of analysis. Given it, rows are collapsed to one
    per entity before anything is counted, because six encounters from one patient
    are one patient's outcome and counting them six times overstates both the rate
    and its precision. Left out, rows are the unit and the result says so rather
    than leaving a reader to assume.
    """
    _require(frame, dimension, outcome)
    if entity_column:
        _require(frame, entity_column)

    events = pd.to_numeric(frame[outcome], errors="coerce")
    working = pd.DataFrame({dimension: frame[dimension], outcome: events})
    if entity_column:
        working[entity_column] = frame[entity_column]
    working = working.dropna(subset=[dimension, outcome])

    present = set(pd.unique(working[outcome]))
    if not present <= {0, 1}:
        raise NotAnalysable(
            f"{outcome} is not a yes-or-no column — it holds {sorted(present)[:5]}. This "
            "compares how often something happened, so it needs one row per observation "
            "with a 1 where it did and a 0 where it did not. A column that already holds "
            "a rate has had its denominator thrown away and cannot be compared here."
        )

    rows = len(working)
    if entity_column:
        # An entity has to sit in exactly one group, or it is counted once in each
        # and the denominators quietly stop being entity counts. Three patients
        # where one changed ward came back as four observations, described as
        # per-patient, with no note saying otherwise — the collapsed note below
        # cannot fire, because the row count and the observation count agree.
        moved = working.groupby(entity_column)[dimension].nunique()
        divided = moved[moved > 1]
        if len(divided):
            raise NotAnalysable(
                f"{len(divided):,} {entity_column} value(s) appear under more than one "
                f"{dimension} — for example {str(divided.index[0])!r}. Counted per "
                f"{entity_column} each would fall into several groups at once, so the "
                f"denominators would no longer be {entity_column} counts. Decide in SQL "
                f"which {dimension} each {entity_column} belongs to — its first, its "
                f"latest, or the one being asked about — and compare that."
            )
        # Any occurrence counts once for the entity: a patient readmitted twice is
        # one readmitted patient, and this is what makes the rate per patient
        # differ from the rate per encounter rather than merely round differently.
        working = working.groupby([entity_column, dimension], as_index=False)[outcome].max()
        unit = entity_column
    else:
        unit = "row"

    counts = working.groupby(dimension)[outcome].agg(["sum", "size"])
    counts = counts.sort_values("size", ascending=False)
    if len(counts) < 2:
        raise NotAnalysable(f"{dimension} has fewer than two groups with data")

    groups = []
    for name, row in counts.head(MAX_COMPARISON_GROUPS).iterrows():
        total, hits = int(row["size"]), int(row["sum"])
        low, high = _wilson(hits, total)
        groups.append(
            {
                "group": str(name),
                "events": hits,
                "observed": total,
                "rate_pct": round(hits / total * 100, 2) if total else None,
                "interval_95_pct": [round(low * 100, 2), round(high * 100, 2)],
            }
        )

    result: dict[str, object] = {
        "outcome": outcome,
        "dimension": dimension,
        "unit_of_analysis": unit,
        "rows_read": rows,
        "observations_counted": int(counts["size"].sum()),
        "groups": groups,
    }
    if entity_column and rows != int(counts["size"].sum()):
        result["collapsed"] = (
            f"{rows:,} rows became {int(counts['size'].sum()):,} {entity_column} values. "
            f"Every figure here is per {entity_column}; per row it would be a different "
            "number, and neither is more correct than the other — they answer different "
            "questions."
        )
    elif not entity_column:
        result["independence"] = (
            "Counted per row, and rows are assumed independent. If several rows describe "
            "the same person, account or machine, pass entity_column so they are counted "
            "once — otherwise both the rate and its interval are overstated."
        )

    first, second = counts.index[0], counts.index[1]
    if int(counts["size"].iloc[1]) < MIN_COMPARISON_ROWS:
        result["note"] = (
            f"Only the rates are reported: the second largest group in {dimension} has "
            f"{int(counts['size'].iloc[1])} observations, and a comparison from that says "
            "more about the sample than the world."
        )
        return result

    a, n1 = int(counts["sum"].loc[first]), int(counts["size"].loc[first])
    c, n2 = int(counts["sum"].loc[second]), int(counts["size"].loc[second])
    result["compared"] = [str(first), str(second)]
    result["comparison"] = _rate_comparison(a, n1, c, n2)
    result["reading"] = (
        "Read the risk difference in points and the relative risk together: a rise from "
        "1% to 3% trebles the risk and moves 2 points, and which of those matters is the "
        "question's business, not this tool's. The intervals are Wilson, which holds at "
        "small counts where the textbook interval runs past 0 and 1. No effect-size band "
        "is offered on purpose — the rank-based one this replaces called a fourfold "
        "difference in readmission small."
    )
    return result


def _rate_comparison(a: int, n1: int, c: int, n2: int) -> dict[str, object]:
    """The two-by-two table, measured the way proportions are measured."""
    first_rate, second_rate = a / n1, c / n2
    found: dict[str, object] = {
        "risk_difference_pct_points": round((first_rate - second_rate) * 100, 2),
    }

    # A ratio needs both arms to have happened at least once; with a zero the log
    # interval is undefined and a corrected estimate would be inventing data.
    if a and c:
        ratio = first_rate / second_rate
        error = np.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n2)
        found["relative_risk"] = round(float(ratio), 3)
        found["relative_risk_interval_95"] = [
            round(float(ratio * np.exp(-CONFIDENCE_Z * error)), 3),
            round(float(ratio * np.exp(CONFIDENCE_Z * error)), 3),
        ]
    else:
        found["relative_risk"] = None
        found["relative_risk_note"] = "One group had no events, so a ratio is undefined."

    if a and c and a < n1 and c < n2:
        odds = (a / (n1 - a)) / (c / (n2 - c))
        found["odds_ratio"] = round(float(odds), 3)

    table = [[a, n1 - a], [c, n2 - c]]
    # An outcome that happened to nobody in either group, or to everybody in both,
    # leaves a column of the two-by-two empty. chi2_contingency raises on the zero
    # expected frequency rather than returning anything, and the whole comparison
    # came back as a tool error — for two rates that are identical, which is a
    # perfectly good answer and the one the question was asking about.
    if not (a + c) or not (n1 - a) + (n2 - c):
        found["p_value"] = None
        found["method"] = (
            "No test: the outcome is the same for every observation in both groups, so "
            "the rates are identical and there is no difference to test."
        )
        return found
    expected = stats.chi2_contingency(table)[3] if min(n1, n2) > 0 else None
    if expected is not None and expected.min() >= 5:
        chi = stats.chi2_contingency(table, correction=False)
        found["p_value"] = float(chi[1])
        found["method"] = "chi-square on the two-by-two table"
    else:
        found["p_value"] = float(stats.fisher_exact(table)[1])
        found["method"] = "Fisher's exact test, because a cell is too thin for chi-square"
    return found


def rank_drivers(frame: pd.DataFrame, measure: str, split: str) -> dict[str, object]:
    """Sweep every usable dimension and rank what moved a measure between two sides.

    Sweeping is the point: asked by hand, the model checks whichever dimension it
    thought of and misses the one that explains the move.
    """
    _require(frame, measure, split)
    sides = frame[split].dropna().unique()
    if len(sides) != 2:
        raise NotAnalysable(
            f"{split} must hold exactly two values to compare, found {len(sides)}: "
            f"{list(sides)[:5]}"
        )
    values = pd.to_numeric(frame[measure], errors="coerce")
    if values.notna().sum() == 0:
        raise NotAnalysable(f"{measure} is not numeric")

    working = frame.assign(**{measure: values})
    # Order of appearance, not alphabetical: the query's own ORDER BY puts the
    # earlier side first, whereas sorting would rank "after" before "before".
    before, after = sides[0], sides[1]

    # A row with no side belongs to neither total and is left out of every pivot
    # below, so it is counted here and named in the result rather than dropped.
    sideless = int(frame[split].isna().sum())
    # Nulls become a level of their own, for the reason _as_level gives. The label
    # each column ended up using is kept, since it is not always the same one.
    missing_labels: dict[str, str] = {}
    for column in [name for name in working.columns if name not in {measure, split}]:
        working[column], label = _as_level(working[column])
        if label:
            missing_labels[column] = label

    totals = working.groupby(split)[measure].sum()
    overall = float(totals.get(after, 0.0) - totals.get(before, 0.0))

    candidates = [
        column
        for column in working.columns
        if column not in {measure, split} and 1 < working[column].nunique() <= MAX_DRIVER_LEVELS
    ]
    dimensions = []
    for column in candidates:
        pivot = working.pivot_table(
            index=column, columns=split, values=measure, aggfunc="sum", fill_value=0.0
        )
        if before not in pivot.columns or after not in pivot.columns:
            continue
        change = (pivot[after] - pivot[before]).sort_values()
        # head and tail overlap at six levels or fewer, listing every mover twice.
        shown = change if len(change) <= 6 else pd.concat([change.head(3), change.tail(3)])
        movers = [
            {"level": str(level), "change": round(float(delta), 2)}
            for level, delta in shown.items()
        ]
        spread = float(change.abs().sum())
        largest = float(change.abs().max())
        dimensions.append(
            {
                "dimension": column,
                "levels": int(len(change)),
                # The biggest single move is what a reader acts on. Spread is total
                # churn, which ranks a dimension high merely for having many levels.
                "largest_move": round(largest, 2),
                "concentration": round(largest / spread, 3) if spread else None,
                # Against an even split of the churn. A dimension with many levels
                # gets a large top move for free; lift near 1 means exactly that.
                "lift_over_uniform": (
                    round(largest / (spread / len(change)), 2) if spread else None
                ),
                "spread": round(spread, 2),
                "movers": sorted(movers, key=lambda item: item["change"]),
            }
        )
        # Named rather than left to be inferred. Where the column already holds the
        # string "(missing)", the nulls are labelled something else — and a reader
        # who assumes the plain one reads a real value as the missing bucket.
        if column in missing_labels:
            dimensions[-1]["missing_level"] = missing_labels[column]

    dimensions.sort(
        key=lambda item: (item["largest_move"], item["lift_over_uniform"] or 0), reverse=True
    )
    result = {
        "measure": measure,
        "comparing": {"from": str(before), "to": str(after)},
        "total_change": round(overall, 2),
        "dimensions_swept": len(dimensions),
        "skipped": [
            column for column in working.columns if column not in candidates + [measure, split]
        ],
        "drivers": dimensions[:6],
        "reading": (
            "Ranked by the largest single level movement. concentration is that move as "
            "a share of all movement in the dimension, and lift_over_uniform compares it "
            "with an even split across levels — a many-levelled dimension earns a big "
            "top move for free, and lift near 1 shows that is all it is. Positive "
            "changes are gains, negative are losses; within a dimension they sum to the "
            "total change. Each dimension is that same change sliced another way, not a "
            "separate part of it: a move here and a move there are the same movement "
            "counted twice, so never add across dimensions, and a large move locates "
            "the change rather than explaining it. A dimension carrying "
            "missing_level has nulls, and that is the level they were gathered into."
        ),
    }
    if sideless:
        result["rows_without_a_side"] = (
            f"{sideless:,} row(s) hold no {split} and belong to neither side, so they are "
            "in none of the movements above and not in the total change either."
        )
    return result


def _as_level(series: pd.Series) -> tuple[pd.Series, str | None]:
    """Nulls as a level of their own, because pivot_table drops a NaN index.

    Left as NaN, every row with a missing value for that dimension leaves the
    sweep — while the reading below promises that a dimension's levels sum to the
    total change. A fixture whose null region carried 100 of a 110 change reported
    the 110 and showed movements totalling 10, with nothing saying where the rest
    had gone.

    The label has to be one the column does not already use. A column holding
    nulls and the literal string "(missing)" collapsed into a single level and was
    dropped from the sweep as constant — losing the whole dimension to the fix for
    losing part of it.
    """
    if not series.isna().any():
        return series, None
    label = _missing_label(series)
    return series.astype(object).where(series.notna(), label), label


def _missing_label(series: pd.Series) -> str:
    """MISSING_LEVEL, widened until no real value in the column answers to it."""
    taken = {str(value) for value in series.dropna().unique()}
    label, suffix = MISSING_LEVEL, 2
    while label in taken:
        label = f"{MISSING_LEVEL} #{suffix}"
        suffix += 1
    return label


def relate(frame: pd.DataFrame, target: str) -> dict[str, object]:
    """Rank every column by how strongly it is associated with a target column.

    Association, not the biggest gap: a small group with an extreme mean looks
    impressive while explaining almost none of the variation.
    """
    _require(frame, target)
    values = pd.to_numeric(frame[target], errors="coerce")
    if values.notna().sum() < 10:
        raise NotAnalysable(f"{target} needs at least 10 numeric values")

    working = frame.assign(**{target: values}).dropna(subset=[target])
    sampled = len(working) > MAX_RELATE_SAMPLE
    if sampled:
        working = working.sample(MAX_RELATE_SAMPLE, random_state=SEED)
    outcome = working[target]

    scored = []
    for column in working.columns:
        if column == target:
            continue
        series = working[column]
        if series.nunique() < 2:
            continue
        # Computed on the rows where both columns are present, so a sparse column is
        # scored on what it has rather than borrowing the target's row count.
        paired = pd.DataFrame({"value": series, "target": outcome}).dropna()
        if len(paired) < MIN_ASSOCIATION_ROWS or paired["value"].nunique() < 2:
            continue
        if pd.api.types.is_numeric_dtype(series):
            rho = stats.spearmanr(paired["value"], paired["target"]).statistic
            if pd.isna(rho):
                continue
            scored.append(
                {
                    "column": column,
                    "kind": "numeric",
                    "strength": round(abs(float(rho)), 4),
                    "rows_used": int(len(paired)),
                    "detail": f"Spearman rho {float(rho):.4f}",
                }
            )
        elif paired["value"].nunique() <= MAX_DRIVER_LEVELS:
            groups = paired.groupby("value")["target"]
            grand = float(paired["target"].mean())
            between = float(((groups.mean() - grand) ** 2 * groups.size()).sum())
            total = float(((paired["target"] - grand) ** 2).sum())
            if total <= 0:
                continue
            scored.append(
                {
                    "column": column,
                    "kind": "categorical",
                    "strength": round(between / total, 4),
                    "rows_used": int(len(paired)),
                    "detail": (
                        f"eta squared {between / total:.4f} over {paired['value'].nunique()} levels"
                    ),
                }
            )

    # Ranked apart, because one list sorted by "strength" claims the two measures
    # are on one scale and they are not. Both run 0 to 1 and that is the whole of
    # what they share: a Spearman rho of 0.4 and an eta squared of 0.4 are not the
    # same finding, and eta squared climbs with the number of levels a column has,
    # so a high-cardinality column outranks a numeric one for having more of them.
    numeric = sorted(
        (item for item in scored if item["kind"] == "numeric"),
        key=lambda item: item["strength"],
        reverse=True,
    )
    categorical = sorted(
        (item for item in scored if item["kind"] == "categorical"),
        key=lambda item: item["strength"],
        reverse=True,
    )
    return {
        "target": target,
        "rows_used": int(len(working)),
        "sampled": sampled,
        "numeric_associations": numeric[:MAX_ASSOCIATIONS],
        "categorical_associations": categorical[:MAX_ASSOCIATIONS],
        "reading": (
            "Two different measures, ranked separately because they do not share a "
            "scale: absolute Spearman correlation for numeric columns, eta squared — "
            "the share of variation the column explains — for categorical ones. Eta "
            "squared also rises with the number of levels, so read each list against "
            "itself and do not place a categorical column above a numeric one on the "
            "number alone. Both measure association, not cause."
        ),
    }


def find_outliers(frame: pd.DataFrame, dimension: str, measure: str) -> dict[str, object]:
    """Which entities stand apart from the rest of their population on a measure.

    Asked which machines behaved unusually, the model ranked by the largest raw
    gap — so the busiest machine won whatever it was doing, and the question went
    unanswered. Standing apart is a question about distance from the rest, not
    about size, and it needs the rest to be measured.

    Median and median absolute deviation rather than mean and standard deviation:
    an outlier inflates both of those, which is how it hides behind them. Each
    entity is summarised by its mean, so a result already carrying one row per
    entity is used as it stands.
    """
    _require(frame, dimension, measure)
    values = pd.to_numeric(frame[measure], errors="coerce")
    if values.notna().sum() == 0:
        raise NotAnalysable(f"{measure} is not numeric")

    working = pd.DataFrame({dimension: frame[dimension], measure: values}).dropna()
    grouped = working.groupby(dimension)[measure]
    per_entity, seen = grouped.mean(), grouped.size()
    if len(per_entity) < MIN_OUTLIER_ENTITIES:
        raise NotAnalysable(
            f"{dimension} has {len(per_entity)} values with data; at least "
            f"{MIN_OUTLIER_ENTITIES} are needed before one can stand apart from the rest. "
            "Compare them directly instead."
        )

    middle = float(per_entity.median())
    deviation = float((per_entity - middle).abs().median())
    if deviation == 0:
        raise NotAnalysable(
            f"More than half of {dimension} share the same {measure}, so there is no spread "
            "to stand apart from. Aggregate differently, or compare the groups directly."
        )
    # 0.6745 is the MAD of a standard normal, so a score reads on the same scale as
    # a z-score for data that is normal and stays meaningful for data that is not.
    score = 0.6745 * (per_entity - middle) / deviation
    flagged = score[score.abs() >= OUTLIER_SCORE].sort_values(key=abs, ascending=False)

    skew = float(per_entity.skew()) if len(per_entity) > 2 else 0.0
    result: dict[str, object] = {
        "dimension": dimension,
        "measure": measure,
        "entities": int(len(per_entity)),
        "median": round(middle, 4),
        "median_absolute_deviation": round(deviation, 4),
        "flagged": int(len(flagged)),
        "outliers": [
            {
                "entity": str(name),
                "value": round(float(per_entity[name]), 4),
                "observations": int(seen[name]),
                "score": round(float(score[name]), 2),
                "direction": "high" if score[name] > 0 else "low",
            }
            for name in flagged.index[:MAX_OUTLIERS_REPORTED]
        ],
        "reading": (
            f"score is distance from the median of all {len(per_entity):,} in units of the "
            f"median absolute deviation; anything past {OUTLIER_SCORE} is reported. It "
            "measures distance from the rest, not size, so the largest entity is not "
            "flagged for being large. Read observations before acting on one: an "
            "entity seen once is far from the median because it has had no chance "
            "to average out, which is a different thing from behaving unusually."
        ),
    }
    if len(flagged) > MAX_OUTLIERS_REPORTED:
        result["note"] = (
            f"{len(flagged):,} entities passed the threshold; the {MAX_OUTLIERS_REPORTED} "
            "furthest out are listed."
        )
    crowded = len(flagged) / len(per_entity) > CROWDED_ABOVE
    if abs(skew) >= SKEWED_ABOVE and crowded:
        result["skew_warning"] = (
            f"{measure} is heavily skewed ({skew:.1f}), so most of what stands out is the "
            "head of a long tail rather than anything anomalous. A rate or a ratio — the "
            "measure divided by whatever drives its size — usually answers 'unusual' better "
            "than a total does."
        )
    return result
