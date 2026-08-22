"""Non-temporal analysis: group comparison, driver sweeps and association ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from smart_data_studio.config import (
    CROWDED_ABOVE,
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

    dimensions.sort(
        key=lambda item: (item["largest_move"], item["lift_over_uniform"] or 0), reverse=True
    )
    return {
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
            "total change."
        ),
    }


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

    scored.sort(key=lambda item: item["strength"], reverse=True)
    return {
        "target": target,
        "rows_used": int(len(working)),
        "sampled": sampled,
        "associations": scored[:15],
        "reading": (
            "Strength runs 0 to 1 and is comparable across columns: absolute Spearman "
            "correlation for numeric columns, eta squared — the share of variation the "
            "column explains — for categorical ones. It measures association, not cause."
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
    per_entity = working.groupby(dimension)[measure].mean()
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
                "score": round(float(score[name]), 2),
                "direction": "high" if score[name] > 0 else "low",
            }
            for name in flagged.index[:MAX_OUTLIERS_REPORTED]
        ],
        "reading": (
            f"score is distance from the median of all {len(per_entity):,} in units of the "
            f"median absolute deviation; anything past {OUTLIER_SCORE} is reported. It "
            "measures distance from the rest, not size, so the largest entity is not "
            "flagged for being large."
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
