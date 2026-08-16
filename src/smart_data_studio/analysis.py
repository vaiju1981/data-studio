"""Non-temporal analysis: group comparison, driver sweeps and association ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from smart_data_studio.config import MAX_DRIVER_LEVELS, MAX_RELATE_SAMPLE, MAX_TEST_SAMPLE

# Cohen's conventions, used only to translate a number into a word.
EFFECT_BANDS = ((0.8, "large"), (0.5, "medium"), (0.2, "small"))
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
    for threshold, word in EFFECT_BANDS:
        if magnitude >= threshold:
            return word
    return "negligible"


def compare_groups(frame: pd.DataFrame, dimension: str, measure: str) -> dict[str, object]:
    """Test whether two groups really differ on a measure, and by how much.

    Effect size leads, because significance does not survive contact with large
    data: at a few million rows every difference is "significant", so a p-value
    alone would rubber-stamp noise as a finding.
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
            for name, row in summary.iterrows()
        ],
    }
    if len(sizes) > 2:
        result["note"] = (
            f"{dimension} has {len(sizes)} groups; the two largest are compared. "
            "Filter to the pair you care about for a different comparison."
        )

    first, second = sizes.index[0], sizes.index[1]
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
    thought of and misses the one that actually explains the move.
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
        movers = [
            {"level": str(level), "change": round(float(delta), 2)}
            for level, delta in list(change.head(3).items()) + list(change.tail(3).items())
        ]
        dimensions.append(
            {
                "dimension": column,
                "levels": int(len(change)),
                # How concentrated the movement is: a dimension where one level
                # explains the whole move is more informative than one spread thin.
                "spread": round(float(change.abs().sum()), 2),
                "movers": sorted(movers, key=lambda item: item["change"]),
            }
        )

    dimensions.sort(key=lambda item: item["spread"], reverse=True)
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
            "Dimensions are ranked by how much movement they account for. Positive "
            "changes are gains, negative are losses; within a dimension they sum to the "
            "total change."
        ),
    }


def relate(frame: pd.DataFrame, target: str) -> dict[str, object]:
    """Rank every column by how strongly it is associated with a target column.

    Association, not the size of the biggest gap: a small group with an extreme
    mean looks impressive by eye while explaining almost none of the variation.
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
        if pd.api.types.is_numeric_dtype(series):
            paired = pd.DataFrame({"a": series, "b": outcome}).dropna()
            if len(paired) < 10:
                continue
            rho = stats.spearmanr(paired["a"], paired["b"]).statistic
            if pd.isna(rho):
                continue
            scored.append(
                {
                    "column": column,
                    "kind": "numeric",
                    "strength": round(abs(float(rho)), 4),
                    "detail": f"Spearman rho {float(rho):.4f}",
                }
            )
        elif series.nunique() <= MAX_DRIVER_LEVELS:
            groups = outcome.groupby(series)
            grand = float(outcome.mean())
            between = float(((groups.mean() - grand) ** 2 * groups.size()).sum())
            total = float(((outcome - grand) ** 2).sum())
            if total <= 0:
                continue
            scored.append(
                {
                    "column": column,
                    "kind": "categorical",
                    "strength": round(between / total, 4),
                    "detail": f"eta squared {between / total:.4f} over {series.nunique()} levels",
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
