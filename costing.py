"""
Assignment 2: the cost model behind the threshold and the human-review split.

This is the file steps 6 and 7 of README.md actually live in. Everything here is
plain numpy on two small, precomputed arrays (`y_true`, `y_score`), one row per
test-set board, so it is cheap enough to re-run on every slider move in app.py
without touching the model or retraining anything.

The business framing (see REPORT.md for the full argument):

  The decision: after etching/soldering, a camera photographs each board and a
  model decides pass / send-to-a-person / fail, before a human inspector or the
  packaging line ever sees it.

  The asymmetry:
    - A board that is actually defective and ships anyway (a miss, a false
      negative) can cause a field failure, a warranty claim, or a recall in
      whatever product the board ends up inside.                 -> COST_FN, high
    - A board that is actually fine but gets auto-rejected (a false alarm, a
      false positive) just costs a technician a couple of minutes confirming
      it's fine before it goes back on the line.                  -> COST_FP, low
    - A board that is actually defective and gets correctly caught still costs
      the pull plus the rework/scrap.                             -> COST_TP
    - A board that is actually fine and correctly passes costs nothing extra. -> COST_TN
    - A board sent to a person for review costs a fixed review time, whether
      it turns out to be good or bad, and we assume the person gets it right.

  The base rate: how often a board is genuinely defective in production. The
  test split is NOT a reliable source for this, because the dataset is built
  specifically to contain many labelled defects (see "What the dataset's
  composition implies" in REPORT.md) -- a real PCB line runs at a far lower
  defect rate than an annotated benchmark does. BASE_RATE below is therefore an
  assumption, stated explicitly, and REPORT.md's costing section says what
  happens to the answer if it is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostAssumptions:
    """All figures in EUR per board, plus the assumed production base rate.

    Defended in REPORT.md's "Cost matrix and operating point" section --
    nothing here should be taken as the real number without reading why.
    """

    cost_fn: float = 120.0  # defective board ships, undetected
    cost_fp: float = 5.0  # good board wrongly auto-rejected (tech. re-checks it)
    cost_tp: float = 15.0  # defective board correctly auto-rejected (pull + rework)
    cost_tn: float = 0.0  # good board correctly passed
    cost_review_defective: float = 20.0  # defective board sent to review (review + rework)
    cost_review_good: float = 5.0  # good board sent to review (review time only)
    base_rate: float = 0.03  # assumed fraction of boards genuinely defective in production


# ---------------------------------------------------------------------------
# Two-outcome (pass / fail) sweep -- this is what README step 6 asks for:
# one threshold, cost against threshold, take the minimum.
# ---------------------------------------------------------------------------


def binary_outcome_rates(y_true: np.ndarray, y_score: np.ndarray, threshold: float):
    """At one accept/reject cutoff: recall (share of real defects caught) and
    the false-accept-side rate (share of good boards wrongly rejected)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_true == 1
    neg = ~pos
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    recall = float((y_score[pos] >= threshold).sum() / n_pos) if n_pos else float("nan")
    fpr = float((y_score[neg] >= threshold).sum() / n_neg) if n_neg else float("nan")
    return recall, fpr


def expected_cost_per_1000_binary(recall: float, fpr: float, costs: CostAssumptions) -> float:
    p = costs.base_rate
    recall = 0.0 if np.isnan(recall) else recall
    fpr = 0.0 if np.isnan(fpr) else fpr
    tp = p * recall
    fn = p * (1 - recall)
    fp = (1 - p) * fpr
    tn = (1 - p) * (1 - fpr)
    cost = tp * costs.cost_tp + fn * costs.cost_fn + fp * costs.cost_fp + tn * costs.cost_tn
    return cost * 1000


def sweep_binary_threshold(y_true, y_score, costs: CostAssumptions, thresholds=None):
    """One row per threshold: recall, false-accept rate, and cost per 1,000 boards."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in thresholds:
        recall, fpr = binary_outcome_rates(y_true, y_score, t)
        cost = expected_cost_per_1000_binary(recall, fpr, costs)
        rows.append({"threshold": float(t), "recall": recall, "fpr": fpr, "cost_per_1000": cost})
    return rows


def best_binary_threshold(y_true, y_score, costs: CostAssumptions, thresholds=None):
    rows = sweep_binary_threshold(y_true, y_score, costs, thresholds)
    best = min(rows, key=lambda r: r["cost_per_1000"])
    return best, rows


# ---------------------------------------------------------------------------
# Three-outcome (pass / review / fail) model -- README step 7.
# t_lo and t_hi are confidences; t_lo <= t_hi always.
#   score <  t_lo            -> pass
#   t_lo <= score < t_hi     -> send for human review
#   score >= t_hi            -> fail (auto-reject)
# ---------------------------------------------------------------------------


def three_way_rates(y_true, y_score, t_lo: float, t_hi: float) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_true == 1
    neg = ~pos

    def frac(mask, cond):
        n = int(mask.sum())
        return float((cond & mask).sum() / n) if n else float("nan")

    return {
        "fail_pos": frac(pos, y_score >= t_hi),  # caught automatically
        "review_pos": frac(pos, (y_score >= t_lo) & (y_score < t_hi)),
        "pass_pos": frac(pos, y_score < t_lo),  # missed
        "fail_neg": frac(neg, y_score >= t_hi),  # wrongly auto-rejected
        "review_neg": frac(neg, (y_score >= t_lo) & (y_score < t_hi)),
        "pass_neg": frac(neg, y_score < t_lo),  # correctly passed
    }


def _z(v: float) -> float:
    return 0.0 if (v is None or np.isnan(v)) else v


def expected_cost_per_1000_three_way(rates: dict, costs: CostAssumptions) -> float:
    p = costs.base_rate
    cost = (
        p * _z(rates["fail_pos"]) * costs.cost_tp
        + p * _z(rates["review_pos"]) * costs.cost_review_defective
        + p * _z(rates["pass_pos"]) * costs.cost_fn
        + (1 - p) * _z(rates["fail_neg"]) * costs.cost_fp
        + (1 - p) * _z(rates["review_neg"]) * costs.cost_review_good
        + (1 - p) * _z(rates["pass_neg"]) * costs.cost_tn
    )
    return cost * 1000


def review_share(rates: dict, base_rate: float) -> float:
    p = base_rate
    return p * _z(rates["review_pos"]) + (1 - p) * _z(rates["review_neg"])


def confusion_counts_three_way(y_true, y_score, t_lo: float, t_hi: float) -> dict:
    """Raw counts on the actual test set (not rates), for display."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    decision = np.where(y_score >= t_hi, "fail", np.where(y_score >= t_lo, "review", "pass"))
    pos = y_true == 1
    counts = {}
    for label, mask in [("defective", pos), ("good", ~pos)]:
        for d in ["pass", "review", "fail"]:
            counts[f"{label}_{d}"] = int(((decision == d) & mask).sum())
    return counts


# ---------------------------------------------------------------------------
# False-accept-rate proxy, for when the test set has no (or very few) clean
# boards -- see the note in REPORT.md's "Dataset" section. This dataset is
# built to contain labelled defects, so a genuinely clean board is rare to
# nonexistent in it. We approximate "how often would a clean board be flagged
# anyway" with the model's per-image spurious-detection rate: does the model
# raise an extra box that does NOT correspond to any real defect, at this
# confidence, regardless of whether it also (separately) found the real ones.
# This is flagged explicitly as an approximation, with its own line in the
# limitations section, not presented as a measured false-accept rate.
# ---------------------------------------------------------------------------


def false_alarm_rate_proxy(detections_df, threshold: float) -> float:
    """detections_df: one row per candidate detection, columns
    ['image', 'confidence', 'is_true_positive']. Returns the share of images
    with at least one false-positive-flagged detection scoring >= threshold.
    """
    if detections_df.empty:
        return float("nan")
    hits = detections_df[(detections_df["confidence"] >= threshold) & (~detections_df["is_true_positive"])]
    n_images = detections_df["image"].nunique()
    return float(hits["image"].nunique() / n_images) if n_images else float("nan")


# ---------------------------------------------------------------------------
# This dataset turned out to contain zero genuinely clean boards in its test
# split (every image has at least one real defect, confirmed empirically:
# the trivial baseline's n_negative is 0). That means `binary_outcome_rates`
# and `three_way_rates` above can never observe a false positive on a clean
# board -- their fpr/fail_neg/review_neg/pass_neg always come back NaN, which
# collapses the cost sweep onto "whatever maximizes recall" and silently
# throws away the false-positive side of the trade-off the assignment is
# actually about. These _with_fpr_proxy variants use the detection-level
# false-alarm rate above as the stand-in for "how often would a clean board
# get flagged", so the sweep is genuinely threshold-sensitive on both sides.
# This substitution is a stated limitation (see REPORT.md), not a hidden one.
# ---------------------------------------------------------------------------


def sweep_binary_threshold_with_fpr_proxy(y_true, y_score, detections_df, costs: CostAssumptions, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in thresholds:
        recall, _ = binary_outcome_rates(y_true, y_score, t)
        fpr_proxy = false_alarm_rate_proxy(detections_df, t)
        cost = expected_cost_per_1000_binary(recall, fpr_proxy, costs)
        rows.append({"threshold": float(t), "recall": recall, "fpr_proxy": fpr_proxy, "cost_per_1000": cost})
    return rows


def best_binary_threshold_with_fpr_proxy(y_true, y_score, detections_df, costs: CostAssumptions, thresholds=None):
    rows = sweep_binary_threshold_with_fpr_proxy(y_true, y_score, detections_df, costs, thresholds)
    best = min(rows, key=lambda r: r["cost_per_1000"])
    return best, rows


def three_way_rates_with_fpr_proxy(y_true, y_score, detections_df, t_lo: float, t_hi: float) -> dict:
    """Like three_way_rates, but the 'actually good' side comes from the
    detection-level false-alarm proxy instead of an empirical negative class
    this dataset doesn't have."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_true == 1

    def frac(mask, cond):
        n = int(mask.sum())
        return float((cond & mask).sum() / n) if n else float("nan")

    fa_hi = false_alarm_rate_proxy(detections_df, t_hi)  # proxy: P(good board auto-rejected)
    fa_lo = false_alarm_rate_proxy(detections_df, t_lo)  # proxy: P(good board flagged at all, >= t_lo)
    fa_hi = 0.0 if np.isnan(fa_hi) else fa_hi
    fa_lo = 0.0 if np.isnan(fa_lo) else fa_lo

    return {
        "fail_pos": frac(pos, y_score >= t_hi),
        "review_pos": frac(pos, (y_score >= t_lo) & (y_score < t_hi)),
        "pass_pos": frac(pos, y_score < t_lo),
        "fail_neg": fa_hi,
        "review_neg": max(0.0, fa_lo - fa_hi),
        "pass_neg": max(0.0, 1 - fa_lo),
    }
