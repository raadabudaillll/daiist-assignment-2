"""
Assignment 2: app.py

A Gradio dashboard built entirely from what train.py already produced:
weights/best.pt and the artifacts in results/. Nothing here trains or
re-evaluates the model from scratch -- it only loads saved numbers and runs
inference on whatever image you hand it.

Run with:

    uv run python main.py app
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

import costing

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "weights" / "best.pt"
RESULTS_DIR = ROOT / "results"

if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(
        f"No weights at {WEIGHTS_PATH}. Run `uv run python main.py train` first -- "
        "this app only serves a model that has already been trained, it never trains one itself."
    )
for required in ["metrics.json", "eval_scores.csv", "cost_sweep.csv"]:
    if not (RESULTS_DIR / required).exists():
        raise FileNotFoundError(
            f"Missing results/{required}. Run `uv run python main.py train` first to produce it."
        )

model = YOLO(str(WEIGHTS_PATH))
metrics = json.loads((RESULTS_DIR / "metrics.json").read_text())
eval_scores = pd.read_csv(RESULTS_DIR / "eval_scores.csv")
cost_sweep = pd.read_csv(RESULTS_DIR / "cost_sweep.csv")
detections_log = pd.read_csv(RESULTS_DIR / "detections_log.csv")

Y_TRUE = eval_scores["has_defect_gt"].to_numpy()
Y_SCORE = eval_scores["max_confidence"].to_numpy()
N_NEGATIVE_BOARDS = int((Y_TRUE == 0).sum())

COSTS = costing.CostAssumptions(**metrics["cost_assumptions"])
DEFAULT_T_HI = float(metrics["best_single_threshold"]["threshold"])
DEFAULT_T_LO = max(0.01, DEFAULT_T_HI - 0.15)

CLASS_NAMES = list(model.names.values())


# ---------------------------------------------------------------------------
# Tab 1: try it on an image
# ---------------------------------------------------------------------------


def annotate(image: Image.Image, t_lo: float, t_hi: float):
    image = image.convert("RGB")
    result = model.predict(image, conf=min(0.05, t_lo), verbose=False)[0]
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    font = ImageFont.load_default(size=max(14, image.size[1] // 32))

    rows = []
    for box in sorted(result.boxes, key=lambda b: -float(b.conf)):
        conf = float(box.conf)
        label = model.names[int(box.cls)]
        x0, y0, x1, y1 = (round(v, 1) for v in box.xyxy[0].tolist())
        if conf >= t_hi:
            color, decision = "red", "FAIL"
        elif conf >= t_lo:
            color, decision = "orange", "REVIEW"
        else:
            color, decision = "gray", "pass (below review threshold)"
        if conf >= t_lo:  # only draw what would actually get a human or auto-reject's attention
            draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
            caption = f"{label} {conf:.2f}"
            draw.rectangle([x0, y0 - font.size - 6, x0 + len(caption) * font.size * 0.62, y0], fill=color)
            draw.text((x0 + 3, y0 - font.size - 4), caption, fill="white", font=font)
        rows.append({"class": label, "confidence": round(conf, 3), "decision": decision})

    board_decision = "PASS"
    if rows:
        best_conf = max(r["confidence"] for r in rows)
        if best_conf >= t_hi:
            board_decision = "FAIL"
        elif best_conf >= t_lo:
            board_decision = "SEND FOR HUMAN REVIEW"

    if rows:
        header = "| class | confidence | decision |\n|---|---|---|\n"
        body = "\n".join(f"| {r['class']} | {r['confidence']:.3f} | {r['decision']} |" for r in rows)
        table_md = header + body
    else:
        table_md = "*No defects detected above the review threshold.*"
    summary = f"### Board decision: **{board_decision}**\n\n{table_md}"
    return draw_img, summary


# ---------------------------------------------------------------------------
# Tab 2: operating point & cost
# ---------------------------------------------------------------------------


def confusion_df(t_lo: float, t_hi: float) -> pd.DataFrame:
    counts = costing.confusion_counts_three_way(Y_TRUE, Y_SCORE, t_lo, t_hi)
    rows = [["Actually defective", counts["defective_pass"], counts["defective_review"], counts["defective_fail"]]]
    if N_NEGATIVE_BOARDS:
        rows.append(["Actually good", counts["good_pass"], counts["good_review"], counts["good_fail"]])
    else:
        rows.append(["Actually good (0 in test split, see note above)", "-", "-", "-"])
    return pd.DataFrame(rows, columns=["", "Pass", "Review", "Fail"])


def cost_curve_plot(t_hi: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cost_sweep["threshold"], cost_sweep["cost_per_1000"], color="steelblue")
    ax.axvline(t_hi, color="red", linestyle="--", label=f"chosen threshold = {t_hi:.2f}")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Expected cost per 1,000 boards (EUR)")
    ax.set_title("Cost vs. threshold (two-outcome pass/fail sweep)")
    ax.legend()
    fig.tight_layout()
    return fig


def per_class_df() -> pd.DataFrame:
    rows = metrics["finetuned"]["per_class"]
    return pd.DataFrame(rows)[["class", "precision", "recall", "AP50", "AP50-95"]].round(3)


def update_operating_point(t_lo: float, t_hi: float):
    t_lo, t_hi = min(t_lo, t_hi), max(t_lo, t_hi)
    if N_NEGATIVE_BOARDS:
        rates = costing.three_way_rates(Y_TRUE, Y_SCORE, t_lo, t_hi)
    else:
        # No genuinely clean boards in this test split -- the "actually good" side
        # uses the detection-level false-alarm proxy instead (see REPORT.md).
        rates = costing.three_way_rates_with_fpr_proxy(Y_TRUE, Y_SCORE, detections_log, t_lo, t_hi)
    cost = costing.expected_cost_per_1000_three_way(rates, COSTS)
    share_review = costing.review_share(rates, COSTS.base_rate) * 100

    baseline_do_nothing = COSTS.base_rate * COSTS.cost_fn * 1000
    baseline_human_checks_everything = 1000 * (
        COSTS.cost_review_defective * COSTS.base_rate + COSTS.cost_review_good * (1 - COSTS.base_rate)
    )
    model_alone_cost = metrics["best_single_threshold"]["cost_per_1000"]

    comparison = pd.DataFrame(
        [
            ["Do nothing (pass everything)", round(baseline_do_nothing, 1)],
            ["Human checks every board", round(baseline_human_checks_everything, 1)],
            ["Model decides alone (best single threshold)", round(model_alone_cost, 1)],
            ["Model + human review band (this operating point)", round(cost, 1)],
        ],
        columns=["Strategy", "Cost per 1,000 boards (EUR)"],
    )

    summary_md = (
        f"**Expected cost per 1,000 boards:** €{cost:,.1f}  \n"
        f"**Share sent for human review:** {share_review:.1f}%  \n"
        f"**Assumed production base rate:** {COSTS.base_rate*100:.1f}% of boards genuinely defective"
    )
    if not N_NEGATIVE_BOARDS:
        summary_md += (
            "  \n*Note: this test split has 0 genuinely clean boards, so the false-accept "
            "side uses a detection-level false-alarm proxy rather than a measured rate "
            "-- see REPORT.md's dataset and limitations sections.*"
        )

    return (
        confusion_df(t_lo, t_hi),
        summary_md,
        cost_curve_plot(t_hi),
        per_class_df(),
        comparison,
    )


with gr.Blocks(title="PCB Defect Inspection Dashboard") as demo:
    gr.Markdown(
        "# PCB Defect Inspection Dashboard\n"
        "Finetuned `yolo11n` on the "
        f"[PCB Defect dataset]({'https://universe.roboflow.com/' + metrics['roboflow']['workspace'] + '/' + metrics['roboflow']['project']}) "
        f"(Roboflow Universe, version {metrics['roboflow']['version']}). "
        "Nothing on this page trains or re-evaluates the model; every number comes from the committed "
        "training run."
    )

    with gr.Tab("Try it on an image"):
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Upload a PCB photo")
                t_lo_img = gr.Slider(0.0, 1.0, value=DEFAULT_T_LO, step=0.01, label="Review threshold (t_lo)")
                t_hi_img = gr.Slider(0.0, 1.0, value=DEFAULT_T_HI, step=0.01, label="Fail threshold (t_hi)")
                run_btn = gr.Button("Inspect", variant="primary")
            with gr.Column():
                image_out = gr.Image(label="Detections")
                verdict_out = gr.Markdown()
        run_btn.click(annotate, inputs=[image_in, t_lo_img, t_hi_img], outputs=[image_out, verdict_out])

    with gr.Tab("Operating point & cost"):
        gr.Markdown(
            "Move the two thresholds to set the pass / human-review / fail boundaries. "
            "Everything below updates from the same held-out test split used to train the model."
        )
        with gr.Row():
            t_lo_slider = gr.Slider(0.0, 1.0, value=DEFAULT_T_LO, step=0.01, label="Review threshold (t_lo)")
            t_hi_slider = gr.Slider(0.0, 1.0, value=DEFAULT_T_HI, step=0.01, label="Fail threshold (t_hi)")

        cost_summary = gr.Markdown()

        with gr.Row():
            confusion_out = gr.Dataframe(label="Confusion matrix (test-split counts)")
            comparison_out = gr.Dataframe(label="This design vs. the three alternatives")

        with gr.Row():
            plot_out = gr.Plot(label="Cost vs. threshold")
            per_class_out = gr.Dataframe(label="Per-class results (test split)")

        for slider in (t_lo_slider, t_hi_slider):
            slider.change(
                update_operating_point,
                inputs=[t_lo_slider, t_hi_slider],
                outputs=[confusion_out, cost_summary, plot_out, per_class_out, comparison_out],
            )

        demo.load(
            update_operating_point,
            inputs=[t_lo_slider, t_hi_slider],
            outputs=[confusion_out, cost_summary, plot_out, per_class_out, comparison_out],
        )

    with gr.Tab("Model vs. baselines"):
        overall = metrics["finetuned"]["overall"]
        coco = metrics["baseline_coco_zero_shot"]
        trivial = metrics["baseline_trivial"]
        gr.Markdown(
            "### Overall, on the held-out test split\n"
            f"- **Finetuned yolo11n** -- precision {overall['precision']:.3f}, recall {overall['recall']:.3f}, "
            f"mAP50 {overall['mAP50']:.3f}, mAP50-95 {overall['mAP50-95']:.3f}\n"
            f"- **Pretrained COCO, zero-shot** -- recall {coco['recall']:.3f} "
            f"({coco['note']})\n"
            f"- **Trivial baseline (always pass)** -- recall {trivial['recall']:.3f}, "
            f"{trivial['n_positive']} real defects in the test split, all of them missed by construction\n"
        )
        gr.Dataframe(per_class_df(), label="Per-class results (finetuned model)")

if __name__ == "__main__":
    demo.launch()
