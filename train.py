"""
Assignment 2: train.py

Downloads the PCB-defect dataset from Roboflow Universe, finetunes yolo11n,
compares it against two baselines on the same held-out test split, evaluates
per class, builds the per-board scores the cost model needs, sweeps the
confidence threshold, and writes every artifact app.py's dashboard reads.

Run with:

    uv run python main.py train

This is the only place that touches the network, the GPU/CPU, and the
dataset. app.py never retrains anything -- it only reads what this script
writes to results/ and weights/.

Where this was trained and with what settings: fill in once you've run it
(see the printed summary at the end), and copy that line into REPORT.md.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

import costing
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"
DATA_DIR = ROOT / "data" / "pcb-defect"

# --- Roboflow Universe dataset this pipeline is built against -------------
# Recorded here AND in REPORT.md, so the pipeline is reproducible against the
# exact data it was built on.
RF_WORKSPACE = "yolo-wfx0p"
RF_PROJECT = "pcb-defect-uqoat"
RF_VERSION = 2  # https://universe.roboflow.com/yolo-wfx0p/pcb-defect-uqoat/dataset/2

MODEL_NAME = "yolo11n.pt"
EPOCHS = 60
IMGSZ = 640
BATCH = 16
PATIENCE = 15
SEED = 0

# Candidate detections below this aren't worth logging, but keep the sweep's
# inputs clean without throwing away anything a sane threshold would ever use.
MIN_LOG_CONF = 0.001
IOU_MATCH_THRESHOLD = 0.5  # board-level TP/FP matching, independent of val()'s own mAP matching


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()


# ---------------------------------------------------------------------------
# 1. Dataset: download + describe
# ---------------------------------------------------------------------------


def download_dataset() -> Path:
    """Pull the recorded Roboflow Universe version as a YOLO-format dataset.

    Uses the "yolov8" export deliberately: Ultralytics detection training
    reads the same images/ + labels/ + data.yaml layout for v5 through v11,
    so this is the most broadly supported export name and nothing about the
    data itself is version-specific.
    """
    from roboflow import Roboflow

    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    if (DATA_DIR / "data.yaml").exists():
        print(f"Dataset already present at {DATA_DIR}, skipping download.")
        return DATA_DIR

    rf = Roboflow(api_key=load_api_key())
    project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
    version = project.version(RF_VERSION)
    dataset = version.download("yolov8", location=str(DATA_DIR))
    return Path(dataset.location)


_SPLIT_FOLDER_NAMES = {"train": "train", "val": "valid", "test": "test"}


def resolve_split_images_dir(cfg: dict, data_yaml: Path, split: str) -> Path:
    """Roboflow's exported data.yaml is not a stable contract: depending on
    the SDK version, train/val/test show up as a plain "train/images", as a
    legacy "../train/images" meant to be stripped by the SDK (and sometimes
    isn't), or resolved against an explicit `path:` key. Rather than guess
    which convention this particular export used, try every one we've
    actually seen in practice and use whichever directory is real and
    actually has images in it.
    """
    base = data_yaml.parent
    split_key = cfg.get(split)
    candidates: list[Path] = []

    # The standard on-disk layout Roboflow produces: <root>/<split_folder>/images
    candidates.append(base / _SPLIT_FOLDER_NAMES[split] / "images")

    if split_key:
        candidates.append((base / split_key).resolve())  # literal path as given
        stripped = split_key.lstrip(".")  # strips a leading "../" down to "/train/images"
        if stripped:
            candidates.append((base / stripped.lstrip("/")).resolve())
        if cfg.get("path"):  # some exports use an explicit `path:` root
            root = Path(cfg["path"])
            if not root.is_absolute():
                root = (base / root).resolve()
            candidates.append((root / split_key).resolve())

    for candidate in candidates:
        if candidate.is_dir() and (any(candidate.glob("*.jpg")) or any(candidate.glob("*.png"))):
            return candidate

    tried = "\n".join(f"  - {c}" for c in dict.fromkeys(candidates))  # de-duplicated, ordered
    raise RuntimeError(
        f"Could not find any images for split '{split}' under {data_yaml}. Tried:\n{tried}\n"
        "Run `!cat` on the data.yaml and `!find` the dataset folder to see what's actually "
        "there, then add the right candidate to resolve_split_images_dir."
    )


def yolo_label_files(split_dir: Path) -> list[Path]:
    labels_dir = split_dir / "labels"
    return sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []


def describe_dataset(data_yaml: Path) -> dict:
    """Instances per class, split sizes, and object size relative to the
    image -- the numbers README step 3 asks us to interpret, not just report.
    """
    cfg = yaml.safe_load(data_yaml.read_text())
    names = cfg["names"] if isinstance(cfg["names"], list) else list(cfg["names"].values())

    stats = {"classes": names, "splits": {}}
    for split in ["train", "val", "test"]:
        if not cfg.get(split):
            continue
        images_dir = resolve_split_images_dir(cfg, data_yaml, split)
        # images/ and labels/ are siblings under the split folder.
        split_dir = images_dir.parent
        label_files = yolo_label_files(split_dir)
        n_images = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))

        instances_per_class = {name: 0 for name in names}
        box_area_fracs = []
        n_with_object = 0
        for lf in label_files:
            lines = [l for l in lf.read_text().strip().splitlines() if l.strip()]
            if lines:
                n_with_object += 1
            for line in lines:
                parts = line.split()
                cls_idx = int(parts[0])
                w, h = float(parts[3]), float(parts[4])
                instances_per_class[names[cls_idx]] += 1
                box_area_fracs.append(w * h)

        stats["splits"][split] = {
            "n_images": n_images,
            "n_images_with_object": n_with_object,
            "n_images_without_object": n_images - n_with_object,
            "instances_per_class": instances_per_class,
            "total_instances": sum(instances_per_class.values()),
            "mean_box_area_fraction": float(np.mean(box_area_fracs)) if box_area_fracs else None,
            "median_box_area_fraction": float(np.median(box_area_fracs)) if box_area_fracs else None,
        }
    return stats


# ---------------------------------------------------------------------------
# 2. Train + the two baselines
# ---------------------------------------------------------------------------


def train_model(data_yaml: Path) -> YOLO:
    model = YOLO(MODEL_NAME)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
        project=str(RUNS_DIR),
        name="pcb_defect",
        exist_ok=True,
        seed=SEED,
        deterministic=False,
    )
    return model


def results_dict_to_overall(results_dict: dict) -> dict:
    return {
        "precision": results_dict.get("metrics/precision(B)"),
        "recall": results_dict.get("metrics/recall(B)"),
        "mAP50": results_dict.get("metrics/mAP50(B)"),
        "mAP50-95": results_dict.get("metrics/mAP50-95(B)"),
    }


def per_class_table(metrics, class_names: dict) -> list[dict]:
    """Per-class precision/recall/AP50/AP50-95 from Ultralytics' own arrays --
    the numbers that let us say which class is dragging the average down.
    """
    box = metrics.box
    rows = []
    for i, cls_idx in enumerate(box.ap_class_index):
        rows.append(
            {
                "class": class_names[int(cls_idx)],
                "precision": float(box.p[i]),
                "recall": float(box.r[i]),
                "AP50": float(box.ap50[i]),
                "AP50-95": float(box.ap[i]) if hasattr(box, "ap") else float(box.maps[int(cls_idx)]),
            }
        )
    return rows


def evaluate_finetuned(model: YOLO, data_yaml: Path):
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        device=DEVICE,
        project=str(RUNS_DIR),
        name="pcb_defect_test",
        exist_ok=True,
    )
    overall = results_dict_to_overall(metrics.results_dict)
    per_class = per_class_table(metrics, model.names)
    return overall, per_class, metrics


def evaluate_coco_zero_shot(data_yaml: Path, our_class_names: list[str]) -> dict:
    """The pretrained-COCO baseline. COCO's 80 classes share no vocabulary with
    any PCB defect, so this model can never emit `missing_hole`, `short`, etc.
    Rather than assert that, we actually run it on the test images and confirm
    it empirically: for every test image, check whether any prediction (at a
    generous conf=0.1) lands on a class name overlapping ours. None can, by
    construction of COCO's label set -- measured, not assumed.
    """
    cfg = yaml.safe_load(data_yaml.read_text())
    test_images_dir = resolve_split_images_dir(cfg, data_yaml, "test")
    coco_model = YOLO(MODEL_NAME)  # pretrained, NOT the finetuned one

    n_images = 0
    n_overlapping_predictions = 0
    for img_path in list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png")):
        n_images += 1
        result = coco_model.predict(str(img_path), conf=0.1, device=DEVICE, verbose=False)[0]
        predicted_names = {coco_model.names[int(b.cls)] for b in result.boxes}
        if predicted_names & set(our_class_names):
            n_overlapping_predictions += 1

    return {
        "model": "yolo11n.pt pretrained on COCO, zero-shot",
        "n_test_images": n_images,
        "images_with_a_relevant_class_predicted": n_overlapping_predictions,
        "recall": 0.0 if n_images else float("nan"),
        "precision": float("nan"),  # undefined: it never predicts a defect class at all
        "note": (
            "COCO has no 'missing_hole', 'short', etc. among its 80 classes, so this "
            "model cannot score a true positive on this task by construction. "
            "Confirmed empirically above rather than assumed."
        ),
    }


def trivial_baseline_rates(y_true: np.ndarray) -> dict:
    """README's other trivial option: predicts nothing, ever. For a detection
    task this is the natural trivial baseline (vs. "majority class", which
    doesn't map cleanly onto bounding boxes). Every board passes; every real
    defect is missed.
    """
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    return {
        "model": "trivial baseline: always predicts 'no defect' (always pass)",
        "recall": 0.0,
        "fpr": 0.0,
        "n_test_images": int(len(y_true)),
        "n_positive": n_pos,
        "n_negative": n_neg,
    }


# ---------------------------------------------------------------------------
# 3. Board-level scores for the cost model: IoU-match every candidate
#    detection as a true or false positive, then a max-confidence score per
#    image for the pass/review/fail sweep.
# ---------------------------------------------------------------------------


def iou_xyxy(a, b) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    areaB = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def read_yolo_labels_as_xyxy(label_path: Path, img_w: int, img_h: int) -> list[tuple]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        _cls, cx, cy, w, h = (float(v) for v in line.split())
        boxes.append(
            (
                (cx - w / 2) * img_w,
                (cy - h / 2) * img_h,
                (cx + w / 2) * img_w,
                (cy + h / 2) * img_h,
            )
        )
    return boxes


def build_board_level_scores(model: YOLO, data_yaml: Path):
    """Walks the test split image by image. Returns:
    - eval_scores: one row per image -> has_defect_gt, max_confidence, n_gt, n_pred
    - detections_log: one row per candidate detection -> image, confidence, is_true_positive
    """
    cfg = yaml.safe_load(data_yaml.read_text())
    images_dir = resolve_split_images_dir(cfg, data_yaml, "test")
    labels_dir = images_dir.parent / "labels"

    image_rows = []
    detection_rows = []

    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    for img_path in image_paths:
        with Image.open(img_path) as im:
            w, h = im.size
        gt_boxes = read_yolo_labels_as_xyxy(labels_dir / f"{img_path.stem}.txt", w, h)

        result = model.predict(str(img_path), conf=MIN_LOG_CONF, device=DEVICE, verbose=False)[0]
        preds = sorted(
            ((tuple(b.xyxy[0].tolist()), float(b.conf)) for b in result.boxes),
            key=lambda x: -x[1],
        )

        claimed = [False] * len(gt_boxes)
        max_conf = 0.0
        for box, conf in preds:
            max_conf = max(max_conf, conf)
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gt_boxes):
                if claimed[j]:
                    continue
                v = iou_xyxy(box, gt)
                if v > best_iou:
                    best_iou, best_j = v, j
            is_tp = best_iou >= IOU_MATCH_THRESHOLD
            if is_tp:
                claimed[best_j] = True
            detection_rows.append({"image": img_path.name, "confidence": conf, "is_true_positive": is_tp})

        image_rows.append(
            {
                "image": img_path.name,
                "has_defect_gt": 1 if gt_boxes else 0,
                "max_confidence": max_conf,
                "n_gt_boxes": len(gt_boxes),
                "n_predictions": len(preds),
            }
        )

    return pd.DataFrame(image_rows), pd.DataFrame(detection_rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    started = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Dataset: {RF_WORKSPACE}/{RF_PROJECT} version {RF_VERSION}")

    dataset_dir = download_dataset()
    data_yaml = dataset_dir / "data.yaml"

    print("\n=== Describing the dataset ===")
    dataset_stats = describe_dataset(data_yaml)
    (RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(dataset_stats, indent=2))
    for split, s in dataset_stats["splits"].items():
        print(f"{split:<6} {s['n_images']:>5} images, {s['total_instances']:>5} instances")
    if all(s["n_images"] == 0 for s in dataset_stats["splits"].values()):
        raise RuntimeError(
            f"Found 0 images in every split under {data_yaml}. This almost always means "
            "the download didn't land where expected, or data.yaml's train/val/test paths "
            "don't resolve the way this script assumes. Delete the data/ folder and re-run "
            "so the download is retried, and if it persists, print(open(data_yaml).read()) "
            "and check it against what's actually on disk before training on an empty dataset."
        )

    class_names = dataset_stats["classes"]

    print("\n=== Training yolo11n ===")
    model = train_model(data_yaml)
    best_weights_path = Path(model.trainer.save_dir) / "weights" / "best.pt"
    model = YOLO(best_weights_path)  # reload the best checkpoint, not the last epoch

    print("\n=== Evaluating the finetuned model on the held-out test split ===")
    overall, per_class, _metrics = evaluate_finetuned(model, data_yaml)
    print("Overall:", overall)
    for row in per_class:
        print(f"  {row['class']:<20} P={row['precision']:.3f}  R={row['recall']:.3f}  AP50={row['AP50']:.3f}")

    print("\n=== Baseline 1: pretrained COCO, zero-shot ===")
    coco_baseline = evaluate_coco_zero_shot(data_yaml, class_names)
    print(coco_baseline)

    print("\n=== Building board-level scores for the cost model ===")
    eval_scores, detections_log = build_board_level_scores(model, data_yaml)
    eval_scores.to_csv(RESULTS_DIR / "eval_scores.csv", index=False)
    detections_log.to_csv(RESULTS_DIR / "detections_log.csv", index=False)

    trivial_baseline = trivial_baseline_rates(eval_scores["has_defect_gt"].to_numpy())
    print("\n=== Baseline 2: trivial (always pass) ===")
    print(trivial_baseline)

    print("\n=== Sweeping the confidence threshold for cost ===")
    costs = costing.CostAssumptions()
    y_true = eval_scores["has_defect_gt"].to_numpy()
    y_score = eval_scores["max_confidence"].to_numpy()

    n_negative_boards = int((y_true == 0).sum())
    if n_negative_boards == 0:
        print(
            "NOTE: 0 genuinely clean boards in the test split -- the board-level false "
            "positive rate is undefined at every threshold (see REPORT.md's dataset "
            "section). Using the detection-level false-alarm-rate proxy from "
            "detections_log.csv instead, so the sweep still reflects the false-positive "
            "side of the trade-off rather than silently discarding it."
        )
    best, sweep_rows = costing.best_binary_threshold_with_fpr_proxy(y_true, y_score, detections_log, costs)
    print(f"Best single threshold: {best['threshold']:.2f}  cost/1000={best['cost_per_1000']:.1f}")
    pd.DataFrame(sweep_rows).to_csv(RESULTS_DIR / "cost_sweep.csv", index=False)

    # Save everything app.py and REPORT.md need in one place.
    summary = {
        "device_used_for_training": DEVICE,
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "roboflow": {"workspace": RF_WORKSPACE, "project": RF_PROJECT, "version": RF_VERSION},
        "class_names": class_names,
        "finetuned": {"overall": overall, "per_class": per_class},
        "baseline_coco_zero_shot": coco_baseline,
        "baseline_trivial": trivial_baseline,
        "cost_assumptions": costs.__dict__,
        "n_negative_boards_in_test_split": n_negative_boards,
        "best_single_threshold": best,
        "training_seconds": time.time() - started,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))

    # Commit the weights the app and check.py will load.
    shutil.copy2(best_weights_path, WEIGHTS_DIR / "best.pt")

    print(f"\nDone in {time.time() - started:.1f}s.")
    print(f"Weights saved to {WEIGHTS_DIR / 'best.pt'}")
    print(f"Artifacts saved to {RESULTS_DIR}/")
    print(f"Trained on device: {DEVICE} -- copy this line into REPORT.md's finetuning section.")


if __name__ == "__main__":
    main()
