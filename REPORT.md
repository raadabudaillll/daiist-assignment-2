# Assignment 2 Report

- **Name**: Raad Abu Dail
- **Student ID**: 18940
- **Email**: rabdulhameed.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

After etching and soldering, a camera takes a picture of each printed circuit 
board before it proceeds to packaging. In current practice, either a human 
visually inspects each board or the boards go directly to packaging with only 
spot checks performed. The decision that the model is trying to assist with is
whether the printed circuit board passes AOI inspection and goes on to packaging
or if it should be pulled.

The two kinds of mistake cost very different amounts:

- **False negative** (a genuinely defective board gets passed: the defect ships
  inside a finished product. Depending on what the board ends up in, that is a 
  field failure, a warranty claim, or a recall, plus the reputational cost of a
  customer finding the fault instead of the factory. The cost is at €120 per board.
- **False positive** (a good board gets pulled): it goes to a technician for a
  1-2 minute manual check, then back on the line. The cost of this is at
  €5 per board, and a few minutes of technician time, nothing more.

That would mean about a 24x ratio. Moreover, it is actionable as it states
that the decision threshold needs to prioritize recall over precision,
and a range of ambiguous cases can be considered for manual processing,
instead of having to make a yes/no decision. "What a person still has to
check" gives a good example of this. For this specific case, that range
is not very wide.

## Dataset

- **Workspace**: `yolo-wfx0p`
- **Project**: `pcb-defect-uqoat`
- **Version**: `2`
- **Link**: https://universe.roboflow.com/yolo-wfx0p/pcb-defect-uqoat/dataset/2

The first image is a close-up photo of a plain printed circuit board, with 
bounding boxes marked around manufacturing defects visible on the image. There 
are six defect types used in this dataset, based on the same taxonomic set as 
the PCB-defect datasets it stems from: `missing_hole`, `mouse_bite`, 
`open_circuit`, `short`, `spur`, and `spurious_copper`.

It fits the problem presented here more appropriately than just similar enough
because these classes of defects are precisely what would be detected by an AOI
camera (defects at the level of copper traces which can be recognized and 
pinpointed visually). Moreover, this dataset is the one which Roboflow uses in 
its public tutorial for defect inspection by engineers.

## What the dataset's composition implies

| Split | Images | Instances | missing_hole | mouse_bite | open_circuit | short | spur | spurious_copper | Mean box area (% of image) |
|---|---|---|---|---|---|---|---|---|---|
| Train | 1,389 | 5,954 | 911 | 883 | 880 | 1,471 | 883 | 926 | 0.11% |
| Val | 115 | 491 | 97 | 98 | 95 | **0** | 99 | 102 | 0.08% |
| Test | 115 | 474 | 96 | 99 | 93 | **0** | 94 | 92 | 0.07% |

Three things that this entails that we couldn't know until after training:

1. The `short` class cannot be scored at all on this split. The `short` 
   class is by far the most common in training (1,471 examples, more than 
   any other class), but there are zero examples of `short` in the
   validation and test sets. This isn't a thin-sample issue but rather a 
   complete absence, and even YOLOv5's own per-class evaluation metrics for the
   validation and test sets prove that: `short` is completely missing from the 
   per-class metrics table below. Whatever actual defect-detection skills this 
   model has in relation to the `short` class, they cannot be assessed using 
   this report, and anyone deploying this model will need to separately 
   validate `short` detection.
2. All objects are tiny. Average bounding-box area is about 0.07-0.11% of
   the total image area. This is consistent with the difference between mAP50 and
   mAP50-95 seen below: small objects are necessarily harder to detect
   accurately than large objects, even when they're detected at all.
3. The train/validation/test splits are consistent in composition relative
   to each other, but not relative to the real-life production line. Train/
   validation/test seem like consistent random splits of the same base image
   population (similar object sizes, similar distribution of classes relative
   to each other, aside from `short`, which is absent in all splits). But,
   as the next section proves empirically, all images in each split contain
   at least one defect, which means this data was created to show defects,
   not to simulate what an actual camera captures on a production line
   (most boards are good). This is crucial to the cost analysis in the next
   section; it means that while the test split can tell us how effectively
   this model detects a defect it's given, it cannot tell us how often the
   model will mistakenly flag a clean board, because there are none.

## Finetuning and baselines

Model: **`yolo11n`**, finetuned from COCO-pretrained weights for 60 epochs at
640px, batch size 16, on a Colab T4 GPU. Object detection, not classification, since the decision
genuinely needs to localize which part of the board is at fault, not just
whether the board as a whole is good or bad.

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **Finetuned yolo11n** | **0.934** | **0.875** | **0.922** | **0.458** |
| Pretrained COCO, zero-shot | undefined (0 relevant predictions) | 0.000 | n/a | n/a |
| Trivial baseline (always pass) | undefined | 0.000 | n/a | n/a |

The gap is exactly as large as expected, and in fact slightly larger: the
**pretrained COCO model never once predicted a class that overlaps our
vocabulary**, across all 115 test images, confirmed empirically rather than
assumed (`train.py` checks every prediction's class name against our six
classes and found zero matches). This mirrors the course notebook's playing-
card result precisely. The trivial baseline (always predict "no defect") trivially gets 0% recall by
construction, since every test image genuinely contains a defect. Both
baselines lose completely; the finetuned model is the only one that does
anything useful at all, which is itself informative about how far a
pretrained general detector is from this task without finetuning.

## Per-class evaluation and failure modes

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|
| missing_hole | 0.972 | 0.979 | **0.977** | 0.580 |
| mouse_bite | 0.901 | 0.859 | 0.901 | 0.429 |
| open_circuit | 0.929 | 0.871 | 0.932 | 0.438 |
| spur | 0.968 | **0.787** | **0.872** | 0.389 |
| spurious_copper | 0.901 | 0.880 | 0.929 | 0.450 |
| short | -- | -- | -- | -- (0 test instances, see above) |

Confusion matrix

**Which class carries the headline number:** missing_hole is both the easiest class
(AP50 0.977) and ties for the largest class in training, which is unsurprising, 
since a missing hole is a strong, unambiguous visual signal (something is absent
where copper should be) compared to a mouse-bite or a short, which require noticing
a subtle edge irregularity.

**Which class drags it down:** spur has the lowest recall (0.787) and the lowest
AP50 (0.872) of the five we can measure, despite having precision as high as any
other class (0.968). That combination -- high precision, lower recall, which means
that when the model does call something a spur, it's almost always right, but it
is missing roughly 1 in 5 real spurs. A real deployment should look specifically
at the images where spur was missed before trusting this model's headline mAP50 
of 0.922, since that overall number is carried mostly by missing_hole and would 
look noticeably worse if reported per class and weighted toward spur's performance.

**What the confusion matrix actually shows (results/confusion_matrix.png, generated by model.val()):**
the real cross-class confusion is between mouse_bite and open_circuit, not spur and spurious_copper
as the surface-level visual similarity might suggest - 6 true mouse_bite instances
get predicted as open_circuit, and 8 true open_circuit instances get predicted as
mouse_bite, a genuine mutual mix-up between those two specifically. spur and 
spurious_copper, by contrast, show zero cross-confusion with each other in the 
matrix; whatever recall the model loses on spur, it loses to background 
(missed entirely), not to mislabeling it as spurious_copper. A smaller, interesting
detail: the model predicts short three times for boxes that were actually spur 
(2) or open_circuit (1), which is evidence it did learn something about what a 
short looks like from the 1,471 training instances, even though its real recall 
on short is unverifiable on this split. The very large "background" 
counts elsewhere in the matrix (in the thousands) are an artifact of Ultralytics
generating this plot at a much lower internal confidence threshold than the 0.60
we actually operate at, and are consistent with the 4,080-of-4,544 false-candidate
rate already measured in detections_log.csv at low confidence.

**Which class counts as "positive" for the cost model below:** all six are treated as
positive. Any one of them, anywhere on the board, is a reason to pull the board. 
The board-level pass/review/fail decision only asks "is there a defect here at all,"
so it doesn't depend on which specific class was found, even though short's 
complete absence from the test set means we cannot verify the model's real-world 
behaviour on that specific failure mode.

## Cost matrix and operating point

**The cost matrix** (EUR per board):

| | Model says no defect | Model says defect |
|---|---|---|
| **Actually good** | €0 (TN) | €5 a technician's quick recheck (FP) |
| **Actually defective** | €120 ships, becomes a field failure/recall (FN) | €15 pulled and reworked (TP) |

**Base rate assumed:** 3% of the boards will be defective in production.
This is a premise that is not obtained from the dataset (the dataset
description section reveals that the test split was 100% defective, and it
cannot be used to obtain a defect rate figure since it does not represent
the base rate in production). 3% is a reasonable defect rate for an AOI-inspected
SMT line; lower in a well-matured and controlled process and higher when the
line is recently started or producing complex PCBs. Choice of threshold
depends on this figure: the higher the base rate, the greater the frequency
of missing defects and the better recall (low threshold) and when the base
rate is very low (<1%), false positive costs outweigh the costs per false
positive despite their relative low cost because of the large proportion
of good boards.

**A real complication:** because the test split has zero genuinely clean boards, the false-positive rate
cannot be measured directly at any threshold as it comes back undefined for
every single value in the sweep. Rather than silently drop the false-positive
side of the trade-off, `costing.py` substitutes a detection-level
false-alarm proxy: the share of boards where the model raises at least one
spurious (non-matching) detection at a given confidence, measured from
4,544 raw candidate detections logged during evaluation (464 true positives,
4,080 false positives, with a clean separation. True positives average
0.61 confidence, false positives average 0.013). This is an approximation,
not a measured production false-accept rate.

**The sweep and the chosen threshold:** sweeping the confidence threshold
from 0.01 to 0.99 and taking the minimum of the cost-per-1,000-boards curve
(using the proxy above) gives:

| Threshold | Recall | False-alarm proxy | Cost / 1,000 boards |
|---|---|---|---|
| 0.30 | 0.974 | 0.122 | €1,123 |
| 0.50 | 0.948 | 0.043 | €825 |
| **0.60** | **0.922** | **0.026** | **€823 (minimum)** |
| 0.70 | 0.765 | 0.026 | €1,316 |
| 0.80 | 0.104 | 0.000 | €3,271 |

**Chosen single threshold: 0.60**. recall 92.2%, false-alarm proxy 2.6%,
cost €823 per 1,000 boards, against €3,600 for doing nothing and far worse
for a threshold that drifts either direction. The curve is sharply peaked,
not flat: moving from 0.60 to 0.70 more than doubles the cost by losing too
much recall; moving to 0.30 pays a real false-alarm penalty for a recall gain
that barely moves the needle (97.4% vs 92.2%).

## What still has to be checked

Now, the dashboard implements three rules: if the confidence is
under `t_lo`, then pass automatically; if it is equal or over `t_hi`,
then reject automatically; otherwise give to a person. Running the whole `(t_lo, t_hi)`
grid with the cost function results in a finding that should definitely
be stated explicitly and not hidden somewhere: **the cost-optimal setting is
`t_lo = t_hi = 0.60` -- a zero width review band**. For this particular cost
setting human review never beats any threshold, since reviewing a bad board
(€20, review time + rework) is more expensive than simply rejecting it (€15),
while reviewing a good board (€5) is equally expensive as false positives
(€5). Reviewing only makes sense in the case of catching up with bad boards
that would be missed in any other way, and turning their cost from €120
to €20. But this particular model separates its scores sharply enough (true
positives have 0.61 mean score, false positives have 0.013 mean score), so
there is virtually no grey area left for a review band to catch up with.

However, we recommend a small non-zero width review band of `t_lo = 0.45`, `t_hi = 0.60`.
This would cost us €845 per 1,000 boards in-sample, which is around €22 (2.7%)
more than the cost-optimal setting. We do that on purpose, as compensation
for the following limitation of our work: the split we used for testing
was selected from the same distribution as for training, hence the
confidences are likely better calibrated than they would be on a new
camera/lighting/board revision setting in a factory.

At that operating point, **2.6% of boards get sent to review**. Assuming a
line producing **5,000 boards/day** (stated assumption) and **1.5 minutes**
per manual check:

- 5,000 x 2.6% = **132 reviews/day**
- 132 x 1.5 min = **198 reviewer-minutes/day**
- Against a 7.5-hour effective workday (450 minutes): **0.44 FTE**

That is realistic as it doesn't need a full-time hire, but is easily within the range of
what a single technician would be able to take in along with other regular tasks, or even
would be divided up between shifts. Were the proportion of review to increase substantially,
(say over 15-20% from a poorly calibrated model in the field), then that would change to
a true personnel question, precisely the sort that the sliders on the dashboard can enable
someone to evaluate before making a design commitment.

**Priced against the three alternatives**, all computed from the same test
split and cost assumptions:

| Strategy | Cost per 1,000 boards |
|---|---|
| Do nothing (pass everything) | €3,600 |
| Human checks every board | €5,450 |
| Model decides alone (single threshold, 0.60) | **€823** |
| Model + human review band (0.45, 0.60), this design | €845 |

The single-threshold approach alone represents the cheapest choice available, 
while the advantage of the review panel is limited to the argument of robustness
presented above and does not relate to savings on costs in this case. "Review
every board by a human" represents the worst choice made, even worse than 
inaction, because the cost of reviewing 1,000 boards is higher than the downside
of having 3% defective products, unless the base rate and cost of an error was 
much higher than what was assumed in this example.

## Limitations & next steps

- **`short` cannot be evaluated on this split at all.** No instances of `short`
  in either validation or test, despite its being the most numerous class in
  training. Next step: Either curate some `short` images by hand into the
  test set from the training pool (taking a certain degree of data leakage,
  obviously noted if done), or collect more `short`-labelled images and use
  them to make up val/test before trusting this model on that class.
- **The false-accept rate is a proxy for that quantity.** There are zero clean
  boards in the test split of this dataset, so the "false positive rate"
  quoted in the costing section is really a false alarm proxy at the
  detection level (i.e., spurious and non-matching detections) rather than
  clean boards that we know to have no defects. A real deployment would need
  a set of truly clean boards in production, photographed using this line's
  camera/lighting arrangement in order to get an accurate value for that rate
  and confirm the €823/1,000 estimate.
- **The cost figures are stated assumptions, not quotes from an actual
  factory.** They are argued in the costing section, but a real deployment
  would replace them with warranty/recall data and technician wages and run
  the entire sweep again with real numbers.
- **This test split is in-distribution with training** (same source, same
  imaging conditions). The recommendation above about review band is partly to
  guard against that. The true next step here is testing against photos
  taken from a camera/lighting configuration that this model has never seen.
- **`spur` is the weakest measurable class** (AP50 0.872, recall 0.787),
  visually confused with `spurious_copper`. More labelling of `spur` images,
  or a focused investigation of the images where this class is missed, would
  be the high value next step here.
  
## Generative AI Use Disclosure

I used AI for assistance in dataset choice, and error debugging (code wise)

