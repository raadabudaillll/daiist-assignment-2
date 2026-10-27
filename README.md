# Assignment 2

Pick a decision somebody currently makes by looking at an image. Find a dataset
on Roboflow Universe that serves it, finetune a YOLO model to make that
decision, then put a cost on being wrong, choose the confidence threshold that
costs least, and decide which cases a person still has to check.

You are not collecting or labelling data. Getting a model to train is three
lines. The hard part is choosing a threshold you would ship and defending it
with numbers.

Pick a problem where a false positive and a false negative cost different
amounts. If they cost the same, half this assignment disappears.

You will be asked to explain every decision later, without your code in front of
you, in the written comprehension check. See Grading.

Setup, your Roboflow API key, and how to check your work are in
[SETUP.md](SETUP.md). Do that first.

## The task

1. **Frame the business problem**, before looking at any dataset. What decision
   does the model support, who acts on it, and what does each kind of mistake
   cost? Write it in REPORT.md. If you cannot say why one kind of error is worse
   than the other, keep looking.
2. **Find a dataset that serves it** on
   [Roboflow Universe](https://universe.roboflow.com). Object detection needs at
   least 2 classes, since the background is already a free negative class.
   Image classification needs at least 3. Use a public dataset as it is. Do not
   fork it: forking makes it private, so nobody else can download what you used.
   Record the workspace, project and version in REPORT.md.
3. **Describe the dataset before training on it.** Instances per class,
   train/val/test sizes, and object size relative to the image. Take it from the
   dataset's health stats on Universe or compute it from the labels. Then say
   what it implies. A class with 40 instances gives you a number you cannot
   trust. A test split that looks nothing like production gives you a number
   that will not hold.
4. **Finetune** `yolo11n` or `yolo11s`, or the `-cls` variant for
   classification. Compare it on the same test split against two baselines:
   the pretrained COCO model used zero-shot, and a trivial baseline that
   predicts the majority class or predicts nothing. A model that loses to the
   trivial baseline is a finding, as long as you report it. Commit your best
   weights to `weights/`.
5. **Report per-class results.** Precision and recall per class, and the
   confusion matrix, alongside the overall number (AP50 and mAP50-95 for
   detection, accuracy for classification). State which class you are treating
   as positive. Say which class carries your headline number and where the model
   fails. An average hides the thing you need to know.
6. **Cost the confusion matrix and pick a threshold.** Assign a cost in euros to
   false positives and to false negatives, and to true positives and true
   negatives if they carry one. State the base rate you assume, meaning how
   often the positive class actually occurs. Sweep the confidence threshold,
   plot cost against threshold, and take the minimum. Justify every figure. "I
   assumed X because Y, and the answer changes if X drops below Z" beats a
   precise number from nowhere.
7. **Decide which cases go to a person.** Three outcomes, not two: pass, fail,
   or send for human review. Work out how many cases a day that sends to a
   person, how long one takes to check, and how many people that needs. Say
   whether that is realistic. Price it against doing nothing, having a person
   check everything, and letting the model decide alone.
8. **Write REPORT.md**, **build the dashboard**, and **submit**.

Steps 6 and 7 are the deliverable. The model is the easy part.

### Dashboard requirements

A Gradio dashboard, built from your trained model, that never trains anything at
startup. It must let you:

- Upload an image and see predictions at the current threshold.
- Move the two thresholds separating pass, human review and fail, and watch the
  confusion matrix, the expected cost per 1,000 cases, and the share sent for
  review all update.
- See the cost against threshold curve, with your chosen threshold marked.
- See per-class results, not just the overall number.

## How you organise your code

```bash
uv run python main.py train
uv run python main.py app
```

**Scripts or notebooks, your choice.** For each stage, `main.py` looks for
`<stage>.py` first, then `<stage>.ipynb`. Name your training code `train.py` or
`train.ipynb`, and your app `app.py` or `app.ipynb`, at the repo root. Keep
exactly one of each pair and delete the other, because `main.py` refuses to
guess. You can use a notebook for one stage and a script for the other.

That naming is the only constraint. How you build them is yours, and each must
run start to finish with no manual steps.

### Where training happens

**Train wherever you like.** Your own machine, a GPU you have access to, Google
Colab, anything. SETUP.md explains the Colab route if you do not have a GPU, but
nothing requires it. `main.py train` is your real pipeline and takes as long as
it takes.

It saves the model it trains into `weights/`. Commit that file: it is what your
app loads and what anyone grading you runs against. Record in REPORT.md where
you trained and what settings you used.

Nothing re-trains your model to check it. `check.py` loads your committed
weights and launches your app, which is what actually has to work for someone
else. See SETUP.md.

## Submission

1. Push your finished branch to your fork.
2. Open a pull request from your fork to the original repo:
   https://github.com/ami232/daiist-assignment-2
3. Submit the PR link on Blackboard.
4. Upload a zipped copy of your repo to Blackboard as a backup, in case your
   fork or the PR becomes unavailable.

REPORT.md must name the exact Roboflow workspace, project and version you used,
so your pipeline can be run against the data it was built on.

## Grading

**Hard requirement:** your training pipeline and your app must each run end to
end via `main.py` with no manual intervention. A submission that fails this
cannot pass, regardless of everything below.

| Component | Weight | What it checks |
|---|---|---|
| Business framing and dataset choice | 20% | Is the decision real, do false positives and false negatives genuinely cost different amounts, and does the framing drive later choices rather than just getting narrated? |
| Dataset description | 15% | Class balance and split quality interpreted, with consequences drawn, not a screenshot pasted in. |
| Finetuning and evaluation | 25% | Does it beat both baselines, is it reported per class, and are the failures named rather than hidden behind an average? |
| Costing and human review | 25% | Are the costs defended, is the threshold chosen by sweeping rather than by taste, and does the review load match the people available? |
| Dashboard | 15% | All required views present, working off your pipeline's real outputs, nothing retrained at startup. |
| **× Written Comprehension Check** | **0–100%** | Multiplies the subtotal |

REPORT.md is not a row of its own. Its content lives in the other five, so its
quality is already captured there. The five rows sum to 100% and form your
subtotal. The Written Comprehension Check then multiplies that subtotal, so a
strong submission with a weak check score is scaled down.

## Generative AI use

Per the syllabus AI Policy, disclosed AI use is fine and must be stated in
REPORT.md's disclosure section. For this assignment specifically:

- **Fine for**: boilerplate and common operations, such as downloading a
  dataset, wiring up the training call, and plotting results.
- **Your own judgement for**: the decisions this assignment grades. Which
  problem to frame, what each kind of error costs, where the thresholds go, how
  much goes to a person, and whether the model is good enough to ship. AI can
  write the code for a decision. The decision has to be yours.
- **REPORT.md**: the ideas and findings must be your own. AI may help with
  formatting, not with the analysis or conclusions.

You have to be able to explain every decision in your submission as if you made
it yourself, because you did. Using a tool to write it does not transfer the
understanding requirement to the tool.