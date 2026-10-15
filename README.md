# EAS 510 - Assignment 1: Digital Forensics Apprentice

A rule-based expert system that matches modified images back to their originals.

This is the **starter repository** for Project 1. Two repositories matter:

| Repository | Role |
|------------|------|
| `delveccj/EAS510_Assignment1_starter` (this repo) | Your **code** lives here. You will fork it, work in your fork, and push your final submission here. |
| `delveccj/EAS510_Assignment1` | The **dataset** (read-only). Clone it for the images; never commit it to your fork. |

## Setup on your Codio box

```bash
# 1. Fork this repo on GitHub, then in your box:
git clone https://github.com/YOUR-USERNAME/EAS510_Assignment1_starter.git
cd EAS510_Assignment1_starter

# 2. Clone the read-only dataset as a sibling folder (images live there):
cd ~/workspace
git clone https://github.com/delveccj/EAS510_Assignment1.git

# 3. Install dependencies:
cd ~/workspace/EAS510_Assignment1_starter
python3 -m pip install -r requirements.txt

# 4. Point this box at YOUR fork (run once):
./setup_git.sh
```

## Repository layout

```
forensics_detective.py   SimpleDetector: register targets + find_best_match
rules.py                 Rule functions (Rule 1: Metadata, Rule 2: Histogram, Rule 3: Template)
test_system.py           Runs the detector over the data folders and writes results_*.txt
scripts/check_output_format.py   Validates a results file against the required format
setup_git.sh, submit.sh  One-time setup + commit/push helper ("backup button")
```

## The task in one paragraph

Implement three interpretable rules (metadata, color histogram, template matching)
that combine into a 0-100 confidence score. Run your system on `modified_images/`
and `random/` and save the full output as `results_v1.txt`. In Phase 2, run on
`hard/`, diagnose a systematic failure, add a **Rule 4**, and save
`results_v1_hard.txt` and `results_v2.txt`. Full instructions are in the Codio guide
and in the assignment PDF in UBLearns.

## Running

```bash
# Phase 1 (easy + random) -> results_v1.txt
python3 test_system.py --modified --random --output results_v1.txt

# Phase 1 on hard cases -> results_v1_hard.txt
python3 test_system.py --hard --output results_v1_hard.txt

# Phase 2 (everything) -> results_v2.txt
python3 test_system.py --modified --hard --random --output results_v2.txt

# Validate any results file against the required format:
python3 scripts/check_output_format.py results_v1.txt
```

The data folders are resolved from a sibling clone of `EAS510_Assignment1`.
If your data is elsewhere, pass `--data-dir /path/to/EAS510_Assignment1`.

## Output format (do not alter)

```
Processing: modified_image_01.jpg
Rule 1 (Metadata): FIRED - Size ratio 0.85 -> 20/30 points
Rule 2 (Histogram): FIRED - Correlation 0.92 -> 25/30 points
Rule 3 (Template): FIRED - Match score 0.76 -> 30/40 points
Final Score: 75/100 -> MATCH to original_03.jpg
```

## Backup doctrine

Your Codio box can be reset at any time. **Your fork on GitHub is the only safe
copy.** After every milestone run `./submit.sh "describe what you did"`. A box
restart never touches GitHub.

## License

Apache 2.0. See `LICENSE`.

## Observed weakness in V1: what failed and why

V1 got 60/60 easy, 56/60 hard, and rejected 15/15 random. On v1, v2 and v5 where the scale was constant, Rule 3 (template matching) fired 10/10. It went 0/10 on v3 (resize), 1/10 on v4 (rotation) and 0/10 on v6 (crop then resize). Three of the four failures were v3 resizes, all rejected at 47: original_01__resize_scale75__compress__q30__v3.jpg, original_02__resize_scale85__compress__q30__v3.jpg, and original_09__resize_scale75__compress__q30__v3.jpg. The fourth was original_09__crop_keep60__bright__compress__q40__v2.jpg at 39, the only brightness-related failure. v6 went through, but the lowest was 50, right on the edge. So the pattern is that template matching can't handle scale or rotation.

## Design decision for V2: what Rule 4 is and why you chose it

Rule 4: ORB keypoint matching with a RANSAC geometry check, counting inliers that agree on one shift, scale and rotation. That is consistent with the diagnosis, as keypoints are scale- and position-invariant, which is exactly what Rule 3 lacks. Rules 1 to 4 are weighted 20/20/30/30. I set the thresholds from measured data: random images scored 0 inliers, so Rule 4 fires at 10 inliers and gives full points at 20.

## Effect of the change: accuracy before/after on easy vs hard

Easy went from 60/60 to 59/60. Hard went from 56/60 to 60/60. Random stayed at 15/15. v3 went from 7/10 to 10/10. Margins increased as well. For v6, the lowest score increased from 50 to 63, and for the v2 brightness image the score increased from 39 to 59.

## Trade-offs: what new costs or risks did Rule 4 introduce

Rule 3 was reweighted from 40 points to 30 points, which hurt the crop_25 images. modified_09_crop_25pct still doesn't work, because a 25% crop covers so little of the frame that ORB finds few points inside it. Its 7 inliers are below what a decoy can reach. Because RANSAC is stochastic, one decoy got 4 inliers in one run and 9 in another, just shy of the 10 required to fire. That's why lowering the threshold to get 60/60 would have been a lucky bump, not a real fix. A full run also took about 3 3/4 minutes.
