"""EAS 510 - Project 1 - Phase 2: the V2 rule set.

Diagnosis from results_v1_hard.txt: Rule 3 (template matching) fired on 30/30
same-scale hard cases (v1, v2, v5) but only 1/30 cases with resize or rotation
(v3, v4, v6). Those survived on Rules 1 and 2 alone, with scores of 47-59.

Rule 4 adds ORB keypoint matching, which is invariant to scale and rotation,
plus a RANSAC geometry check so coincidental matches do not count.

Rules 1-3 run unchanged from rules.py. Their points are rescaled here so the
four rules still sum to 100: Rules 1 and 2 lose the most because they proved
weakest (Rule 1 gave a random decoy 30 points; Rule 2 breaks under brightness
and contrast changes).
"""

from functools import lru_cache

import cv2
import numpy as np

import rules

WEIGHTS = {1: 20, 2: 20, 3: 30, 4: 30}

# New names on purpose: the detector searches rules.py first, so reusing the
# old names would silently run the 30/30/40 versions instead of these.
RULES = ("rule1_metadata_v2", "rule2_histogram_v2",
         "rule3_template_v2", "rule4_keypoints")


def _reweight(ev, out_of):
    """Same evidence, smaller budget: scale the points to the V2 weight."""
    ev = dict(ev)
    ev["score"] = int(round(ev["score"] * out_of / ev["out_of"]))
    ev["out_of"] = out_of
    return ev


def rule1_metadata_v2(target, input_path):
    return _reweight(rules.rule1_metadata(target, input_path), WEIGHTS[1])


def rule2_histogram_v2(target, input_path):
    return _reweight(rules.rule2_histogram(target, input_path), WEIGHTS[2])


def rule3_template_v2(target, input_path):
    return _reweight(rules.rule3_template(target, input_path), WEIGHTS[3])


_ORB = cv2.ORB_create(nfeatures=1000)
_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING)
ORB_MAX_DIM = 1024       # shrink the original to this size; suspect gets the same factor
RATIO_TEST = 0.75        # keep a pair only if clearly better than the runner-up
INLIERS_FOR_FULL = 20    # decoys score 0 inliers, so 20 agreeing points is overwhelming
FIRE_AT = 0.5            # fires at 10 inliers, well clear of the decoys' 0


@lru_cache(maxsize=2048)
def _orb(path, scale):
    gray = rules._gray_full(path)
    if gray is None:
        return None, None
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return _ORB.detectAndCompute(gray, None)


def rule4_keypoints(target, input_path):
    """ORB keypoints + RANSAC: survives the resize and rotation that defeat Rule 3."""
    out = {"rule": 4, "name": "Keypoints", "fired": False, "score": 0,
           "out_of": WEIGHTS[4], "note": "Keypoint score 0.00", "metric": 0.0}
    try:
        src = rules._gray_full(target["path"])
        if src is None:
            return out
        scale = min(1.0, ORB_MAX_DIM / max(src.shape))
        kp_s, des_s = _orb(target["path"], scale)
        kp_i, des_i = _orb(input_path, scale)
        if des_s is None or des_i is None or len(kp_s) < 2 or len(kp_i) < 2:
            return out

        pairs = _MATCHER.knnMatch(des_i, des_s, k=2)
        good = [p[0] for p in pairs
                if len(p) == 2 and p[0].distance < RATIO_TEST * p[1].distance]

        inliers = 0
        if len(good) >= 4:
            pts_i = np.float32([kp_i[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pts_s = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(pts_i, pts_s, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = int(mask.sum())

        metric = min(1.0, inliers / INLIERS_FOR_FULL)
        out["metric"] = round(metric, 3)
        out["note"] = f"Keypoint score {out['metric']:.2f}"
        if out["metric"] >= FIRE_AT:
            out["fired"] = True
            out["score"] = int(round(out["out_of"] * out["metric"]))
    except Exception:
        pass
    return out
