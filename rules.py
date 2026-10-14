"""EAS 510 - Project 1: Rule functions for the Digital Forensics Apprentice.

Each rule compares one suspect image against one registered original and
returns a dict:

    {
        "rule": 1,
        "name": "Metadata",
        "fired": bool,          # True if the rule found supporting evidence
        "score": int,           # points awarded out of "out_of"
        "out_of": int,
        "note": "Size ratio 0.85",   # short human-readable metric
        "metric": 0.85,              # raw similarity measure in [0, 1]
    }

The points budgets are fixed by the assignment: Rule 1 = 30, Rule 2 = 30,
Rule 3 = 40. The detector sums them into a 0-100 confidence score.

These starter implementations are deliberately WEAK baselines (they hinge on
simple thresholds). Improving them is the assignment.

Reads are cached per path so one suspect against many originals does not
re-decode images needlessly.
"""

import os
from functools import lru_cache

import numpy as np
import cv2
from PIL import Image

#: Rule names used by SimpleDetector.evaluate() (V1).
RULES = ("rule1_metadata", "rule2_histogram", "rule3_template")


@lru_cache(maxsize=256)
def _arr(path):
    return cv2.imread(path)


@lru_cache(maxsize=256)
def _hist(path):
    img = _arr(path)
    if img is None:
        return None
    hists = [cv2.calcHist([img], [i], None, [32], [0, 256]) for i in range(3)]
    hist_all = np.concatenate(hists).ravel().astype(np.float32)
    cv2.normalize(hist_all, hist_all)
    return hist_all


def _downscale(img, max_dim=512):
    """Fit an image into `max_dim` before expensive cv2 work."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


@lru_cache(maxsize=256)
def _gray_full(path):
    img = _arr(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


@lru_cache(maxsize=256)
def _size(path):
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def rule1_metadata(target, input_path):
    """File-level consistency: could this file have been made from this original?

    Evidence, from measuring the dataset:
    - A crop or re-save is never larger than its source.
    - Same pixel size means a re-save (compression, brightness, format change).
    - Smaller with the same aspect ratio means a proportional crop. A JPEG crop
      keeps roughly the same bytes per pixel as its source (0.26-0.30 against
      0.23 for original_00), while the random decoys sit far lower
      (0.02-0.04 B/px against 0.075-0.23 for the originals).
    """
    out = {"rule": 1, "name": "Metadata", "fired": False, "score": 0,
           "out_of": 30, "note": "Size ratio 0.00", "metric": 0.0}
    try:
        src_w, src_h = _size(target["path"]) or (0, 0)
        in_w, in_h = _size(input_path) or (0, 0)
        if not (src_w and src_h and in_w and in_h):
            return out
        if in_w > src_w or in_h > src_h:
            return out  # larger than the original: cannot be derived from it

        if (in_w, in_h) == (src_w, src_h):
            metric = 1.0  # same frame, re-saved
        else:
            aspect_diff = abs((in_w / in_h) / (src_w / src_h) - 1)
            src_bpp = os.path.getsize(target["path"]) / (src_w * src_h)
            in_bpp = os.path.getsize(input_path) / (in_w * in_h)
            bpp_ratio = in_bpp / src_bpp
            if aspect_diff > 0.02 or not (0.5 <= bpp_ratio <= 2.0):
                metric = 0.0
            else:
                metric = min(bpp_ratio, 1 / bpp_ratio)

        out["metric"] = round(metric, 3)
        out["note"] = f"Size ratio {out['metric']:.2f}"
        if out["metric"] >= 0.5:
            out["fired"] = True
            out["score"] = int(round(out["out_of"] * out["metric"]))
    except Exception:
        pass
    return out


def rule2_histogram(target, input_path):
    """Color histogram correlation. Robust to crops and spatial changes."""
    out = {"rule": 2, "name": "Histogram", "fired": False, "score": 0,
           "out_of": 30, "note": "Correlation 0.00", "metric": 0.0}
    try:
        hs, hi = _hist(target["path"]), _hist(input_path)
        if hs is None or hi is None:
            return out
        corr = float(cv2.compareHist(hs, hi, cv2.HISTCMP_CORREL))
        out["metric"] = round(max(0.0, min(1.0, corr)), 3)
        out["note"] = f"Correlation {out['metric']:.2f}"
        if out["metric"] >= 0.5:
            out["fired"] = True
            out["score"] = int(round(out["out_of"] * out["metric"]))
    except Exception:
        pass
    return out


def rule3_template(target, input_path):
    """Template matching. Detects when the suspect is contained in the original.

    cv2.matchTemplate needs the template (suspect) smaller than the target
    (original); on rotated/resized suspects this rule degrades fast. Fixing
    that becomes Rule 4 territory.
    """
    out = {"rule": 3, "name": "Template", "fired": False, "score": 0,
           "out_of": 40, "note": "Match score 0.00", "metric": 0.0}
    try:
        src_full = _gray_full(target["path"])
        nd_full = _gray_full(input_path)
        if src_full is None or nd_full is None:
            return out
        # A suspect larger than the original cannot be a crop of it.
        if (nd_full.shape[0] > src_full.shape[0]) or (nd_full.shape[1] > src_full.shape[1]):
            return out
        # One scale factor, taken from the original, applied to both images,
        # so a crop keeps its true size relative to the original.
        scale = min(1.0, 512 / max(src_full.shape))
        src_g = cv2.resize(src_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        nd_g = cv2.resize(nd_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(src_g, nd_g, cv2.TM_CCOEFF_NORMED)
        metric = float(res.max())
        out["metric"] = round(max(0.0, min(1.0, metric)), 3)
        out["note"] = f"Match score {out['metric']:.2f}"
        if out["metric"] >= 0.4:
            out["fired"] = True
            out["score"] = int(round(out["out_of"] * out["metric"]))
    except Exception:
        pass
    return out