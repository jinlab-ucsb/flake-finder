"""Real flake or tape glue?  The 100x pictures of a piece sorted, so that only real flakes are kept.

At 10x a patch of tape glue (residue) can look like a thin flake.  At 100x the two differ plainly: a flake has crisp
edges and a flat inside, whatever its thickness; glue is a soft-edged, mottled patch, often with droplets.  This module
measures that on each raw 100x picture (linear mosaic, as flakeid.hundred reads it) and gives the probability that the
picture shows a real flake (profiles/real100.json, a logistic model fitted to pictures labelled by eye with
tools/fit_real100.py).

Measured in the central 70 % of the frame (a flake only at the frame's rim is not the target), on the three channels'
log ratio to the substrate:
    gf_top     how steep the steepest edges are (the strongest 0.5 %)
    sharp      at the strongest large-scale edges: fine-scale gradient / coarse-scale gradient (a step: high; a soft
               edge: about 1), whatever the contrast
    edge       how strong those large-scale edges are
    mottle     local variation inside the material (median over it)
    flat_frac  the share of the material that is locally flat
    cover      the share of the centre that is not substrate
The substrate colour is the run's (the median of the frames' commonest colours), so that a flake filling the whole
frame is not taken for substrate: a frame covered by flat material is a real (large) flake without any edge in view.

    python -m flakeid.real100 <piece>_100x [--sort] [--cut 0.5]

writes <piece>_100x/real100.csv (picture, p_real, verdict, the measures); --sort moves the pictures judged glue (the raw
TIFF, its colour copy and shifted / mosaic frames) into <piece>_100x/rejected/.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROFILE = Path(__file__).parent / "profiles" / "real100.json"
FEATURES = ("gf_top", "sharp", "edge", "mottle", "flat_frac", "cover")
CENTRE = 0.70                      # the central part of the frame that is judged
FULL_COVER, FULL_FLAT = 0.85, 0.70  # a frame covered by flat material: a large flake, no edge needed (real ones: flat 0.78-0.98; glue films over the whole frame: 0.49-0.59)


def channels(raw, dark=128.0):
    """(r, g, b) half-resolution channels of a raw mosaic (red at the top-left), dark removed; or of an RGB picture."""
    a = np.asarray(raw, np.float32)
    if a.ndim == 3:
        return a[::2, ::2, 0], a[::2, ::2, 1], a[::2, ::2, 2]
    a = a - float(dark)
    return a[0::2, 0::2], 0.5 * (a[0::2, 1::2] + a[1::2, 0::2]), a[1::2, 1::2]


def frame_mode(raw, dark=128.0):
    """The commonest colour of a frame (r, g, b)."""
    out = []
    for c in channels(raw, dark):
        h = np.bincount(np.clip(c[::2, ::2], 0, 65535).astype(np.int64).ravel())
        h = np.convolve(h, np.ones(9), "same")
        out.append(float(np.argmax(h)))
    return np.array(out)


def features(raw, substrate, dark=128.0):
    r, g, b = channels(raw, dark)
    s = np.maximum(np.asarray(substrate, np.float32), 1.0)
    L = np.log(np.clip(np.stack([r / s[0], g / s[1], b / s[2]], 2), 0.05, 20.0)).astype(np.float32)
    H, W = L.shape[:2]
    m0 = (1.0 - CENTRE) / 2.0
    y0, y1, x0, x1 = int(m0 * H), int((1 - m0) * H), int(m0 * W), int((1 - m0) * W)

    def grad(sig):
        q = cv2.GaussianBlur(L, (0, 0), sig)
        gx, gy = cv2.Sobel(q, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(q, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        return np.sqrt((gx * gx + gy * gy).sum(axis=2))[y0:y1, x0:x1]
    gf, gc = grad(1.5), grad(8.0)
    k = max(50, gc.size // 200)
    idx = np.argpartition(gc.ravel(), gc.size - k)[-k:]
    out = dict(gf_top=float(np.partition(gf.ravel(), gf.size - k)[-k:].mean()),
               sharp=float(np.median(gf.ravel()[idx] / np.maximum(gc.ravel()[idx], 1e-6))),
               edge=float(np.median(gc.ravel()[idx])))
    mat = np.abs(L[y0:y1, x0:x1]).max(axis=2) > 0.06
    out["cover"] = float(mat.mean())
    q = cv2.GaussianBlur(L, (0, 0), 1.5)[y0:y1, x0:x1]
    mu = cv2.blur(q, (15, 15))
    sd = np.sqrt(np.maximum(cv2.blur(q * q, (15, 15)) - mu * mu, 0)).sum(axis=2)
    inner = cv2.erode(mat.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    if inner.sum() > 200:
        out["mottle"], out["flat_frac"] = float(np.median(sd[inner])), float((sd[inner] < 0.02).mean())
    else:
        out["mottle"], out["flat_frac"] = 0.02, 0.5
    return out


def vector(f):
    return [math.log(max(f["gf_top"], 1e-4)), f["sharp"], math.log(max(f["edge"], 1e-4)), math.log(max(f["mottle"], 1e-4)),
            f["flat_frac"], f["cover"]]


def p_real(f, model=None):
    """Probability that the picture shows a real flake."""
    if f["cover"] >= FULL_COVER and f["flat_frac"] >= FULL_FLAT:
        return 0.99                                   # (flat material all over the frame: a large flake)
    m = model or load_model()
    x = (np.array(vector(f)) - np.array(m["mean"])) / np.array(m["std"])
    return float(1.0 / (1.0 + math.exp(-(float(np.dot(x, m["coef"])) + m["intercept"]))))


_MODEL = None


def load_model(path=None):
    global _MODEL
    if path is None and _MODEL is not None:
        return _MODEL
    m = json.loads(Path(path or PROFILE).read_text())
    if path is None:
        _MODEL = m
    return m


def _env(name, default):
    import os
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _areas(folder):
    """{target label: the flake's area at 10x (um2)} from <piece>_targets.csv beside the 100x folder."""
    f = Path(folder).parent / (Path(folder).name[:-len("_100x")] + "_targets.csv")
    try:
        return {r["label"]: float(r["area_um2"]) for r in csv.DictReader(open(f, encoding="utf-8")) if r.get("area_um2")}
    except (OSError, ValueError, KeyError):
        return {}


def judge(folder, dark=128.0, cut=None, log=print, min_flat=None, min_area=None):
    """Every picture of <piece>_100x judged: [dict(file, order, label, p_real, real, why, features...)], written to
    real100.csv.  How picky (arguments, else the environment the scan program sets from its settings):
        cut       FLAKEID_REAL100_CUT   least probability of a real flake (default the model's, 0.5; lower keeps doubtful ones)
        min_flat  FLAKEID_MIN_FLAT      least share of the flake that is flat and clean, 0..1 (0 = any; speckled, dirty or
                                        stepped flakes have 0.2-0.4, clean ones 0.75 or more)
        min_area  FLAKEID_MIN_AREA_UM2  least area of the flake at 10x, um2 (0 = any)
    real = 1 only when all three pass; why says which did not ("glue", "dirty", "small")."""
    folder = Path(folder)
    idx = next(folder.glob("*_100x_index.csv"))
    rows = list(csv.DictReader(open(idx)))
    m = load_model()
    cut = _env("FLAKEID_REAL100_CUT", float(m.get("cut", 0.5))) if cut is None else float(cut)
    min_flat = _env("FLAKEID_MIN_FLAT", 0.0) if min_flat is None else float(min_flat)
    min_area = _env("FLAKEID_MIN_AREA_UM2", 0.0) if min_area is None else float(min_area)
    areas = _areas(folder)
    frames = {}
    for r in rows:
        f = folder / r["file"]
        if not f.is_file():
            f = folder / "rejected" / r["file"]
        a = cv2.imread(str(f), cv2.IMREAD_UNCHANGED) if f.is_file() else None
        if a is not None and a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        frames[r["file"]] = a
    modes = np.array([frame_mode(a, dark) for a in frames.values() if a is not None])
    if not len(modes):
        return []
    sub = np.median(modes, axis=0)                    # the run's substrate: most frames are mostly substrate
    out = []
    for r in rows:
        a = frames[r["file"]]
        if a is None:
            continue
        f = features(a, sub, dark)
        p = p_real(f, m)
        area = areas.get(r["label"])
        why = [w for w, bad in (("glue", p < cut), ("dirty", f["flat_frac"] < min_flat), ("small", area is not None and area < min_area)) if bad]
        out.append(dict(file=r["file"], order=r["order"], label=r["label"], p_real=round(p, 3), real=int(not why), why=";".join(why),
                        area_um2="" if area is None else round(area), stage_x_um=r.get("stage_x_um", ""), stage_y_um=r.get("stage_y_um", ""),
                        z_um=r.get("z_um", ""), **{k: round(v, 4) for k, v in f.items()}))
    with open(folder / "real100.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    log(f"{folder.name}: {sum(o['real'] for o in out)} of {len(out)} 100x pictures kept (real flake: probability >= {cut:g}; flat share >= "
        f"{min_flat:g}; area >= {min_area:g} um2); left out: " + ", ".join(f"{sum(w in o['why'].split(';') for o in out)} {w}" for w in ("glue", "dirty", "small"))
        + "; real100.csv")
    return out


def collect(folder, verdicts, dest, log=print):
    """The kept flakes' colour pictures as PNG in dest (one folder to look through: <label>_x<stage x>_y<stage y>.png)
    with flakes.csv (picture, stage position, focus, area, the measures).  Pictures there from an earlier sorting are
    replaced."""
    folder, dest = Path(folder), Path(dest)
    if dest.is_dir():
        for f in dest.glob("*.png"):
            f.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in verdicts:
        if not v["real"]:
            continue
        src = next((f for d in (folder, folder / "rejected") for f in (d / (v["file"][:-4] + s) for s in ("_colour.png", "_colour.tif")) if f.is_file()), None)
        if src is None:
            continue
        a = cv2.imread(str(src), cv2.IMREAD_COLOR)
        try:
            name = f"{v['label']}_x{float(v['stage_x_um']):.0f}_y{float(v['stage_y_um']):.0f}.png"
        except ValueError:
            name = v["label"] + ".png"
        cv2.imwrite(str(dest / name), a)
        rows.append(dict(picture=name, **{k: v[k] for k in ("label", "stage_x_um", "stage_y_um", "z_um", "area_um2", "p_real", "flat_frac", "mottle")}))
    if rows:
        with open(dest / "flakes.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    log(f"{folder.name}: {len(rows)} real flakes' pictures in {dest}")


def sort(folder, verdicts, log=print):
    """The pictures judged glue moved into <folder>/rejected/ (the raw TIFF, its colour copy, shifted and mosaic frames)."""
    folder = Path(folder)
    rej = folder / "rejected"
    n = 0
    for v in verdicts:
        if v["real"]:
            continue
        stem = v["file"][:-4]
        for f in list(folder.glob(stem + "*")):
            if f.is_file():
                rej.mkdir(exist_ok=True)
                shutil.move(str(f), str(rej / f.name))
                n += 1
    log(f"{folder.name}: {n} files of {sum(1 - v['real'] for v in verdicts)} glue pictures moved to {rej}")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="+")
    ap.add_argument("--sort", action="store_true", help="move the glue pictures into rejected/")
    ap.add_argument("--cut", type=float, help="least probability of a real flake (default 0.5)")
    ap.add_argument("--min-flat", type=float, help="least flat, clean share of the flake, 0..1")
    ap.add_argument("--min-area", type=float, help="least area of the flake, um2")
    ap.add_argument("--collect", metavar="FOLDER", help="the kept flakes' colour pictures as PNG into this folder, with flakes.csv")
    a = ap.parse_args(argv)
    for f in a.folder:
        v = judge(f, cut=a.cut, min_flat=a.min_flat, min_area=a.min_area)
        if a.collect:
            collect(f, v, a.collect)
        if a.sort:
            sort(f, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
