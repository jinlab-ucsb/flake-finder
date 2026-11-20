"""Fit the colour model (camera profile) of the glovebox BX53M + DP22 raw data to the flake colours of scanned chips,
without known thicknesses (the method of flakeid.calibrate, with what this camera needs).

    py -3.9 tools/bx53m/fit_colour_profile.py TILES [TILES ...] -o flakeid/profiles/NAME.json
                                              [--material hbn] [--oxide 285] [--bounds xc10|wide] [--maxiter 120]

TILES: <piece>_tiles folders of raw 10x scans (each chip gets its own fitted oxide).  Differences from flakeid.calibrate:
  - a white LED lamp, NA 0.25, no colour matrix (raw data);
  - the flat samples are picked with a noise-scaled test: the red channel is dim on ~290 nm oxide (6 % of full scale), so
    the fixed test of the analysis left out every strongly red flake (60 and 230 nm hBN) and the fit never saw them;
  - samples darker than the substrate in every channel (off the chip, bulk, dirt) are left out;
  - the profile gets "substrate_anchors" from the first chip (thickness.oxide_from_substrate: the oxide of every later
    chip from its substrate colour).
Takes ~7 minutes per run.  Check the result: two runs with --bounds xc10 and wide should give the same thicknesses.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FLAKEID_CAL_NA", "0.25")


def samples(src, flat):
    from flakeid.background import compute_contrast
    from flakeid.io import load_image
    img = load_image(src, flat=flat)
    cm = compute_contrast(img.rgb, img.pixel_um, max_value=img.max_value)
    c = cm.contrast
    cb, c2 = cv2.blur(c, (5, 5)), cv2.blur(c * c, (5, 5))
    sd = np.sqrt(np.maximum(c2 - cb * cb, 0))
    tol = np.maximum(0.02, 2.0 * np.asarray(cm.noise, float)[None, None, :] * np.sqrt(np.maximum(1.0 + cb, 0.05)))   # (shot noise)
    satd = cv2.dilate(cm.saturated.any(axis=2).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    m = cm.chip_mask & ~satd & np.all(sd < tol, axis=2) & (np.abs(cb).max(axis=2) > 0.05)
    m &= np.abs(cv2.blur(c, (15, 15)) - cb).max(axis=2) < 0.03 * (1 + np.abs(cb).max(axis=2))      # (not an edge or a slope)
    s = cb[m]
    s = s[~np.all(s < -0.3, axis=1)]
    clip = img.clip_value if img.clip_value is not None else img.max_value - 0.5
    return s, cm.substrate_rgb.astype(float), float(clip)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tiles", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--material", default="hbn")
    ap.add_argument("--oxide", type=float, default=285.0, help="the wafers' nominal oxide, nm")
    ap.add_argument("--flat", default=str(ROOT / "flats" / "flat_10x_dp22_raw.npz"))
    ap.add_argument("--bounds", choices=["xc10", "wide"], default="xc10", help="ranges of the camera's channels")
    ap.add_argument("--maxiter", type=int, default=120)
    a = ap.parse_args()
    from flakeid import calibrate as C
    from flakeid.thickness import save_profile
    C.PARAMS_LED = [("led_ratio", 0.2, 20.0), ("led_pc", 530.0, 610.0), ("led_pw", 90.0, 190.0)]
    ds = []
    for t in a.tiles:
        s, sub, clip = samples(t, a.flat)
        st = C.stratified(s, n=15000, edges=(0.0, 0.15, 0.4, 0.8, 1.5, 10.0))
        print(f"{t}: {len(s)} samples -> {len(st)}; substrate R/G {sub[0] / sub[1]:.4f}, B/G {sub[2] / sub[1]:.4f}", flush=True)
        ds.append((st, C.resolve_material(a.material), a.oxide, sub, clip))
    p, loss = C.fit(ds, maxiter=a.maxiter, tone="flare", ccm=False, led=True, wide=a.bounds == "wide", trim=0.9)
    ox = [round(a.oxide * p[f"oxide_{i}"], 1) for i in range(len(ds))]
    sub = ds[0][3]
    save_profile(a.output, C.camera_from_params(p), na=0.25, notes=f"BX53M + DP22 raw, LED, bounds {a.bounds}; tools/bx53m/fit_colour_profile.py",
                 extra={"materials": {}, "substrate_anchors": [[ox[0], ox[0], [round(sub[0] / sub[1], 4), round(sub[2] / sub[1], 4)]]],
                        "calibration": {"tiles": [str(t) for t in a.tiles], "fitted_oxide_nm": ox, "loss": loss,
                                        "params": {k: round(float(v), 4) for k, v in p.items()}}})
    print(json.dumps({k: round(float(v), 3) for k, v in p.items()}), "\noxide", ox, "\nwrote", a.output)


if __name__ == "__main__":
    main()
