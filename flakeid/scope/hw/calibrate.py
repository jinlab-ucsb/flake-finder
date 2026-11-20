"""Measure M, the camera pixel -> stage um matrix, of the objective in place (setting M: the 10x objective's).

    py -3.9 -m flakeid.scope.hw.calibrate [--step 40] [--settings scan_settings.json] [--write]

THIS MOVES THE STAGE: by --step um in +X and back, then in +Y and back, three times (it asks first).  Nothing else
moves; the focus is not touched.  Close Olympus Stream, put the 10x objective in, and put something with detail in focus under it
(flakes, tape residue, a scratch; not bare substrate).

How: a picture before and after every single step (one axis at a time, 0.6 s after the stage has stopped).  The shift
of the picture between the two (phase correlation) gives the matrix: with the scan program's convention (the stage
position that centres pixel p is stage + M (p - c)), a stage move d shifts the picture by -inverse(M) d.  Each step
is measured forwards and backwards and the six values per axis are averaged, so play in the stage and a chip that
creeps a little do not add up (comparing with one first picture did: 2026-11-13).

--step: a quarter of the field or less, so most of the picture stays in view (150 um suits 10x; 15 um for 100x).
--write: put M into the settings file (a copy of the old file is kept).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def grey(img):
    return np.asarray(img, np.float32)[..., :3].mean(axis=2) if np.ndim(img) == 3 else np.asarray(img, np.float32)


def shift_px(a, b):
    """(dx, dy) by which the picture's content moved from frame a to frame b (pixels), and the correlation's response."""
    import cv2
    a, b = grey(a), grey(b)
    a, b = a - cv2.GaussianBlur(a, (0, 0), 25), b - cv2.GaussianBlur(b, (0, 0), 25)      # (uneven illumination out)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return np.array([dx, dy]), float(resp)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", help="settings file (JSON)")
    ap.add_argument("--step", type=float, default=150.0, help="stage step in um (default 150)")
    ap.add_argument("--exposure", type=float, help="exposure in ms (default: as the camera is)")
    ap.add_argument("--write", action="store_true", help="write M into the settings file")
    ap.add_argument("--yes", action="store_true", help="do not ask before moving")
    a = ap.parse_args(argv)
    from ..settings import load
    from .bx53m import XY, make_core
    s = load(a.settings, hardware="bx53m")
    if a.write and not a.settings:
        ap.error("--write needs --settings")
    core = make_core(s)
    core.loadSystemConfiguration()
    try:
        x0, y0 = core.getXPosition(XY), core.getYPosition(XY)
        lim = core.stageLimitsUm(a.step + 10.0)
        print(f"stage at x {x0:.1f}, y {y0:.1f} um; step {a.step:g} um in +X, then in +Y, each time back to here")
        if not (lim[0] <= x0 <= lim[1] and lim[2] <= y0 <= lim[3]):
            print("too close to the end of the stage's travel for this step: move towards the middle")
            return 1
        if not a.yes and input("The stage will move.  Is it clear, in focus on something with detail?  Type yes: ").strip().lower() != "yes":
            print("cancelled: nothing moved")
            return 1
        if a.exposure:
            core.setExposure(a.exposure)

        def snap():
            time.sleep(0.6)
            core.snapImage()
            return np.array(core.getImage())

        def step(dx, dy):
            """One step: (the picture's shift in px, the correlation's response, how far the stage says it moved)."""
            before, p = snap(), (core.getXPosition(XY), core.getYPosition(XY))
            core.setXYPosition(XY, p[0] + dx, p[1] + dy)
            core.waitForDevice(XY)
            q = (core.getXPosition(XY), core.getYPosition(XY))
            d, r = shift_px(before, snap())
            return d, r, (q[0] - p[0], q[1] - p[1])

        f0 = snap()
        H, W = f0.shape[:2]
        sat = float((f0 >= 254).any(axis=2).mean()) if f0.ndim == 3 else 0.0
        per_um, resp = {"x": [], "y": []}, []
        for k in range(3):
            for ax, (dx, dy) in (("x", (a.step, 0.0)), ("y", (0.0, a.step))):
                for sign in (1.0, -1.0):
                    d, r, moved = step(sign * dx, sign * dy)
                    m = moved[0] if ax == "x" else moved[1]
                    print(f"  {ax} {m:+8.2f} um: picture shift {d[0]:+8.2f}, {d[1]:+8.2f} px (response {r:.2f})")
                    if abs(m) > 1e-6:
                        per_um[ax].append(d / m)
                    resp.append(r)
        sx, sy = np.mean(per_um["x"], axis=0), np.mean(per_um["y"], axis=0)       # px per um of stage move
        scat = max(float(np.linalg.norm(np.std(per_um[ax], axis=0))) for ax in "xy") * a.step
        f1 = snap()
        sb, rb = shift_px(f0, f1)
        print(f"back at the start: picture shift {sb[0]:+.2f}, {sb[1]:+.2f} px (response {rb:.2f}); scatter of the steps {scat:.1f} px")
        M = -np.linalg.inv(np.column_stack([sx, sy]))
        um = float(np.sqrt(abs(np.linalg.det(M))))
        print(f"\nM = [[{M[0, 0]:.4f}, {M[0, 1]:.4f}], [{M[1, 0]:.4f}, {M[1, 1]:.4f}]]   ({um:.4f} um per pixel; "
              f"field {um * W:.0f} x {um * H:.0f} um; {'mirrored' if np.linalg.det(M) < 0 else 'not mirrored'})")
        shift_max = max(float(np.abs(sx).max()), float(np.abs(sy).max())) * a.step
        warn = []
        if min(resp) < 0.15:
            warn.append("weak correlation: too little detail in the picture, or out of focus")
        if shift_max > 0.4 * min(W, H):
            warn.append("the picture moved by more than 40 % of the field: use a smaller --step")
        if shift_max < 20:
            warn.append("the picture moved by less than 20 px: use a larger --step")
        if scat > 0.05 * shift_max:
            warn.append("the steps disagree by more than 5 %: play in the stage, or the sample moves on it")
        if sat > 0.05:
            warn.append(f"{100 * sat:.0f} % of the picture is saturated: use a shorter --exposure")
        for w in warn:
            print("WARNING: " + w)
        if warn:
            print("M not written: fix the above and run it again" if a.write else "(do not trust this M: fix the above and run it again)")
            return 1
        Mr = [[round(float(v), 4) for v in row] for row in M]
        if a.write:
            p = Path(a.settings)
            d = json.loads(p.read_text()) if p.is_file() else {}
            if p.is_file():
                shutil.copy2(p, p.with_name(p.stem + time.strftime("_before_%Y%m%d_%H%M%S") + p.suffix))
            d["M"] = Mr
            d.setdefault("hardware", "bx53m")
            p.write_text(json.dumps(d, indent=1))
            print(f"M written to {p}")
        else:
            print(f'for the settings file:   "M": {json.dumps(Mr)}')
        return 0
    finally:
        core.unloadAllDevices()


if __name__ == "__main__":
    sys.exit(main())
