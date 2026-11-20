"""Measure the 100x objective against the 10x one on the glovebox microscope: the XY offset (setting offset_100x_um),
the pixel ratio (ratio_100x) and the focus offset (focus_offset_um).

    py -3.9 -m flakeid.scope.hw.offset100 [--settings scan_settings.json] [--at X Y Z] [--write]

THIS MOVES THE FOCUS AND TURNS THE NOSEPIECE (it asks first).  Start with the 10x objective in, in focus, on a place
with small sharp details near the centre of the picture (a few small flakes: edges in X and Y), or give such a place
with --at (stage um, 10x focus).  What it does:

  1. a 10x picture there
  2. the focus lowered to 20 um below where the 100x focus is expected (10x focus + focus_offset_um), then the
     nosepiece turned to 100x: the stage is never nearer the 100x objective than its expected focus while it turns
  3. a focus sweep upwards through the expected focus (within the focus calibration's safety limit), the sharpest Z
  4. a 100x picture; it is shrunk by the ratio and found in the 10x picture: where the 100x objective looks, in 10x
     pixels, gives the offset
  5. the focus lowered again, the nosepiece back to 10x, the 10x focus as it was

offset_100x_um is the stage move that puts under the 100x objective what was centred at 10x.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def grey(a):
    return np.asarray(a, np.float32)[..., :3].mean(axis=2)


def sharpness(a):
    import cv2
    # (a quarter of the resolution: at 100x a camera pixel is far finer than the optics resolve, and pixel noise would
    # outweigh the edges)
    g = cv2.resize(grey(a), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return float(np.sqrt(gx * gx + gy * gy).mean() / max(float(g.mean()), 1.0))


def locate(a10, a100, ratio0):
    """Where the 100x picture is in the 10x one: (centre x, y in 10x px, ratio, correlation), the best of several ratios."""
    import cv2
    A = grey(a10)
    A = A - cv2.GaussianBlur(A, (0, 0), 40)
    best = None
    for ratio in ratio0 * np.linspace(0.94, 1.06, 13):
        B = cv2.resize(grey(a100), None, fx=1.0 / ratio, fy=1.0 / ratio, interpolation=cv2.INTER_AREA)
        B = B - float(B.mean())
        r = cv2.matchTemplate(A, B.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        _, v, _, (x, y) = cv2.minMaxLoc(r)
        if best is None or v > best[3]:
            best = (x + B.shape[1] / 2.0, y + B.shape[0] / 2.0, float(ratio), float(v))
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings")
    ap.add_argument("--at", nargs=3, type=float, metavar=("X", "Y", "Z"), help="go there first (stage um, the 10x focus)")
    ap.add_argument("--z100", type=float, help="the 100x focus here, found by eye (hw.zcal's window): no focus sweep")
    ap.add_argument("--out", default=".", help="folder for the two pictures")
    ap.add_argument("--write", action="store_true", help="write the results into the settings file")
    ap.add_argument("--yes", action="store_true", help="do not ask before moving")
    a = ap.parse_args(argv)
    from PIL import Image
    from ..settings import load
    from .bx53m import XY, Z, make_core
    from .cbfm import CBFMNosepiece
    s = load(a.settings, hardware="bx53m")
    if a.write and not a.settings:
        ap.error("--write needs --settings")
    pos = s.objective_positions
    core = make_core(s)
    core.loadSystemConfiguration()
    nose = CBFMNosepiece(s.nosepiece_port)
    core.nosepiece = nose
    M = np.array(s.M, float)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def expose(target=0.486):
        for _ in range(8):
            core.snapImage()
            lv = float(np.median(np.asarray(core.getImage())[..., 1])) / 255.0
            if abs(lv / target - 1) < 0.08:
                break
            core.setExposure(min(500.0, max(0.5, core.getExposure() * min(max(target / max(lv, 0.01), 0.3), 3.0))))
        return core.getExposure()

    def snap():
        time.sleep(0.3)
        core.snapImage()
        return np.array(core.getImage())

    def set_z(z):
        core.setPosition(Z, z)
        core.waitForDevice(Z)

    try:
        if nose.position() != pos["10x"]:
            print("put in the 10x objective first (the program's live view, or Olympus Stream)")
            return 1
        if not a.yes and input("The focus will move and the nosepiece will turn to 100x and back.  Type yes: ").strip().lower() != "yes":
            return 1
        nose.set_illumination(s.illumination or "brightfield", s.illumination_level)
        if a.at:
            core.setXYPosition(XY, a.at[0], a.at[1])
            core.waitForDevice(XY)
            set_z(a.at[2] - 5)
            set_z(a.at[2])
        z10 = core.getPosition(Z)
        core.setExposure(s.exposure_ms)
        e10 = expose()
        A = snap()
        Image.fromarray(A).save(out / "offset_10x.png")
        ze = z10 + s.focus_offset_um
        print(f"10x: Z {z10:.2f}, exposure {e10:.1f} ms; 100x focus expected at Z {ze:.2f}")
        set_z(ze - 20.0)
        nose.set_position(pos["100x"])
        time.sleep(1.0)
        core._obj = (0.0, None)
        print("100x in; focus limit", core.focusLimitsUm())
        core.setExposure(s.target_exposure_ms)
        e100 = expose()                                 # (once, before the sweeps: the sharpness of two frames compares
        zs = np.arange(ze - 10.0, ze + 10.01, 1.0)      # only at the same exposure)
        if a.z100 is not None:                          # (focused by eye: only a short fine sweep around it)
            zs = np.array([a.z100])
        best, sweep = None, []
        for z in zs:                                    # (coarse, upwards)
            set_z(float(z))
            sh = sharpness(snap())
            sweep.append((float(z), sh))
            if best is None or sh > best[1]:
                best = (float(z), sh)
        print("  coarse sweep (Z: sharpness x1000): " + " ".join(f"{z:.0f}:{1000 * v:.0f}" for z, v in sweep))
        zf = np.arange(best[0] - 2.0, best[0] + 2.01, 0.25)
        set_z(float(zf[0]) - 3.0)
        sh = []
        for z in zf:
            set_z(float(z))
            sh.append(sharpness(snap()))
        print("  fine sweep: " + " ".join(f"{z:.2f}:{1000 * v:.0f}" for z, v in zip(zf, sh)))
        z100 = float(zf[int(np.argmax(sh))])
        set_z(z100 - 3.0)
        set_z(z100)
        B = snap()
        Image.fromarray(B).save(out / "offset_100x.png")
        print(f"100x: sharpest at Z {z100:.2f} (sharpness {max(sh):.4f}, at the ends {sh[0]:.4f}, {sh[-1]:.4f}), exposure {e100:.1f} ms")
        set_z(z100 - 20.0)
        nose.set_position(pos["10x"])
        time.sleep(1.0)
        core._obj = (0.0, None)
        core.setExposure(e10)
        set_z(z10 - 5)
        set_z(z10)
        cx, cy, ratio, corr = locate(A, B, s.ratio_100x)
        H, W = A.shape[:2]
        off = -(M @ np.array([cx - W / 2.0, cy - H / 2.0]))
        print(f"the 100x picture is in the 10x one at pixel {cx:.1f}, {cy:.1f} (centre {W / 2:.0f}, {H / 2:.0f}); correlation {corr:.2f}")
        print(f"\noffset_100x_um = [{off[0]:.1f}, {off[1]:.1f}]   ratio_100x = {ratio:.3f}   focus_offset_um = {z100 - z10:.1f}")
        if corr < 0.5 or max(sh) < 1.15 * max(sh[0], sh[-1]):
            print("WARNING: weak match or no clear focus: not trustworthy (a place with more small details?)")
            return 1
        if a.write:
            p = Path(a.settings)
            d = json.loads(p.read_text()) if p.is_file() else {}
            if p.is_file():
                shutil.copy2(p, p.with_name(p.stem + time.strftime("_before_%Y%m%d_%H%M%S") + p.suffix))
            d.update(offset_100x_um=[round(float(off[0]), 1), round(float(off[1]), 1)], focus_offset_um=round(z100 - z10, 1))
            p.write_text(json.dumps(d, indent=1))
            print(f"offset_100x_um and focus_offset_um written to {p}")
        return 0
    finally:
        try:
            nose.close()
        finally:
            core.unloadAllDevices()


if __name__ == "__main__":
    sys.exit(main())
