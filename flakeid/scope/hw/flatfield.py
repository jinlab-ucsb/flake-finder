"""Flat-field frames on the glovebox microscope without the windowed flat-field job: bare substrate at a grid of stage
positions, then dark frames with the lamp switched off by the control box, written as tile folders that
python -m flakeid.mm flat turns into the flat field.

    py -3.9 -m flakeid.scope.hw.flatfield [--settings scan_settings.json] [--objective 10x] [--span 800] [--grid 5]

THIS MOVES THE STAGE by up to --span um around where it is (it asks first); the focus is not touched.  Before: the
objective in, an empty piece of the wafer in focus (on its edge or a speck of dust, then moved to a clean part).  Dust
on the chip is at another pixel in every frame and drops out of the median; dust in the light path stays, as it should.

Writes <local_root>/flat_<objective>_<date>/blank and /dark, then flats/flat_<objective>_dp22_raw.npz in the program
folder, and prints the substrate's colour (the reference for this camera).  --use writes the flat into the settings
file (flat10, or flat100 for the nominal oxide given with --oxide).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


class _UI:
    stop_requested = pause_requested = False
    headless_analysis_off = True
    preview = None
    def log(self, msg): pass
    def message(self, title, msg): print(title, msg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings")
    ap.add_argument("--objective", default="10x", choices=["10x", "100x"])
    ap.add_argument("--span", type=float, default=800.0, help="half width of the grid of positions, um (default 800)")
    ap.add_argument("--grid", type=int, default=5, help="positions per side (default 5: 25 frames)")
    ap.add_argument("--dark-frames", type=int, default=10)
    ap.add_argument("--oxide", type=float, default=285.0, help="the wafer's nominal oxide, nm (for the names)")
    ap.add_argument("--use", action="store_true", help="write the flat into the settings file")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    from ..scan import Scanner
    from ..settings import load
    from .bx53m import XY, make_core
    s = load(a.settings, hardware="bx53m", verbose=False, save_colour=False)
    hundred = a.objective == "100x"
    core = make_core(s)
    core.loadSystemConfiguration()
    sc = Scanner(core, s, _UI())
    out = Path(s.local_root) / time.strftime(f"flat_{a.objective}_%Y%m%d_%H%M%S")
    try:
        sc.setup_devices()
        if sc.objective()[1] != a.objective:
            print(f"the {a.objective} objective is not in (now: {sc.objective()[1]})")
            return 1
        if not a.yes and input(f"The stage will move up to {a.span:g} um around here.  Type yes: ").strip().lower() != "yes":
            return 1
        sc.view_scale = 1.0 / s.ratio_100x if hundred else 1.0
        sc.apply_camera(s.target_exposure_ms if hundred else s.exposure_ms)
        sc.DARK = s.dark * (sc.full_scale() + 1) / 4096.0
        sc.check_exposure(True)
        exp = core.getExposure()
        x0, y0 = core.getXPosition(XY), core.getYPosition(XY)
        sc.tile_objective = a.objective
        span = a.span / (s.ratio_100x if hundred else 1.0) * (3.0 if hundred else 1.0)      # (100x: +-3 fields of its own)
        offs = np.linspace(-span, span, a.grid)
        k = 0
        for j, dy in enumerate(offs):
            for i, dx in enumerate(offs if j % 2 == 0 else offs[::-1]):
                sc.move_to(x0 + dx, y0 + dy)
                lab = f"Pos-{i:03d}_{j:03d}"
                sc.save_tile(out / "blank" / lab, k, lab, i, j)
                k += 1
        sc.move_to(x0, y0)
        f = sc.frame()
        bl = sc.blocks(f)
        rgb = [float(np.median(bl[:, c])) for c in (2, 3, 4)]
        print(f"{k} frames at {exp:.2f} ms; substrate (dark removed) R {rgb[0]:.0f}  G {rgb[1]:.0f}  B {rgb[2]:.0f}: "
              f"R/G {rgb[0] / rgb[1]:.4f}, B/G {rgb[2] / rgb[1]:.4f}")
        sc.nose._set("EILSW 0")                         # (the lamp off: dark frames)
        time.sleep(1.5)
        for d in range(a.dark_frames):
            lab = f"Pos-{d:03d}_000"
            sc.save_tile(out / "dark" / lab, d, lab, d, 0)
        dk = sc.frame()
        print(f"{a.dark_frames} dark frames (lamp off): median {float(np.median(dk)):.1f} counts")
        sc.nose.set_illumination(s.illumination or "brightfield", s.illumination_level)
        if sc.writer is not None:
            sc.writer.flush()
    finally:
        sc.close_nosepiece()
        core.unloadAllDevices()
    flats = Path(s.flats_dir) if s.flats_dir else ROOT / "flats"
    flats.mkdir(parents=True, exist_ok=True)
    npz = flats / f"flat_{a.objective}_dp22_raw{'' if not hundred else f'_{a.oxide:g}nm'}.npz"
    r = subprocess.run([sys.executable, "-m", "flakeid.mm", "flat", str(out / "blank"), "--dark", str(out / "dark"), "-o", str(npz)],
                       cwd=str(ROOT), capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-1500:])
    if r.returncode:
        return 1
    (out / "substrate.json").write_text(json.dumps(dict(objective=a.objective, oxide_nominal_nm=a.oxide, exposure_ms=exp, substrate_counts=rgb,
                                                        bit_depth=12, flat=str(npz), date=time.strftime("%Y-%m-%d %H:%M")), indent=1))
    if a.use and a.settings:
        p = Path(a.settings)
        d = json.loads(p.read_text())
        if hundred:
            d["flat100"] = [e for e in d.get("flat100", []) if not str(e).startswith(f"{a.oxide:g}=")] + [f"{a.oxide:g}={npz.as_posix()}"]
        else:
            d["flat10"] = npz.as_posix()
        p.write_text(json.dumps(d, indent=1))
        print("written into", p)
    print("flat field:", npz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
