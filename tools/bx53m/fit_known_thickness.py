"""Check the thin-film model against flakes of known thickness photographed on ANOTHER microscope.

    py -3.9 tools/bx53m/fit_known_thickness.py TABLE.txt [--fix-n] [--split N] [-o OUT.json]

TABLE.txt: lines "thickness_nm R G B", the flake's brightness over the substrate's per channel (1 = as the substrate);
a missing channel is left out (fixed columns as in the lab's table, or 4 numbers).  --split N: the rows from N on are a
second set (other oxide, same camera).

The other camera's channels, lamp, stray light and tone curve are unknown and are fitted; they say nothing about this
microscope.  What carries over is the material: the colours repeat with the optical thickness n x d, so known
thicknesses give hBN's refractive index (a factor on optics.n_hbn), which a fit without known thicknesses cannot.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flakeid.optics import CameraModel, ContrastModel, n_hbn  # noqa: E402

NAMES = ["c_r", "c_g", "c_b", "w_r", "w_g", "w_b", "ir_cut", "lamp_k", "f_r", "f_g", "f_b", "gamma", "oxide", "oxide2", "n_scale"]
BOUNDS = [(570, 650), (500, 565), (430, 495), (30, 140), (30, 140), (30, 140), (620, 720), (2600, 7000), (0, .4), (0, .4), (0, .4),
          (1.0, 2.6), (250, 330), (250, 330), (0.85, 1.15)]
WL = np.arange(400.0, 721.0, 10.0)


def read(path):
    out = []
    for line in Path(path).read_text().splitlines()[1:]:
        if not line.strip():
            continue
        v = line.split()
        if len(v) == 4:
            out.append([float(x) for x in v])
            continue
        # fixed columns: thickness, then R, G, B right-aligned near columns 21, 29, 38 (a blank = not measured)
        m = [(k.start(), float(k.group())) for k in re.finditer(r"[\d.]+", line)]
        row = [m[0][1], np.nan, np.nan, np.nan]
        for pos, val in m[1:]:
            end = pos + len(str(val))
            row[1 + int(np.argmin([abs(end - 21), abs(end - 29), abs(end - 38)]))] = val
        out.append(row)
    return np.array(out)


def predict(p, d, oxide):
    cam = CameraModel(lamp_temp_k=p["lamp_k"], centers_nm=(p["c_r"], p["c_g"], p["c_b"]), fwhm_nm=(p["w_r"], p["w_g"], p["w_b"]),
                      ir_cut_nm=p["ir_cut"], gamma=p["gamma"], flare=(p["f_r"], p["f_g"], p["f_b"]))
    tab = np.stack([WL, n_hbn(WL).real * p["n_scale"], np.zeros_like(WL)], axis=1)
    return ContrastModel("hbn", oxide, na=0.3, camera=cam, n_angles=3, wl=WL, index_table=tab).contrast(d) + 1.0


class Loss:
    def __init__(self, sets, fix_n):
        self.sets, self.fix_n = sets, fix_n

    def resid(self, x):
        p = dict(zip(NAMES, x))
        if self.fix_n:
            p["n_scale"] = 1.0
        r = []
        for k, s in enumerate(self.sets):
            e = np.log(predict(p, s[:, 0], p["oxide" if k == 0 else "oxide2"])) - np.log(s[:, 1:])
            r.append(e[np.isfinite(e)])
        return np.concatenate(r)

    def __call__(self, x):
        e = np.sort(np.abs(self.resid(x)))
        return float(np.mean(e[: int(0.9 * len(e))] ** 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table")
    ap.add_argument("--fix-n", action="store_true", help="hBN's index as in the program (factor 1)")
    ap.add_argument("--split", type=int, help="rows from this one on (0-based) are a second set")
    ap.add_argument("-o", "--output")
    a = ap.parse_args()
    t = read(a.table)
    sets = [t[:a.split], t[a.split:]] if a.split else [t]
    loss = Loss(sets, a.fix_n)
    res = differential_evolution(loss, BOUNDS, popsize=12, maxiter=150, tol=1e-8, seed=1, polish=True, workers=-1, updating="deferred")
    p = dict(zip(NAMES, res.x))
    if a.fix_n:
        p["n_scale"] = 1.0
    e = loss.resid(res.x)
    print(f"{len(t)} flakes ({[len(s) for s in sets]}), {len(e)} values; rms log error {np.sqrt(res.fun):.3f} (best 90 %), all {np.sqrt(np.mean(e ** 2)):.3f}")
    print(json.dumps({k: round(float(v), 3) for k, v in p.items()}))
    if a.output:
        Path(a.output).write_text(json.dumps(dict(params=p, loss=res.fun, table=str(a.table)), indent=1))


if __name__ == "__main__":
    main()
