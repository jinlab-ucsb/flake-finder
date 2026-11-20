"""TRIAL (2026-11-16, paths of the test chip written in): can the 100x colour tell the 10x colour's ambiguous thicknesses
apart?  Measures the flake colour in the middle of every raw 100x picture, fits the 100x light cone, stray light and (--lamp)
lamp on flakes with one 10x reading, then tests the ambiguous ones.  py -3.9 tools/bx53m/check100_colour_trial.py --lamp"""
import sys, csv, json, numpy as np, cv2
sys.path.insert(0, r"c:\Users\Glovebox\Documents\GitHub\flake-finder")
from scipy.optimize import differential_evolution
from flakeid import real100
from flakeid.optics import ContrastModel, CameraModel
from flakeid.thickness import load_profile, camera_from_profile

R = "C:/FlakeScans/"
S = str(__import__("pathlib").Path(__import__("tempfile").gettempdir()))
OX = 295.5
rows = lambda p: list(csv.DictReader(open(p, newline="")))


def measure():
    z = np.load(r"c:\Users\Glovebox\Documents\GitHub\flake-finder\flats\flat_100x_dp22_raw_285nm.npz")
    flat = z["flat"].astype(np.float32)[::2, ::2]
    dark = np.median(z["dark"].reshape(-1, 3), axis=0)
    T = rows(R + "results/raw_run_3_led3/raw_3/raw_3_tiles_live_terraces.csv")
    P = np.array([[float(t["stage_x_um"]), float(t["stage_y_um"])] for t in T]); D = np.array([float(t["inscribed_diameter_um"]) for t in T])
    out = []
    for name in ("all100_1", "raw_3"):
        h = R + name + "_100x/"
        for r in rows(h + name + "_100x_index.csv"):
            q = np.array([float(r["stage_x_um"]) - 21.1, float(r["stage_y_um"]) + 38.4])
            d = np.hypot(*(P - q).T); ins = np.flatnonzero(d < D / 2)
            if not len(ins):
                continue
            t = T[int(ins[np.argmin(d[ins])])]
            raw = cv2.imread(h + r["file"], cv2.IMREAD_UNCHANGED)
            if raw is None:
                raw = cv2.imread(h + "rejected/" + r["file"], cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue
            ch = [np.asarray(c, np.float32) for c in real100.channels(raw, 0.0)]
            hh, ww = ch[0].shape
            f = cv2.resize(flat, (ww, hh), interpolation=cv2.INTER_AREA) if flat.shape[:2] != (hh, ww) else flat
            ch = [(c - dark[k]) / np.maximum(f[..., k], 0.2) for k, c in enumerate(ch)]
            sub = []
            for c in ch:
                hist = np.bincount(np.clip(c[::2, ::2], 0, 4095).astype(np.int64).ravel()); hist = np.convolve(hist, np.ones(9), "same")
                sub.append(float(np.argmax(hist)))
            cy, cx, w = hh // 2, ww // 2, 45                       # ~10 um square in the middle
            patch = np.stack([c[cy - w:cy + w, cx - w:cx + w] for c in ch], 2).reshape(-1, 3)
            med = np.median(patch, 0); mad = np.median(np.abs(patch - med), 0) / np.maximum(med, 1)
            out.append(dict(pic=f"{name} #{int(r['order']):03d}", terrace=t["terrace_id"], t10=float(t["thickness_nm"]),
                            alt=[float(x) for x in t["alt_thickness_nm"].split(";") if x], c10=[float(t["contrast_" + k]) for k in "rgb"],
                            c100=(med / np.array(sub) - 1).tolist(), mad=float(mad.max()), sub=sub))
    json.dump(out, open(S + r"\c100.json", "w"), indent=0)
    return out


CAM = camera_from_profile(load_profile("bx53m_dp22_raw_led"))


def curve(na, flare, d):
    flare = list(flare)
    led, irc = CAM.led, CAM.ir_cut_nm
    if len(flare) > 3:                      # + the lamp through this objective: (blue/phosphor ratio, phosphor centre, width), IR cut
        led = (CAM.led[0], CAM.led[1], flare[4], flare[5], flare[3]); irc = flare[6]
    cam = CameraModel(**{**CAM.__dict__, "flare": tuple(flare[:3]), "led": led, "ir_cut_nm": irc})
    return np.minimum(ContrastModel("hbn", OX, na=na, camera=cam, n_angles=8).contrast(np.atleast_1d(d)), CLIP[None])


CLIP = np.array([8.0, 3.1, 0.78])          # the camera clips: blue at 3600 counts over a substrate of 1992


class Loss:
    def __init__(self, d, c):
        self.d, self.c = d, c

    def __call__(self, x):
        e = np.sort(np.linalg.norm(curve(x[0], x[1:], self.d) - self.c, axis=1))
        return float(np.mean(e[: int(0.85 * len(e))] ** 2))


if __name__ == "__main__":
    import os
    M = json.load(open(S + r"\c100.json")) if os.path.exists(S + r"\c100.json") and "--again" not in sys.argv else measure()
    gsub = np.median(np.array([m["sub"] for m in M]), 0)
    for m in M:
        m["c100"] = ((np.array(m["c100"]) + 1) * np.array(m["sub"]) / gsub - 1).tolist()
    n0 = len(M)
    M = [m for m in M if m["mad"] < 0.06 and np.linalg.norm(m["c100"]) > 0.08]
    print(n0, "pictures on a region;", len(M), "with a flat flake (not bare substrate) in the middle")
    seen, U = set(), []
    for m in M:                                    # one picture per region
        if m["terrace"] not in seen:
            seen.add(m["terrace"]); U.append(m)
    one = [m for m in U if not m["alt"]]; two = [m for m in U if m["alt"]]
    print(len(U), "regions with a flat middle at 100x;", len(one), "with one 10x reading,", len(two), "ambiguous")
    d = np.array([m["t10"] for m in one]); c = np.array([m["c100"] for m in one])
    res = differential_evolution(Loss(d, c), [(0.3, 0.95), (0, .6), (0, .6), (0, .6)] + ([(0.2, 20), (530, 610), (90, 190), (630, 720)] if "--lamp" in sys.argv else []),
                                 seed=1, maxiter=120, popsize=12, polish=True)
    na, fl = res.x[0], res.x[1:]
    e = np.linalg.norm(curve(na, fl, d) - c, axis=1)
    print(f"fit on the unambiguous: light cone NA {na:.2f}, stray light {np.round(fl, 3)}; residual median {np.median(e):.3f}, p90 {np.percentile(e, 90):.3f}")
    # blind: the 100x colour alone against the whole 0-400 nm curve
    grid = np.arange(0, 400.5, 0.5); cur = curve(na, fl, grid)
    def blind(c):
        dist = np.linalg.norm(cur - np.asarray(c)[None], axis=1); return grid[int(np.argmin(dist))], float(dist.min()), dist
    b = np.array([blind(m["c100"])[0] for m in one])
    print("100x colour alone vs the 10x thickness (unambiguous): within 5 nm", int((abs(b - d) < 5).sum()), "within 15 nm", int((abs(b - d) < 15).sum()), "of", len(one))
    for m, bb, ee in sorted(zip(one, b, e), key=lambda q: q[0]["t10"]):
        if abs(bb - m["t10"]) >= 15:
            print(f"   off: {m['pic']}  10x {m['t10']:.0f} nm, 100x alone {bb:.0f} nm; C100 {np.round(m['c100'], 2)}  residual at the 10x value {ee:.2f}")
    print("\nambiguous at 10x:  picture, 10x choice / alternatives -> 100x residual at each, 100x alone")
    for m in sorted(two, key=lambda q: q["t10"]):
        cand = [m["t10"]] + m["alt"]
        _, _, dist = blind(m["c100"])
        r = [float(dist[np.abs(grid - t) <= 6].min()) for t in cand]          # (best within 6 nm of each candidate)
        k = int(np.argmin(r))
        print(f"   {m['pic']:14s} region {m['terrace']:>4s}  " + "  ".join(f"{t:.0f} nm: {x:.2f}" for t, x in zip(cand, r))
              + f"   -> {cand[k]:.0f} nm" + ("  (clear)" if sorted(r)[1] > 2 * sorted(r)[0] + 0.03 else "  (close)") + f"   100x alone {blind(m['c100'])[0]:.0f}   C100 {np.round(m['c100'], 2)}")
