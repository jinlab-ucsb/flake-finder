"""Remeasure flake coordinates (Avram, 2026-10-29): the chips of a finished run were moved (knocked, or taken off and put
back), so its microscope stage coordinates are off; they are put right without imaging and screening everything again.
(The designer's AFM lists work in chip coordinates, which a move on the microscope does not change.)

The start screen's "Remeasure flake coordinates" since 2026-10-30c (Scanner.run_remeasure): no new scan.  The flakes
imaged at 100x (all, the kept ones, or a list: the designer's "Export these flakes (JSON)", parse_flake_list;
select_visits) are measured at 100x again and again until two passes in a row agree within remeasure_agree_um (3 um)
for every flake after the shift they share (compare_passes): the stage slips by tens of um between rows, so one
measurement per flake is not trusted.  Two calibration spots (calib_spots, calib_picture: crops of the old 10x scan
near both ends of what was on the stage) centred by hand give the rough rigid move; each flake is then found at 10x
(locate), then at 100x pass after pass (Scanner.find100: here, else a 3 x 3 mosaic of 100x frames stitched by stage
position (stitch), else by hand or skipped).  The last pass is applied (apply_positions: as below, each flake to where
it was found, the rest by the fitted move) and tabled (write_passes, <piece>_remeasure_passes.csv).

Before that (and still from the command line below): the chips are scanned at 10x again (the same chips / pieces / polygon
windows as any scan, "10x scan only" with the stitched image and its viewer, its own run), then each new piece's 10x
image is matched to the old run's pieces (whole-image features, then full-resolution patches), giving per piece the
chip's rigid move new = R old + t (rotation and shift, in stage um).  Then (2026-10-30, Scanner.remeasure_visit) the microscope visits each
flake imaged at 100x where that move puts it, takes one 10x frame and finds the flake's own 100x picture in it
(locate): each flake found moves to where it was found (FlakeMoves), everything else by the rigid move most visited
flakes agree with (fit_visits); one new 10x scan cannot be trusted flake by flake (20261027_run_13: whole rows of 10x
tiles 80-150 um off, a different set in each scan).  The old run's own folder is then changed in place
(2026-10-29; a copy into the new run before): every stage coordinate of each matched piece is moved: the viewer (flake
and terrace positions, its pixel -> stage map), the 100x and IR pictures' positions and outlines, the target lists, the
tables, the chip file, the run file, the designer's bundle (designer/bundle.json: a page refresh shows the new
coordinates, no rebuild).  Each file is first saved to <old run>/old_coords/<time>/ (restore() / --restore puts it
back).  Pictures, scores and everything else stay as they are.

A picture taken with another objective was at (10x position + that objective's offset): it moves as
    new = R (old - offset) + t + offset
(100x: offset_100x_um of its regions file, 24, -18.3 by default; IR: the 100x offset plus the IR camera's, from the
domain index against the 100x index).  The designer's AFM centres (100x/afm_centers.json: offsets from their 100x
picture) stay as they are: the designer keeps drawing the old pictures on the old chip map, where they still fit.  Focus
(z) values are not changed, nor are relative shifts (the
100x index's shift_um / recentre_um: a few um, turned by the chip's rotation by less than 0.1 um per degree).

    python -m flakeid.scope.remeasure OLD_RUN_DIR NEW_VIEWER_DIR [NEW_VIEWER_DIR ...]
    python -m flakeid.scope.remeasure OLD_RUN_DIR --restore [TIME]
"""
from __future__ import annotations

from flakeid.hwconst import OFFSET_100X, RATIO_100X  # noqa: F401  (the microscope's 100x ratio and offset)

import csv
import io
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np

# (OFFSET_100X: flakeid.hwconst)         # 10x -> 100x stage offset (flakeid.hundred.OFFSET_100X: the analysis's default)
MIN_INLIERS = 25                    # whole-image feature matches a piece pair needs to count as the same chip
OVERVIEW_DOWN = 4                   # the whole-image match: the viewer pyramid level at 1/4 (~4.4 um per pixel)
PATCH_PX = 256                      # full-resolution patches of the fine match (~280 um)
SEARCH_PX = 40                      # ... searched this far around the whole-image estimate
MIN_PEAK = 0.5                      # normalized correlation a patch / flake match needs
N_SPOTS = 40                        # patches for the fine fit
FLAKE_PX = 200                      # each flake's check: a patch of this size around it
FLAKE_MIN_PEAK = 0.4                # ... the correlation it needs (and to be within FLAKE_OK_UM of the chip's move)
FLAKE_OK_UM = 3.0                   # a flake found within this of where the chip's move puts it: confirmed
CHECK_DIR = "10x_check"             # (before 2026-10-29 the new 10x scans went to <new run>/10x_check/: not a piece)


# ---- the viewers' 10x images and their pixel -> stage maps
def read_viewer(vdir):
    """(D, html) of a chip viewer (<vdir>/index.html, const D = {...})."""
    html = (Path(vdir) / "index.html").read_text(encoding="utf-8")
    m = re.search(r"const D=(\{.*?\});\n", html, re.S)
    if not m:
        raise ValueError(f"{vdir}: not a chip viewer (no const D)")
    return json.loads(m.group(1)), html


def write_viewer_data(vdir, D):
    """Replace the const D = {...} of <vdir>/index.html."""
    p = Path(vdir) / "index.html"
    html = p.read_text(encoding="utf-8")
    m = re.search(r"const D=(\{.*?\});\n", html, re.S)
    p.write_text(html[:m.start(1)] + json.dumps(D, separators=(",", ":")) + html[m.end(1):], encoding="utf-8")


class Canvas:
    """A viewer's stitched 10x image (its tile pyramid) and pixel <-> stage um (the viewer's stage_map: the nearest
    tile's position and the camera matrix; viewers without one: an affine fit to the flake table)."""

    def __init__(self, vdir):
        self.dir = Path(vdir)
        self.D, _ = read_viewer(vdir)
        I = self.D["info"]["image"]
        self.W, self.H, self.L, self.T = int(I["width"]), int(I["height"]), int(I["maxLevel"]), int(I.get("tileSize") or 256)
        sm = self.D.get("stage_map")
        if sm and sm.get("P"):
            self.M = np.asarray(sm["M"], float)
            self.P = np.asarray(sm["P"], float)
            self.C = np.asarray(sm["q"], float) + np.asarray(sm["wh"], float) / 2.0
        else:
            px = [(f["tx"], f["ty"]) for f in self.D.get("flakes", []) if f.get("sx") is not None]
            # (the 10x positions: a flake placed by its 100x picture, flakeid.pos100, holds its correction in pos)
            st = [pos10(f) for f in self.D.get("flakes", []) if f.get("sx") is not None]
            if len(px) < 3:
                raise ValueError(f"{vdir}: no stage map and too few flakes with stage positions")
            A = np.c_[np.asarray(px, float), np.ones(len(px))]
            coef = np.linalg.lstsq(A, np.asarray(st, float), rcond=None)[0]
            self.M, self.P, self.C = coef[:2].T.copy(), coef[2][None, :].copy(), np.zeros((1, 2))
        self.Minv = np.linalg.inv(self.M)
        self.um_px = math.sqrt(abs(np.linalg.det(self.M)))
        self._cache = {}

    def to_stage(self, px):
        p = np.atleast_2d(np.asarray(px, float))
        k = np.argmin(((p[:, None, :] - self.C[None]) ** 2).sum(-1), axis=1)
        return self.P[k] + (p - self.C[k]) @ self.M.T

    def to_px(self, st):
        s = np.atleast_2d(np.asarray(st, float))
        p = (s - self.P[0]) @ self.Minv.T + self.C[0]                # (one affine for a live canvas; refined below)
        for _ in range(2):
            k = np.argmin(((p[:, None, :] - self.C[None]) ** 2).sum(-1), axis=1)
            p = (s - self.P[k]) @ self.Minv.T + self.C[k]
        return p

    def _tile(self, level, tx, ty, color=False):
        key = (level, tx, ty, color)
        if key not in self._cache:
            import cv2
            if len(self._cache) > 400:
                self._cache.clear()
            im = cv2.imread(str(self.dir / "tiles" / "image" / str(level) / f"{tx}_{ty}.jpg"),
                            cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
            self._cache[key] = im
        return self._cache[key]

    def region(self, x0, y0, w, h, down=1, color=False):
        """Grey pixels [y0, y0 + h) x [x0, x0 + w) of the image at 1/down (a power of two), float 0..1; NaN off it.
        color: BGR (h, w, 3) instead."""
        level = self.L - int(round(math.log2(down)))
        out = np.full((h, w, 3) if color else (h, w), np.nan, np.float32)
        T = self.T
        for ty in range(max(y0 // T, 0), (y0 + h - 1) // T + 1):
            for tx in range(max(x0 // T, 0), (x0 + w - 1) // T + 1):
                im = self._tile(level, tx, ty, color)
                if im is None:
                    continue
                ys, xs = ty * T - y0, tx * T - x0
                a, b = max(ys, 0), max(xs, 0)
                c, d = min(ys + im.shape[0], h), min(xs + im.shape[1], w)
                if c > a and d > b:
                    out[a:c, b:d] = im[a - ys:c - ys, b - xs:d - xs] / 255.0
        return out

    def overview(self, down=OVERVIEW_DOWN):
        return self.region(0, 0, -(-self.W // down), -(-self.H // down), down)


def pos10(f):
    """A viewer flake's position in its 10x picture: sx, sy less the correction a 100x picture gave it (flakeid.pos100)."""
    p = f.get("pos") or [0.0, 0.0]
    return f["sx"] - p[0], f["sy"] - p[1]


def _band(g, sigma_hi=1.0, sigma_lo=12.0):
    """Band-passed grey (flake edges, not the illumination), NaN -> 0."""
    import cv2
    g = np.nan_to_num(g, nan=float(np.nanmedian(g)) if np.isfinite(g).any() else 0.0)
    return cv2.GaussianBlur(g, (0, 0), sigma_hi) - cv2.GaussianBlur(g, (0, 0), sigma_lo)


def rigid_fit(src, dst):
    """Rotation R and translation t with dst ~ R src + t (least squares, no scaling or mirroring), and the rms."""
    A, B = np.asarray(src, float), np.asarray(dst, float)
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    t = mb - R @ ma
    res = B - (A @ R.T + t)
    return R, t, float(np.sqrt((res ** 2).sum(1).mean()))


def coarse_match(old: Canvas, new: Canvas, down=OVERVIEW_DOWN):
    """The chip's move from whole-image features (ORB on both images at 1/down), fitted in stage um with RANSAC:
    dict(R, t, inliers, matches) or None (not the same chip)."""
    import cv2

    def feats(cv):
        if getattr(cv, "_feats", None) is None:     # (each piece is matched against several: computed once)
            g = cv.overview(down)
            u8 = np.clip(np.nan_to_num(g, nan=0.0) * 255, 0, 255).astype(np.uint8)
            u8 = cv2.createCLAHE(2.0, (8, 8)).apply(u8)
            orb = cv2.ORB_create(nfeatures=20000, scaleFactor=1.3, nlevels=4, fastThreshold=10)
            cv._feats = orb.detectAndCompute(u8, None)
        return cv._feats
    ko, do = feats(old)
    kn, dn = feats(new)
    if do is None or dn is None or len(ko) < MIN_INLIERS or len(kn) < MIN_INLIERS:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(do, dn, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) < MIN_INLIERS:
        return None
    so = old.to_stage(np.array([ko[m.queryIdx].pt for m in good]) * down)
    sn = new.to_stage(np.array([kn[m.trainIdx].pt for m in good]) * down)
    A, inl = cv2.estimateAffinePartial2D(so.astype(np.float32), sn.astype(np.float32), method=cv2.RANSAC,
                                         ransacReprojThreshold=4.0 * old.um_px * down / 2, maxIters=20000, confidence=0.999)
    if A is None or inl is None:
        return None
    inl = inl.ravel().astype(bool)
    if inl.sum() < MIN_INLIERS:
        return None
    scale = math.hypot(A[0, 0], A[1, 0])
    if abs(scale - 1.0) > 0.03:                     # (the same microscope: the same scale)
        return None
    R, t, rms = rigid_fit(so[inl], sn[inl])
    return dict(R=R, t=t, inliers=int(inl.sum()), matches=len(good), rms_um=rms)


def _subpix(r, y, x):
    def p(a, b, c):
        d = a - 2 * b + c
        return 0.0 if abs(d) < 1e-9 else 0.5 * (a - c) / d
    dy = p(r[y - 1, x], r[y, x], r[y + 1, x]) if 0 < y < r.shape[0] - 1 else 0.0
    dx = p(r[y, x - 1], r[y, x], r[y, x + 1]) if 0 < x < r.shape[1] - 1 else 0.0
    return x + dx, y + dy


def patch_match(old: Canvas, new: Canvas, R, t, s_old, size=PATCH_PX, search=SEARCH_PX):
    """Where the old image around old stage point s_old is in the new image: (new stage point, peak) or None.  The old
    patch is warped into the new image's pixel frame by the current estimate, then found by normalized correlation
    within search px."""
    import cv2
    pn = new.to_px([R @ np.asarray(s_old, float) + t])[0]
    # new pixel -> old pixel near this point (affine): new px -> new stage -> old stage -> old px
    J = old.Minv @ R.T @ new.M
    po = old.to_px([s_old])[0]
    h = size // 2
    big = size + 2 * search
    x0, y0 = int(round(pn[0])) - h - search, int(round(pn[1])) - h - search
    reg = new.region(x0, y0, big, big)
    if np.isnan(reg).mean() > 0.3:
        return None
    # template: old pixels at J (q - centre_new) + po for q on the new grid of the patch
    m = int(math.ceil(size * 0.75))
    src = old.region(int(po[0]) - m, int(po[1]) - m, 2 * m, 2 * m)
    if np.isnan(src).mean() > 0.3:
        return None
    off = po - (np.array([int(po[0]) - m, int(po[1]) - m]))
    A = np.hstack([J, (off - J @ np.array([h, h]))[:, None]]).astype(np.float32)
    tpl = cv2.warpAffine(np.nan_to_num(src, nan=0.0), A, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    tb, rb = _band(tpl), _band(reg)
    if tb.std() < 1e-3:
        return None
    r = cv2.matchTemplate(rb.astype(np.float32), tb.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, peak, _, (mx, my) = cv2.minMaxLoc(r)
    sx, sy = _subpix(r, my, mx)
    found = np.array([x0 + sx + h, y0 + sy + h])
    return new.to_stage([found])[0], float(peak)


def spots(old: Canvas, n=N_SPOTS):
    """Old stage points for the fine fit: the flakes (or a grid where there is no flake table), spread over the piece
    (farthest-point order)."""
    pts = np.array([pos10(f) for f in old.D.get("flakes", []) if f.get("sx") is not None], float)
    if len(pts) < n:
        gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 7) * old.W, np.linspace(0.1, 0.9, 7) * old.H)
        pts = np.vstack([pts.reshape(-1, 2), old.to_stage(np.c_[gx.ravel(), gy.ravel()])])
    if len(pts) <= n:
        return pts
    c = pts.mean(0)
    sel = [int(np.argmin(((pts - c) ** 2).sum(1)))]
    d = ((pts - pts[sel[0]]) ** 2).sum(1)
    while len(sel) < n:
        k = int(np.argmax(d))
        sel.append(k)
        d = np.minimum(d, ((pts - pts[k]) ** 2).sum(1))
    return pts[sel]


def register(old: Canvas, new: Canvas, coarse=None, log=print):
    """The chip's move old -> new: dict(R, t, rot_deg, shift_um, rms_um, n_spots, coarse) or None."""
    c = coarse or coarse_match(old, new)
    if c is None:
        return None
    R, t = c["R"], c["t"]
    src, dst = [], []
    for s in spots(old):
        m = patch_match(old, new, R, t, s)
        if m is not None and m[1] >= MIN_PEAK:
            src.append(s)
            dst.append(m[0])
    src, dst = np.array(src), np.array(dst)
    keep = np.ones(len(src), bool)
    rms = float("nan")
    for _ in range(4):
        if keep.sum() < 3:
            break
        R, t, rms = rigid_fit(src[keep], dst[keep])
        res = np.sqrt(((dst - (src @ R.T + t)) ** 2).sum(1))
        new_keep = res <= max(3.0 * rms, 1.5)
        if (new_keep == keep).all():
            break
        keep = new_keep
    if keep.sum() < 3:
        log(f"  fine match: only {int(keep.sum())} patches agree; the whole-image estimate is used")
        R, t, rms = c["R"], c["t"], c["rms_um"]
    rot = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    ctr = src[keep].mean(0) if keep.sum() else np.zeros(2)
    return dict(R=R, t=t, rot_deg=rot, shift_um=(R @ ctr + t - ctr).tolist(), rms_um=rms, n_spots=int(keep.sum()),
                n_tried=len(src), coarse=dict(inliers=c["inliers"], matches=c["matches"], rms_um=c["rms_um"]))


def check_flakes(old: Canvas, new: Canvas, R, t):
    """Each old flake found again in the new image: rows (id, old sx, sy, new sx, sy, found offset um, peak, ok)."""
    rows = []
    for f in old.D.get("flakes", []):
        if f.get("sx") is None:
            continue
        s = np.array(pos10(f), float)
        ns = R @ s + t
        m = patch_match(old, new, R, t, s, size=FLAKE_PX, search=12)
        d = float(np.hypot(*(m[0] - ns))) if m else float("nan")
        ok = bool(m and m[1] >= FLAKE_MIN_PEAK and d <= FLAKE_OK_UM)
        rows.append(dict(flake_id=f["id"], old_x_um=round(s[0], 2), old_y_um=round(s[1], 2), new_x_um=round(ns[0], 2),
                         new_y_um=round(ns[1], 2), found_offset_um=round(d, 2) if d == d else "",
                         peak=round(m[1], 3) if m else "", confirmed=ok))
    return rows


# ---- moving every stage coordinate of a copied piece folder
class Move:
    """p -> R p + t for 10x positions; R (p - o) + t + o for positions of another objective (offset o)."""

    def __init__(self, R, t, o100=OFFSET_100X, o_ir=None):
        self.R, self.t = np.asarray(R, float), np.asarray(t, float)
        self.o100 = np.asarray(o100, float)
        self.o_ir = np.asarray(o_ir if o_ir is not None else o100, float)

    def pt(self, x, y, o=None):
        p = np.array([float(x), float(y)])
        if o is not None:
            p = p - o
        q = self.R @ p + self.t + (o if o is not None else 0.0)
        return float(q[0]), float(q[1])

    def flat(self, xs, o=None, nd=2):
        """A flat [x, y, x, y, ...] list."""
        out = []
        for i in range(0, len(xs) - 1, 2):
            out += [round(v, nd) for v in self.pt(xs[i], xs[i + 1], o)]
        return out

    def mat(self, M):
        return (self.R @ np.asarray(M, float)).tolist()

    def vec(self, v):
        return (self.R @ np.asarray(v, float)).tolist()

    def pos(self, get, put):
        """A row's correction from its 100x picture (flakeid.pos100: pos_dx_um, pos_dy_um), turned with the chip: the
        position moved is R (10x position + correction) + t, so the correction becomes R correction."""
        a, b = _num(get("pos_dx_um")), _num(get("pos_dy_um"))
        if a is not None and b is not None:
            v = self.vec([a, b])
            put(round(v[0], 2), round(v[1], 2))

    def for_flake(self, fid):
        """The move of flake fid's coordinates (a rigid move: the same for all)."""
        return self

    def for_pic(self, stem):
        """The move of the coordinates of 100x / IR picture `stem` (its flake's)."""
        return self


class FlakeMoves(Move):
    """A piece's move with each visited flake's own (remeasure visits, 2026-10-30): R, t (the rigid move fitted to the
    visits) for what belongs to no one flake (the tile map, the chip file, the scan positions) and for the flakes not
    visited; R, t_f for each flake found again (t_f: from where it was found).  pics: {picture stem: flake id} (the
    viewer's H100: which flake each 100x / IR picture is of)."""

    def __init__(self, R, t, o100=OFFSET_100X, o_ir=None, flakes=None, pics=None):
        super().__init__(R, t, o100, o_ir)
        self.flakes = {int(k): Move(R, v, self.o100, self.o_ir) for k, v in (flakes or {}).items()}
        self.pics = dict(pics or {})

    def for_flake(self, fid):
        try:
            return self.flakes.get(int(float(fid)), self)
        except (TypeError, ValueError):
            return self

    def for_pic(self, stem):
        fid = self.pics.get(stem)
        return self if fid is None else self.for_flake(fid)


def _row_move(mv, r):
    """The move of one table row: its flake's (flake_id), else its picture's (main_file, file), else the piece's."""
    if r.get("flake_id") not in (None, ""):
        return mv.for_flake(r["flake_id"])
    for k in ("main_file", "file"):
        if r.get(k):
            return mv.for_pic(Path(r[k]).stem)
    return mv


def _num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _fmt(x, like):
    """x written with as many decimals as the old value had (at least 2)."""
    s = str(like)
    nd = max(2, len(s.split(".")[1])) if "." in s else 2
    return f"{x:.{min(nd, 6)}f}"


def move_csv(path, mv: Move, o=None, poly_cols=(), mat_cols=()):
    """stage_x_um / stage_y_um of every row; 'x y;x y' polygon columns; 'a;b;c;d' camera-matrix columns.  Returns rows."""
    p = Path(path)
    raw = p.read_bytes().decode("utf-8-sig")
    nl = "\r\n" if "\r\n" in raw else "\n"
    rd = csv.DictReader(io.StringIO(raw))
    fields, rows = rd.fieldnames or [], list(rd)
    n = 0
    for r in rows:
        m = _row_move(mv, r)
        x, y = _num(r.get("stage_x_um")), _num(r.get("stage_y_um"))
        if x is not None and y is not None:
            nx, ny = m.pt(x, y, o)
            r["stage_x_um"], r["stage_y_um"] = _fmt(nx, r["stage_x_um"]), _fmt(ny, r["stage_y_um"])
            m.pos(r.get, lambda a, b: r.update(pos_dx_um=f"{a:.2f}", pos_dy_um=f"{b:.2f}"))
            n += 1
        for c in poly_cols:
            if r.get(c):
                pts = []
                for q in r[c].split(";"):
                    a = q.split()
                    if len(a) == 2 and _num(a[0]) is not None and _num(a[1]) is not None:
                        nx, ny = m.pt(float(a[0]), float(a[1]), o)
                        pts.append(f"{nx:.1f} {ny:.1f}")
                r[c] = ";".join(pts)
        for c in mat_cols:
            v = [_num(a) for a in (r.get(c) or "").split(";")]
            if len(v) == 4 and all(a is not None for a in v):
                r[c] = ";".join(f"{a:.6g}" for a in np.ravel(m.mat(np.reshape(v, (2, 2)))))
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=fields, lineterminator=nl)
    w.writeheader()
    w.writerows(rows)
    with open(p, "w", encoding="utf-8", newline="") as fh:      # (Path.write_text has no newline= before Python 3.10)
        fh.write(out.getvalue())
    return n


def move_pos(path, mv: Move, o=None):
    """A Micro-Manager position list (.pos): the XY stage's Position_um of each position."""
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    n = [0]

    def walk(x, m):
        if isinstance(x, dict):
            pr = (x.get("Properties") or {}).get("scalar")
            if isinstance(pr, dict) and pr.get("flake_id"):     # (a target: its flake's move for all inside)
                m = mv.for_flake((pr["flake_id"] or {}).get("scalar"))
            pu = x.get("Position_um")
            dev = (x.get("Device") or {}).get("scalar", "")
            a = pu.get("array") if isinstance(pu, dict) else None
            if "XY" in str(dev).upper() and isinstance(a, list) and len(a) == 2 and all(_num(v) is not None for v in a):
                pu["array"] = [round(v, 4) for v in m.pt(a[0], a[1], o)]
                n[0] += 1
            if isinstance(pr, dict) and "pos_dx_um" in pr:
                m.pos(lambda k: (pr.get(k) or {}).get("scalar"),
                      lambda a, b: pr.update(pos_dx_um=dict(type="STRING", scalar=f"{a:.2f}"), pos_dy_um=dict(type="STRING", scalar=f"{b:.2f}")))
            for v in x.values():
                walk(v, m)
        elif isinstance(x, list):
            for v in x:
                walk(v, m)
    walk(d, mv)
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return n[0]


def move_chip_json(path, mv: Move, rot_deg):
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))

    def xy(a):
        return [round(v, 2) for v in mv.pt(a[0], a[1])] + list(a[2:])
    for k in ("start",):
        if isinstance(d.get(k), list) and len(d[k]) >= 2:
            d[k] = xy(d[k])
    for k in ("corners", "edge_points", "scan_polygon", "focus_points"):
        if isinstance(d.get(k), list):
            d[k] = [xy(a) if isinstance(a, list) and len(a) >= 2 else a for a in d[k]]
    for k in ("edges", "sides"):
        for e in d.get(k) or []:
            if isinstance(e, dict):
                if isinstance(e.get("point"), list):
                    e["point"] = xy(e["point"])
                if isinstance(e.get("normal"), list):
                    e["normal"] = [round(v, 6) for v in mv.vec(e["normal"])]
    fs = d.get("focus_surface")
    if isinstance(fs, dict):
        if isinstance(fs.get("centre"), list):
            fs["centre"] = xy(fs["centre"])
        if isinstance(fs.get("points"), list):
            fs["points"] = [xy(a) if isinstance(a, list) and len(a) >= 2 else a for a in fs["points"]]
    fp = d.get("focus_plane")
    if isinstance(fp, list) and len(fp) == 3:        # z = a x + b y + c, in the new stage frame
        ab = mv.R @ np.array(fp[:2], float)
        d["focus_plane"] = [float(ab[0]), float(ab[1]), float(fp[2] - ab @ mv.t)]
    if isinstance(d.get("M"), list):
        d["M"] = mv.mat(d["M"])
    if _num(d.get("rotation_deg")) is not None:
        d["rotation_deg"] = round(float(d["rotation_deg"]) + rot_deg, 4)
    d["remeasured"] = dict(R=mv.R.tolist(), t=mv.t.tolist(), note="stage coordinates moved by the chip's measured move; "
                                                                   "the focus values are the old ones")
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")


def move_live_json(path, mv: Move):
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    for k in ("flakes", "terraces"):
        for r in d.get(k) or []:
            if _num(r.get("stage_x_um")) is not None and _num(r.get("stage_y_um")) is not None:
                m = mv.for_flake(r.get("flake_id"))
                r["stage_x_um"], r["stage_y_um"] = m.pt(r["stage_x_um"], r["stage_y_um"])
                m.pos(r.get, lambda a, b, r=r: r.update(pos_dx_um=a, pos_dy_um=b))
    sm = (d.get("metadata") or {}).get("stage_map")
    if isinstance(sm, dict) and isinstance(sm.get("P"), list):
        sm["P"] = [list(mv.pt(*a[:2])) for a in sm["P"]]
        if isinstance(sm.get("M"), list):
            sm["M"] = mv.mat(sm["M"])
    p.write_text(json.dumps(d), encoding="utf-8")


def move_viewer(vdir, mv: Move, note):
    D, _ = read_viewer(vdir)
    T = D.get("terraces")
    for f, fid in ([(f, f.get("id")) for f in D.get("flakes", [])]
                   + [(e, e.get("fl")) for e in (T.values() if isinstance(T, dict) else T or [])]):
        m = mv.for_flake(fid)
        if f.get("sx") is not None and f.get("sy") is not None:
            f["sx"], f["sy"] = [round(v, 1) for v in m.pt(f["sx"], f["sy"])]
        if isinstance(f.get("pos"), list) and len(f["pos"]) >= 2:          # (flakeid.pos100's correction, turned)
            f["pos"] = [round(v, 2) for v in m.vec(f["pos"][:2])] + list(f["pos"][2:])
    sm = D.get("stage_map")
    if isinstance(sm, dict) and sm.get("P"):
        sm["P"] = [list(mv.pt(*a[:2])) for a in sm["P"]]
        sm["M"] = mv.mat(sm["M"])
    if D.get("stage_origin"):
        D["stage_origin"] = list(mv.pt(*D["stage_origin"][:2]))
    D["remeasured"] = note
    write_viewer_data(vdir, D)


def move_regions(path, mv: Move):
    p = Path(path)
    mv = mv.for_pic(p.name[:-len("_regions.json")])
    d = json.loads(p.read_text(encoding="utf-8"))
    o = np.asarray(d.get("offset_100x_um") or mv.o100, float)
    if isinstance(d.get("stage_um"), list) and len(d["stage_um"]) == 2:
        d["stage_um"] = list(mv.pt(*d["stage_um"], o))
    for k in ("terrace_stage_um", "debris_stage_um", "steps_stage_um"):
        if isinstance(d.get(k), list):
            d[k] = [mv.flat(q) for q in d[k]]
    for f in d.get("fine_stage_um") or []:
        if isinstance(f, dict) and isinstance(f.get("poly"), list):
            # (a fine area in several pieces: a list of flat polygons; 72 of 590 files of 20261027_run_13 were not moved)
            f["poly"] = [mv.flat(q) for q in f["poly"]] if f["poly"] and isinstance(f["poly"][0], list) else mv.flat(f["poly"])
    p.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")


def move_mosaic_json(path, mv: Move, o):
    p = Path(path)
    mv = mv.for_pic(p.name[:-len("_mosaic.json")])
    d = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(d.get("stage_um"), list):
        d["stage_um"] = [list(mv.pt(a[0], a[1], o)) for a in d["stage_um"]]
    if isinstance(d.get("um_per_px"), list):
        d["um_per_px"] = mv.mat(d["um_per_px"])
    p.write_text(json.dumps(d), encoding="utf-8")


def _nums(v, n=2):
    return isinstance(v, (list, tuple)) and len(v) == n and all(_num(a) is not None for a in v)


def move_afm_centers(path, mv: Move):
    """The designer's AFM centres (<piece>/100x/afm_centers.json, keyed by the 100x picture's file name): each centre is an
    offset from its picture's centre, as a fraction of the picture and in um.  The run is changed in place, so the
    designer still cuts each flake out of its old 100x picture and places it on the old 10x chip map (the chip's own
    coordinates, which the AFM lists use): the offsets stay right as they are and get no turn.  A top-level rot_deg
    (written by earlier versions of this remeasure, which the designer would turn the offsets by) is removed.  A stage
    position, if any, is moved like the others."""
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))

    def walk(x, m):
        if isinstance(x, dict):
            for k, v in list(x.items()):
                if _nums(v) and "stage" in k.lower() and "frac" not in k.lower():
                    x[k] = [round(a, 3) for a in m.pt(*v)]
                else:
                    walk(v, m.for_pic(re.sub(r"\.(tiff?|jpe?g|png)$", "", k, flags=re.I)) if m is mv else m)     # (a picture's entry: its flake's move)
        elif isinstance(x, list):
            for v in x:
                walk(v, m)
    walk(d, mv)
    if isinstance(d, dict):
        d.pop("rot_deg", None)
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")


def ir_offset(piece_dir, o100):
    """The IR pictures' offset from 10x: the 100x offset plus the median (IR - 100x) position of the same labels."""
    def pos(f):
        out = {}
        try:
            for r in csv.DictReader(io.StringIO(Path(f).read_bytes().decode("utf-8-sig"))):
                x, y = _num(r.get("stage_x_um")), _num(r.get("stage_y_um"))
                if x is not None and y is not None and r.get("label"):
                    out.setdefault(r["label"], (x, y))
        except OSError:
            pass
        return out
    a = {}
    for f in Path(piece_dir).rglob("*_100x_index.csv"):
        a.update(pos(f))
    d = []
    for f in Path(piece_dir).rglob("*_domains_index.csv"):
        for k, v in pos(f).items():
            if k in a:
                d.append(np.subtract(v, a[k]))
    return np.asarray(o100, float) + (np.median(d, axis=0) if d else 0.0)


def _mover(f, mv, rot):
    """The function that moves the stage coordinates in file f (None: it holds none)."""
    nm = f.name
    if nm.endswith("_regions.json"):
        return lambda: move_regions(f, mv)
    if nm.endswith("_mosaic.json"):
        return lambda: move_mosaic_json(f, mv, mv.o_ir if "_IR_" in nm or f.parent.name == "domains" else mv.o100)
    if nm.endswith("_100x_index.csv"):
        return lambda: move_csv(f, mv, mv.o100)
    if nm.endswith("_100x_mosaic.csv"):
        return lambda: move_csv(f, mv, mv.o100, mat_cols=("um_per_px",))
    if nm.endswith("_domains_index.csv"):
        return lambda: move_csv(f, mv, mv.o_ir)
    if nm.endswith("_domains_mosaic.csv"):
        return lambda: move_csv(f, mv, mv.o_ir, mat_cols=("um_per_px",))
    if nm.endswith(".csv") and ("_targets" in nm or "_retake" in nm):
        return lambda: move_csv(f, mv, mv.o100, poly_cols=("terrace_poly_um",))
    if nm.endswith(".csv") and ("_live_flakes" in nm or "_live_terraces" in nm or "_live_clean" in nm):
        return lambda: move_csv(f, mv)
    if nm.endswith("_live_flakes.json"):
        return lambda: move_live_json(f, mv)
    if nm.endswith(".pos"):
        return lambda: move_pos(f, mv, mv.o100 if "_targets" in nm else None)
    if nm.endswith("_chip.json"):
        return lambda: move_chip_json(f, mv, rot)
    if nm == "afm_centers.json":
        return lambda: move_afm_centers(f, mv)
    return None


def pictures(piece):
    """{100x / IR picture stem: flake id} of a piece's viewer (const H100, hundred.attach)."""
    html = (Path(piece) / "index.html").read_text(encoding="utf-8")
    m = re.search(r"const H100=(.*?);\n", html, re.S)
    try:
        H = json.loads(m.group(1)) if m else {}
    except ValueError:
        H = {}
    return {Path(e["f"]).stem: int(fid) for fid, es in (H or {}).items() for e in es if e.get("f")}


def piece_move(piece, R, t, flakes=None):
    """The Move of a piece folder: its 100x offset (from its regions files) and its IR offset.  flakes: {flake id:
    t_f} of the flakes visited (FlakeMoves), None: one rigid move."""
    piece = Path(piece)
    o100 = OFFSET_100X
    for f in sorted((piece / "100x").glob("*_regions.json")):
        try:
            o100 = tuple(json.loads(f.read_text(encoding="utf-8")).get("offset_100x_um") or o100)
            break
        except (OSError, ValueError):
            pass
    if flakes:
        return FlakeMoves(R, t, o100, ir_offset(piece, o100), flakes=flakes, pics=pictures(piece))
    return Move(R, t, o100, ir_offset(piece, o100))


def move_bundle(path, moves):
    """The designer's bundle (<run>/designer/bundle.json, make_designer_bundle (afylab/flakeid-designer).py): each moved piece's flakes'
    stage_um (their 100x picture's position) and its chip map's chip_to_stage, so the designer shows the new
    coordinates after a page refresh, without a rebuild (its chip coordinates, which the AFM lists use, do not change).
    moves: {piece: Move}.  Returns the number of flakes moved."""
    p = Path(path)
    b = json.loads(p.read_text(encoding="utf-8"))
    n = 0
    for f in b.get("flakes") or []:
        mv = moves.get(f.get("scan"))
        if mv is not None and f.get("rank") is not None:
            mv = mv.for_flake(f["rank"])                # (its flake's own move, when it was visited)
        if mv is not None and _nums(f.get("stage_um")):
            f["stage_um"] = [round(v, 3) for v in mv.pt(*f["stage_um"], mv.o100)]
            n += 1
        if mv is not None and _nums(f.get("stage_origin_um")):
            f["stage_origin_um"] = [round(v, 3) for v in mv.pt(*f["stage_origin_um"])]
    for k, sc in (b.get("scans") or {}).items():
        mv = moves.get(k)
        if mv is None:
            continue
        A = sc.get("chip_to_stage")
        if isinstance(A, list) and len(A) == 2 and all(_nums(r, 3) for r in A):
            A = np.asarray(A, float)
            lin, off = mv.R @ A[:, :2], mv.R @ A[:, 2] + mv.t
            sc["chip_to_stage"] = np.c_[lin, off].round(9).tolist()
        if _nums(sc.get("stage_origin_um")):
            sc["stage_origin_um"] = [round(v, 3) for v in mv.pt(*sc["stage_origin_um"])]
    p.write_text(json.dumps(b, separators=(",", ":")), encoding="utf-8")
    return n


def move_piece(piece, R, t, note, backup=None, log=print, flakes=None):
    """Move every stage coordinate in a piece folder of a run's results (share layout: the viewer on top, data/) in place;
    each file changed is first copied to backup/<its path in the piece> (the old coordinates).  flakes: {flake id: t_f}
    of the flakes visited (each moved by R, t_f).  Returns the number of files changed."""
    piece = Path(piece)
    mv = piece_move(piece, R, t, flakes)
    rot = math.degrees(math.atan2(mv.R[1, 0], mv.R[0, 0]))
    jobs = [(piece / "index.html", lambda: move_viewer(piece, mv, note))] if (piece / "index.html").is_file() else []
    for f in sorted(piece.rglob("*")):
        fn = _mover(f, mv, rot) if f.is_file() else None
        if fn is not None:
            jobs.append((f, fn))
    if backup is not None:                             # (all copied before any is changed)
        for f, _ in jobs:
            b = Path(backup) / f.relative_to(piece)
            b.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, b)
    n = 0
    for f, fn in jobs:
        try:
            fn()
            n += 1
        except Exception as e:                         # (one unreadable file does not stop the rest; it is reported)
            log(f"  WARNING: {f.relative_to(piece)}: coordinates not moved ({type(e).__name__}: {e})")
    try:
        from ..zoomview import upgrade_page            # (the current viewer page: it shows the remeasure note)
        upgrade_page(piece / "index.html", backup=False)
    except Exception:
        pass
    return n


def move_run_file(src, dst, moves):
    """The old run's run file (<run>.json) with each remeasured piece's centre, polygon and mount corners moved."""
    d = json.loads(Path(src).read_text(encoding="utf-8"))
    for p in d.get("pieces") or []:
        mv = moves.get(p.get("name"))
        if mv is None:
            continue
        if isinstance(p.get("center"), list) and len(p["center"]) >= 2:
            p["center"] = list(mv.pt(*p["center"][:2])) + list(p["center"][2:])
        if isinstance(p.get("polygon"), list):
            p["polygon"] = [list(mv.pt(*q[:2])) + list(q[2:]) for q in p["polygon"]]
        for m in p.get("mounts") or []:
            if isinstance(m, dict) and isinstance(m.get("corners"), list):
                m["corners"] = [list(mv.pt(*q[:2])) + list(q[2:]) for q in m["corners"]]
    d["remeasured"] = {k: dict(R=v.R.tolist(), t=v.t.tolist()) for k, v in moves.items()}
    Path(dst).write_text(json.dumps(d, indent=1), encoding="utf-8")


# ---- the whole job
BACKUP_DIR = "old_coords"          # <run>/old_coords/<YYYYmmdd_HHMMSS>/: every file a remeasure changed, as it was


def old_pieces(run_dir):
    """The piece folders (with a viewer) of an old run's results folder."""
    return sorted(p for p in Path(run_dir).iterdir() if p.is_dir() and (p / "index.html").is_file()
                  and p.name not in (CHECK_DIR, BACKUP_DIR))


def match_pieces(old_run, new_viewers, log=print):
    """Each new piece's 10x image matched to the old run's pieces (whole-image features, then patches): dict(olds {old
    piece: its folder}, regs [(new piece, old piece, register()'s result, check_flakes() rows)]).  Nothing is changed."""
    old_run = Path(old_run)
    olds = {p.name: p for p in old_pieces(old_run)}
    log(f"remeasure: old run {old_run} ({len(olds)} pieces), new 10x scans: {', '.join(new_viewers) or 'none'}")
    oc = {}
    for k, p in olds.items():
        try:
            oc[k] = Canvas(p)
        except Exception as e:
            log(f"  old piece {k}: no 10x image to match ({e})")
    cands = []
    nc = {}
    for nk, nv in new_viewers.items():
        try:
            nc[nk] = Canvas(nv)
        except Exception as e:
            log(f"  new piece {nk}: no 10x image ({e})")
            continue
        for ok_, o in oc.items():
            c = coarse_match(o, nc[nk])
            log(f"  {nk} against old {ok_}: " + (f"{c['inliers']} of {c['matches']} features agree" if c else "no match"))
            if c:
                cands.append((c["inliers"], nk, ok_, c))
    used_new, used_old = set(), set()
    regs = []
    for _, nk, ok_, c in sorted(cands, key=lambda a: -a[0]):
        if nk in used_new or ok_ in used_old:
            continue
        used_new.add(nk)
        used_old.add(ok_)
        reg = register(oc[ok_], nc[nk], c, log=log)
        if reg is not None:
            regs.append((nk, ok_, reg, check_flakes(oc[ok_], nc[nk], reg["R"], reg["t"])))
    return dict(olds=olds, regs=regs, new_viewers=dict(new_viewers))


# ---- visiting the flakes (2026-10-30): one new 10x scan cannot be trusted flake by flake (on 20261027_run_13 whole rows
# of 10x tiles were 80-150 um off, a different set in each scan), so after the scan the microscope goes to each flake
# imaged at 100x, takes one 10x frame and finds the flake's own 100x picture in it (Scanner.remeasure_visit)
VISIT_MIN_PEAK = 0.5            # correlation the 100x picture needs in the 10x frame (as the 100x re-centring) ...
VISIT_MAX_SECOND = 0.85         # ... and the best match elsewhere at most this fraction of it
VISIT_MAX_UM = 400.0            # found farther than this from where the chip's move puts it: not believed
VISIT_EXCL_UM = 40.0            # "the best match elsewhere": at least this far from the best (the picture is ~115 um
                                # across; at half resolution the correlation peak itself is ~20 um wide: with the
                                # re-centring's 15 um, 3 of 18 simulated flakes failed on their own peak's flank)
VISIT_FIT_UM = 5.0              # the rigid move of the visits: the one most flakes fit within this (pairs of flakes
                                # tried: a chip that slipped on its slide is a group of its own, not averaged in)
NEAR_UM, NEAR_K, NEAR_AGREE_UM, NEAR_MIN_UM = 2000.0, 3, 10.0, 5.0
                                # a flake not visited: the residual of its NEAR_K nearest visited flakes within NEAR_UM,
                                # when they agree within NEAR_AGREE_UM and it is over NEAR_MIN_UM (a chip that slipped
                                # on its slide); else the rigid move


def focus100(piece):
    """{100x picture stem: (z_um, z10_um)} from a piece's 100x index (<piece>/data/<piece>_100x/<piece>_100x_index.csv on
    the share; the local layout too): the focus each picture was taken at (focus "predicted" too: it was imaged there),
    and the 10x focus there."""
    out = {}
    for f in sorted(Path(piece).rglob("*_100x_index.csv")):
        try:
            with open(f, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    try:
                        z = float(r.get("z_um") or "nan")
                    except ValueError:
                        continue
                    try:
                        z10 = float(r.get("z10_um") or "nan")
                    except ValueError:
                        z10 = float("nan")
                    if math.isfinite(z) and r.get("file"):
                        out[Path(r["file"]).stem] = (z, z10 if math.isfinite(z10) else None)
        except OSError:
            pass
    return out


def labels100(piece):
    """{100x picture stem: its target label} from a piece's 100x index (the label names the flake: "01_F012_15nm")."""
    out = {}
    for f in sorted(Path(piece).rglob("*_100x_index.csv")):
        try:
            with open(f, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r.get("file") and r.get("label"):
                        out.setdefault(Path(r["file"]).stem, r["label"])
        except OSError:
            pass
    return out


def visit_targets(piece):
    """The flakes to visit in an old piece: one per flake with a 100x picture (the viewer's H100), dict(flake_id, stem,
    picture (its JPEG), c_old (the picture's centre, 10x stage um), o100 (its 10x -> 100x offset), um_px (the
    full-resolution frame's), frame_wh, z100_old / z10_old (the focus it was taken at and the 10x focus there, from
    the 100x index; None if not known))."""
    piece = Path(piece)
    out, seen = [], set()
    zs = focus100(piece)
    for stem, fid in pictures(piece).items():
        reg, pic = piece / "100x" / f"{stem}_regions.json", piece / "100x" / f"{stem}.jpg"
        if fid in seen or not reg.is_file() or not pic.is_file():
            continue
        try:
            R = json.loads(reg.read_text(encoding="utf-8"))
            o = np.asarray(R.get("offset_100x_um") or OFFSET_100X, float)
            c = np.asarray(R["stage_um"], float) - o
        except (OSError, ValueError, KeyError, TypeError):
            continue
        z100, z10 = zs.get(stem, (None, None))
        out.append(dict(flake_id=fid, stem=stem, picture=str(pic), c_old=c, o100=o, um_px=float(R.get("px_um") or 1.09 / RATIO_100X),
                        frame_wh=list(R.get("frame_wh") or (1376, 1038)), z100_old=z100, z10_old=z10))
        seen.add(fid)
    return out


def locate(frame_grey, M, picture_bgr, pic_um_px, R=None, frac=1.0, excl_um=VISIT_EXCL_UM, edges=True):
    """Where a flake's old 100x picture is in a 10x frame.  frame_grey: the frame (grey, any scale), M: um per frame pixel
    at that scale (stage = frame's stage + M (p - centre)); pic_um_px: the picture's um per pixel; R: the chip's rotation
    since (the picture turned with it).  frac: the square taken from the picture's middle, as a fraction of its height
    (a 100x frame: 0.5, so the picture's middle can be found anywhere in a frame of the same size).  Returns
    dict(offset (stage um of the picture's centre from the frame's), peak, second) or None (no structure)."""
    import cv2
    from .retarget import match_crop
    M = np.asarray(M, float)
    um = math.sqrt(abs(np.linalg.det(M)))
    g = picture_bgr.astype(np.float32).mean(axis=2) / 255.0
    k = pic_um_px / um
    small = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    # a square from its middle, turned as the chip (frame pixels: M^-1 R M), then the largest square inside that is
    # all picture
    h, w = small.shape
    sq = max(16, int(min(h, w) * min(float(frac), 1.0)))
    small = small[(h - sq) // 2:(h - sq) // 2 + sq, (w - sq) // 2:(w - sq) // 2 + sq]
    if R is not None:
        A = np.linalg.inv(M) @ np.asarray(R, float) @ M
        c = np.array([(sq - 1) / 2.0, (sq - 1) / 2.0])    # (the centre: OpenCV's pixel centres are whole numbers; turning
        small = cv2.warpAffine(small, np.hstack([A, (c - A @ c)[:, None]]).astype(np.float32), (sq, sq), flags=cv2.INTER_LINEAR)
        inner = int(sq / max(abs(A[0, 0]) + abs(A[0, 1]), abs(A[1, 0]) + abs(A[1, 1]), 1.0))
        inner -= (sq - inner) % 2                       # about sq / 2 moved the picture by up to 0.7 px at 90 degrees)
        o = (sq - inner) // 2                           # (an even margin: the centre stays where it was)
        small = small[o:o + inner, o:o + inner]
    m = match_crop(small, np.asarray(frame_grey, np.float32), um_px=um, excl_um=excl_um, edges=edges)
    if m is None:
        return None
    return dict(offset=(M @ (-np.asarray(m["d"], float))).tolist(), peak=float(m["peak"]), second=float(m["second"]))


def _u8(g):
    """Grey (any range) to 0..255, stretched between its 1st and 99th percentiles."""
    g = np.nan_to_num(np.asarray(g, np.float32))
    lo, hi = np.percentile(g, (1, 99))
    return (np.clip((g - lo) / max(float(hi - lo), 1e-6), 0, 1) * 255).astype(np.uint8)


FEAT_MIN_INLIERS = 15           # a features match: at least this many keypoints agree on one move ...
FEAT_MAX_SCALE = 0.08           # ... at the same scale (within 8 %) ...
FEAT_MAX_TURN_DEG = 4.0         # ... and turned as the chip (within 4 deg)


def locate_feat(frame_grey, M, picture_bgr, pic_um_px, R=None, min_inl=FEAT_MIN_INLIERS):
    """Where a flake's old 100x picture is in a frame (or a mosaic), by keypoints (AKAZE) matched between the two and
    one move fitted to them (RANSAC): independent of the colours and the brightness, and the picture need not be all
    in the frame (half of it is enough).  Same arguments and result as locate() (peak: the share of the matches that
    agree; inliers: how many), or None."""
    import cv2
    M = np.asarray(M, float)
    um = math.sqrt(abs(np.linalg.det(M)))
    g = picture_bgr.astype(np.float32).mean(axis=2)
    k = pic_um_px / um
    small = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    turn = 0.0
    if R is not None:
        A = np.linalg.inv(M) @ np.asarray(R, float) @ M
        turn = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    a, b = _u8(small), _u8(frame_grey)
    det = cv2.AKAZE_create()
    k1, d1 = det.detectAndCompute(a, None)
    k2, d2 = det.detectAndCompute(b, None)
    if d1 is None or d2 is None or len(k1) < min_inl or len(k2) < min_inl:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    if len(good) < min_inl:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=4000)
    if A is None or inl is None:
        return None
    n = int(inl.sum())
    sc = math.hypot(A[0, 0], A[1, 0])
    ang = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    if n < min_inl or abs(sc - 1.0) > FEAT_MAX_SCALE or abs(((ang - turn + 180) % 360) - 180) > FEAT_MAX_TURN_DEG:
        return None
    c = A @ np.array([small.shape[1] / 2.0, small.shape[0] / 2.0, 1.0])      # (the picture's centre in the frame)
    d = c - np.array([b.shape[1] / 2.0, b.shape[0] / 2.0])
    return dict(offset=(M @ d).tolist(), peak=n / float(len(good)), second=0.0, inliers=n)


MASKED_UM_PX = 0.5              # the masked match (locate_masked): both pictures at this many um per pixel
MASKED_MIN_OVERLAP = 0.25       # ... the old picture at least this much (of its area) in the frame
MASKED_MIN_PEAK = 0.65          # ... recognised: a correlation of at least this ...
MASKED_MAX_SECOND = 0.75        # ... and the best one elsewhere (EXCL100_UM away) at most this fraction of it


def _edges_colour(img, k):
    """A picture ((h, w, 3) colour, any channel order, or grey) as its edges for locate_masked(): shrunk by k, the log of
    each channel (a gain per channel, i.e. another white balance or exposure, only adds a constant), its gradient
    magnitude summed over the channels, smoothed.  A flake edge shows in whichever channel has the contrast (a blue
    flake on a red substrate can have almost none in grey)."""
    import cv2
    a = np.asarray(img, np.float32)
    if a.ndim == 2:
        a = a[..., None]
    a = cv2.resize(a, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    if a.ndim == 2:
        a = a[..., None]
    L = cv2.GaussianBlur(np.log(np.maximum(a, 1.0)), (0, 0), 1.0)
    if L.ndim == 2:
        L = L[..., None]
    e = np.zeros(L.shape[:2], np.float32)
    for c in range(L.shape[2]):
        gx = cv2.Sobel(L[..., c], cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(L[..., c], cv2.CV_32F, 0, 1)
        e += gx * gx + gy * gy
    return cv2.GaussianBlur(np.sqrt(e), (0, 0), 1.5)


def masked_ncc(A, B, ma=None, mb=None, min_overlap=MASKED_MIN_OVERLAP):
    """Normalized correlation of B (moving) placed everywhere on A (fixed), over the pixels both hold (masks ma, mb:
    the pixels that count) only: B may lie partly outside A (Padfield, "Masked object registration in the Fourier
    domain", IEEE TIP 2012; scikit-image's phase_cross_correlation with masks is the same).  Returns (r, n): r[y, x] for
    B's top-left pixel at A's (x - wb + 1, y - hb + 1), -1 where fewer than min_overlap of B's pixels overlap; n: how
    many overlap."""
    import cv2
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    ha, wa = A.shape
    hb, wb = B.shape
    ma = np.ones((ha, wa)) if ma is None else np.asarray(ma, np.float64)
    mb = np.ones((hb, wb)) if mb is None else np.asarray(mb, np.float64)
    A, B = A * ma, B * mb
    sh = (cv2.getOptimalDFTSize(ha + hb - 1), cv2.getOptimalDFTSize(wa + wb - 1))
    fb_m, fb_b, fb_bb = (np.fft.rfft2(x[::-1, ::-1], sh) for x in (mb, B, B * B))
    fa_m, fa_a, fa_aa = (np.fft.rfft2(x, sh) for x in (ma, A, A * A))

    def xc(fa, fb):
        return np.fft.irfft2(fa * fb, sh)[:ha + hb - 1, :wa + wb - 1]
    n = np.round(xc(fa_m, fb_m))
    nc = np.maximum(n, 1.0)
    sa, sb = xc(fa_a, fb_m), xc(fa_m, fb_b)
    num = xc(fa_a, fb_b) - sa * sb / nc
    va = xc(fa_aa, fb_m) - sa * sa / nc
    vb = xc(fa_m, fb_bb) - sb * sb / nc
    r = num / np.sqrt(np.maximum(va * vb, 1e-12))
    r[(n < min_overlap * mb.sum()) | (va <= 1e-9 * max(float(va.max()), 1e-12)) | (vb <= 1e-9 * max(float(vb.max()), 1e-12))] = -1.0
    return r, n


def locate_masked(frame, M, picture_bgr, pic_um_px, R=None, frame_mask=None, excl_um=None,
                  min_overlap=MASKED_MIN_OVERLAP):
    """Where a flake's old 100x picture is relative to a frame (or a mosaic), when only part of it is in the frame (at
    least min_overlap of it): the edges of both in colour (_edges_colour: independent of the white balance and the
    exposure), the masked normalized correlation over the part they share (masked_ncc).  frame: (h, w, 3) or grey,
    M: um per frame pixel; frame_mask: the frame's pixels that hold a picture (a mosaic's holes left out).  Same
    result as locate() (+ overlap: the share of the picture in the frame), or None.
    (20261027_13 F35, 2026-11-01: a large flake 53, 37 um off, half of it in the frame, a grey substrate where the
    old picture's was red: locate() 0.05, keypoints none; this 0.91, elsewhere 0.35.)"""
    import cv2
    excl_um = EXCL100_UM if excl_um is None else excl_um
    M = np.asarray(M, float)
    um = math.sqrt(abs(np.linalg.det(M)))
    kf = um / MASKED_UM_PX
    A = _edges_colour(frame, kf)
    ma = np.ones(A.shape, np.float32)
    if frame_mask is not None:
        ma = cv2.resize(np.asarray(frame_mask, np.float32), (A.shape[1], A.shape[0]), interpolation=cv2.INTER_AREA)
    # the picture at the frame's (shrunk) pixels, turned as the chip: A frame pixel = M^-1 R M picture pixel
    kp = pic_um_px / MASKED_UM_PX
    B = _edges_colour(picture_bgr, kp)
    mb = np.ones(B.shape, np.float32)
    Mn = M / um                                         # (the frame's axes: M's direction, at MASKED_UM_PX)
    if R is not None:
        T = np.linalg.inv(Mn) @ np.asarray(R, float) @ Mn
        hb, wb = B.shape
        c = np.array([(wb - 1) / 2.0, (hb - 1) / 2.0])
        corners = np.array([[0, 0], [wb, 0], [0, hb], [wb, hb]], float) - c
        q = corners @ T.T
        size = np.ceil(q.max(0) - q.min(0)).astype(int) + 1
        cn = (size - 1) / 2.0
        Aff = np.hstack([T, (cn - T @ c)[:, None]]).astype(np.float32)
        B = cv2.warpAffine(B, Aff, (int(size[0]), int(size[1])), flags=cv2.INTER_LINEAR)
        mb = cv2.warpAffine(mb, Aff, (int(size[0]), int(size[1])), flags=cv2.INTER_NEAREST)
    # the edges' own borders (the frame's and picture's ends, a hole) are not edges of the sample
    ma = cv2.erode((ma > 0.5).astype(np.uint8), np.ones((7, 7), np.uint8))
    mb = cv2.erode((mb > 0.5).astype(np.uint8), np.ones((7, 7), np.uint8))
    if ma.sum() < 100 or mb.sum() < 100:
        return None
    r, n = masked_ncc(A, B, ma, mb, min_overlap)
    iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
    pk = float(r[iy, ix])
    if pk <= -1.0:
        return None
    sub = [0.0, 0.0]
    for ax, (a_, c_) in enumerate(((r[iy, ix - 1], r[iy, ix + 1]) if 0 < ix < r.shape[1] - 1 else (None, None),
                                   (r[iy - 1, ix], r[iy + 1, ix]) if 0 < iy < r.shape[0] - 1 else (None, None))):
        if a_ is not None and a_ > -1 and c_ > -1:
            den = a_ - 2 * pk + c_
            sub[ax] = float(np.clip(0.5 * (a_ - c_) / den, -0.5, 0.5)) if den < 0 else 0.0
    r2 = r.astype(np.float32).copy()
    cv2.circle(r2, (int(ix), int(iy)), max(3, int(round(excl_um / MASKED_UM_PX))), -1.0, -1)
    hb, wb = B.shape
    ha, wa = A.shape
    # the picture's centre (its turned copy's centre) in shrunk frame pixels, from the frame's centre
    d = np.array([ix + sub[0] - (wb - 1) + wb / 2.0 - wa / 2.0, iy + sub[1] - (hb - 1) + hb / 2.0 - ha / 2.0])
    return dict(offset=(Mn @ d * MASKED_UM_PX).tolist(), peak=pk, second=float(r2.max()),
                overlap=float(n[iy, ix] / max(float(mb.sum()), 1.0)))


def stitch(frames, M, centre, mask=False):
    """Frames [(stage position, grey or (h, w, C) colour)] of one scale pasted by their stage positions into one picture,
    the overlaps blended (each frame weighted by a window that falls to its edges).  M: um per frame pixel; centre: the
    stage position of the picture's centre (shape / 2, as locate() counts it).  A flake between two frames of a mosaic
    is whole in the picture, where neither frame alone holds enough of it.  mask: also the part that holds a frame
    (a frame left out, e.g. at the stage's limit, leaves a hole: filled with the mean)."""
    M = np.asarray(M, float)
    Mi = np.linalg.inv(M)
    h, w = frames[0][1].shape[:2]
    extra = tuple(np.asarray(frames[0][1]).shape[2:])
    d = np.array([Mi @ (np.asarray(p, float) - np.asarray(centre, float)) for p, _ in frames])
    ex, ey = np.ceil(np.abs(d).max(axis=0)).astype(int)
    W, H = w + 2 * ex, h + 2 * ey
    acc, wt = np.zeros((H, W) + extra, np.float32), np.zeros((H, W), np.float32)
    win = (np.outer(np.hanning(h + 2)[1:-1], np.hanning(w + 2)[1:-1]) + 1e-3).astype(np.float32)
    wx = win[..., None] if extra else win
    for (_, g), dd in zip(frames, d):
        x0, y0 = ex + int(round(dd[0])), ey + int(round(dd[1]))
        acc[y0:y0 + h, x0:x0 + w] += wx * np.asarray(g, np.float32)
        wt[y0:y0 + h, x0:x0 + w] += win
    out = acc / np.maximum(wt, 1e-6)[(...,) + ((None,) if extra else ())]
    out[wt == 0] = out[wt > 0].mean(axis=0)
    return (out, wt > 0) if mask else out


def fit_visits(visits, log=print):
    """The rigid move fitted to the flakes found again (c_old -> c_new), the outliers left out (a wrong match, or a chip
    that slipped on its slide: those flakes still get where they were found).  Returns dict(R, t, rms_um, n_fit,
    outliers, flakes {flake id: t_f, each found flake's own move R c_old + t_f = c_new}) or None (fewer than 3 found)."""
    found = [v for v in visits if v.get("c_new") is not None]
    if len(found) < 3:
        return None
    src = np.array([v["c_old"] for v in found], float)
    dst = np.array([v["c_new"] for v in found], float)
    # the move most flakes agree with: from each pair of flakes (at most 3000 pairs), the most within VISIT_FIT_UM
    n = len(src)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > 3000:
        rng = np.random.default_rng(0)
        pairs = [pairs[k] for k in rng.choice(len(pairs), 3000, replace=False)]
    best = None
    for i, j in pairs:
        if np.hypot(*(src[i] - src[j])) < 200.0:        # (too close together for the turn)
            continue
        R_, t_, _ = rigid_fit(src[[i, j]], dst[[i, j]])
        k_ = np.hypot(*(dst - (src @ R_.T + t_)).T) <= VISIT_FIT_UM
        if best is None or k_.sum() > best.sum():
            best = k_
    keep = best if best is not None and best.sum() >= 3 else np.ones(n, bool)
    R, t, rms = rigid_fit(src[keep], dst[keep])
    for _ in range(3):
        nk = np.hypot(*(dst - (src @ R.T + t)).T) <= VISIT_FIT_UM
        if nk.sum() < 3 or (nk == keep).all():
            break
        keep = nk
        R, t, rms = rigid_fit(src[keep], dst[keep])
    res = np.hypot(*(dst - (src @ R.T + t)).T)
    flakes = {int(v["flake_id"]): (np.asarray(v["c_new"], float) - R @ np.asarray(v["c_old"], float)) for v in found}
    for v, r_, k_ in zip(found, res, keep):
        v["residual_um"], v["fit"] = round(float(r_), 2), bool(k_)
    return dict(R=R, t=t, rms_um=float(rms), n_fit=int(keep.sum()), n_found=len(found),
                outliers=[int(v["flake_id"]) for v, k_ in zip(found, keep) if not k_], flakes=flakes)


def near_moves(piece, R, t, flakes):
    """The flakes not visited that take their visited neighbours' residual (NEAR_*): {flake id: t_f}."""
    if not flakes:
        return {}
    D, _ = read_viewer(piece)
    vis = [(fid, tf) for fid, tf in flakes.items()]
    # each visited flake's place (10x stage um, old) and its residual against the rigid move
    pos = {f["id"]: np.array(pos10(f), float) for f in D.get("flakes", []) if f.get("sx") is not None}
    P = np.array([pos[f] for f, _ in vis if f in pos])
    Q = np.array([np.asarray(tf, float) - np.asarray(t, float) for f, tf in vis if f in pos])
    if len(P) < NEAR_K:
        return {}
    out = {}
    for fid, p in pos.items():
        if fid in flakes:
            continue
        d = np.hypot(*(P - p).T)
        k = np.argsort(d)[:NEAR_K]
        k = k[d[k] <= NEAR_UM]
        if len(k) < NEAR_K:
            continue
        med = np.median(Q[k], axis=0)
        if np.hypot(*(Q[k] - med).T).max() <= NEAR_AGREE_UM and math.hypot(*med) > NEAR_MIN_UM:
            out[fid] = np.asarray(t, float) + med
    return out


def write_visits(path, visits):
    cols = ["flake_id", "picture", "old_x_um", "old_y_um", "predicted_x_um", "predicted_y_um", "found_x_um", "found_y_um",
            "peak", "second", "residual_um", "used"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for v in visits:
            cn, q = v.get("c_new"), v.get("predicted")
            w.writerow(dict(flake_id=v["flake_id"], picture=v["stem"], old_x_um=round(v["c_old"][0], 2), old_y_um=round(v["c_old"][1], 2),
                            predicted_x_um=round(q[0], 2) if q is not None else "", predicted_y_um=round(q[1], 2) if q is not None else "",
                            found_x_um=round(cn[0], 2) if cn is not None else "", found_y_um=round(cn[1], 2) if cn is not None else "",
                            peak=v.get("peak", ""), second=v.get("second", ""), residual_um=v.get("residual_um", ""),
                            used=("fits the chip's move" if v.get("fit") else "its own place (off the chip's move)")
                            if cn is not None else (v.get("why") or "not found")))


# ---- remeasure by passes that must agree (Avram, 2026-10-30; Scanner.run_remeasure): the 10x and 100x positions of a
# run are only as good as the stage, which slips in Y now and then (20261029_run_1: two rows of 10x tiles 85 um short,
# creeping 40 um along one row, while it reported where it was sent).  So: the flakes to remeasure are chosen (all
# imaged at 100x, the kept ones, or a list: the designer's "Export flakes (JSON)"); the operator centres two spots of
# the old 10x image, at both ends of what was on the stage, at 10x (the chips' rough move); each flake is found at 10x
# (one frame, its own 100x picture in it); then at 100x every flake is found again and its position noted (pass A1),
# and again (A2), ...  until two passes in a row agree: at least AGREE_FRAC of the flakes within AGREE_UM, after taking
# off the shift they share (a slip of the stage between two passes moves every flake alike; one during a pass does
# not).  The last pass gives the coordinates of the flakes that agree (the rest by the chip's move fitted to them, as
# remeasure()).  (2026-11-02: was every flake within 3 um.)
AGREE_UM = 5.0                  # two passes agree: at least AGREE_FRAC of the flakes within this of the other pass ...
AGREE_FRAC = 0.5                # ... (their common shift off); settings remeasure_agree_um / remeasure_agree_frac
MAX_PASSES = 8                  # passes at 100x at most (then the remeasure failed: report_failed)
CAL_END = 0.2                   # a calibration spot: within this fraction of the stage's extent from one end
CAL_WIN_UM = 700.0              # ... the area whose structure scores it (about half a 10x field)
CAL_PIC_UM = 3000.0             # ... the picture shown: this wide, the 10x field boxed, a cross at the spot
CAL_SEP_UM = 500.0              # "Skip": the next spot at least this far from the skipped ones
FIELD100_FRAC = 0.5             # at 100x the middle half of the old picture is looked for in the new frame
EXCL100_UM = 12.0               # ... the best match elsewhere: at least this far from the best
FLAKE_RE = re.compile(r"(?:^|[:/\\])([^:/\\]+?)[-_][fF]0*(\d+)(?=$|[^0-9])")


def parse_flake_list(obj):
    """{(piece or None, flake number)} from a flake list: the designer's export ({"flakes": [ids]}: "run:piece-f0012"),
    a list of such ids, of labels ("piece_F012_6nm"), of dicts {piece, flake} or of plain numbers (any piece)."""
    if isinstance(obj, dict):
        obj = obj.get("flakes") or obj.get("ids") or []
    out = set()
    for it in obj or []:
        if isinstance(it, dict):
            pc, n = it.get("piece") or it.get("scan"), it.get("flake", it.get("flake_number", it.get("flake_id")))
            if pc is not None and n is None:
                q = FLAKE_RE.search(str(pc))
                pc, n = (q.group(1), q.group(2)) if q else (pc, None)
            if n is None and it.get("id"):
                q = FLAKE_RE.search(str(it["id"]))
                pc, n = (q.group(1), q.group(2)) if q else (pc, None)
            try:
                out.add((str(pc).split(":")[-1] if pc else None, int(float(n))))
            except (TypeError, ValueError):
                pass
            continue
        if isinstance(it, (int, float)):
            out.add((None, int(it)))
            continue
        q = FLAKE_RE.search(str(it))
        if q:
            out.add((q.group(1), int(q.group(2))))
        elif str(it).strip().lstrip("Ff#").isdigit():
            out.add((None, int(str(it).strip().lstrip("Ff#"))))
    return out


def kept_flakes(old_run):
    """{(piece, flake number)} kept in the screening: the designer's <run>/kept/kept_flakes.csv."""
    f = Path(old_run) / "kept" / "kept_flakes.csv"
    if not f.is_file():
        raise FileNotFoundError(f"{f} not found: no kept flakes (keep flakes in the designer's screening, or export them "
                                "as a list and choose Custom)")
    out = set()
    with open(f, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            q = FLAKE_RE.search(str(r.get("flake") or ""))
            pc, n = (r.get("piece") or (q.group(1) if q else None)), (r.get("flake_number") or (q.group(2) if q else None))
            try:
                out.add((pc, int(float(n))))
            except (TypeError, ValueError):
                pass
    return out


def imaging_order(piece, v):
    """Sort key of a visit: its piece, then the 100x picture's number (<stem>_100x_<n>: the order they were taken)."""
    m = re.search(r"_100x_(\d+)", v.get("stem") or "")
    return (str(piece), int(m.group(1)) if m else 10 ** 9, int(v["flake_id"]))


def random_visits(out, n, seed=None):
    """n of the visits {piece: [visits]}: the first and the last imaged at 100x (imaging_order, over all pieces) and
    n - 2 more at random (Avram, 2026-11-02: "if they put 10, do first, last and 8 random"); all if there are no more
    than n.  Same structure back, each piece's visits in their original order."""
    import random
    allv = sorted(((p, v) for p, vs in out.items() for v in vs), key=lambda pv: imaging_order(*pv))
    n = max(1, int(n))
    if len(allv) <= n:
        return out
    pick = {0, len(allv) - 1} if n >= 2 else {0}
    rest = [i for i in range(1, len(allv) - 1)]
    pick |= set(random.Random(seed).sample(rest, max(0, n - len(pick))))
    keep = {id(allv[i][1]) for i in pick}
    return {p: [v for v in vs if id(v) in keep] for p, vs in out.items() if any(id(v) in keep for v in vs)}


def select_visits(old_run, which="all", custom=None, log=print, n_random=10, seed=None):
    """The flakes to remeasure: {old piece folder: [visit_targets() entries]}.  which: "all" (every flake imaged at
    100x), "kept" (kept_flakes), "custom" (custom: a flake list file, parse_flake_list) or "random" (n_random of all
    imaged at 100x: random_visits)."""
    want = None
    if which == "kept":
        want = kept_flakes(old_run)
    elif which == "custom":
        want = parse_flake_list(json.loads(Path(custom).read_text(encoding="utf-8-sig")))
        if not want:
            raise ValueError(f"{custom}: no flakes in it (a JSON list of flake ids, or the designer's export)")
    out, hit = {}, set()
    for p in old_pieces(old_run):
        vs = visit_targets(p)
        if want is not None:
            keep = []
            for v in vs:
                for key in ((p.name, int(v["flake_id"])), (None, int(v["flake_id"]))):
                    if key in want:
                        keep.append(v)
                        hit.add(key)
                        break
            vs = keep
        if vs:
            out[p] = vs
    if which == "random":
        n_all = sum(len(v) for v in out.values())
        out = random_visits(out, n_random, seed)
        sel = sorted(((p, v) for p, vs in out.items() for v in vs), key=lambda pv: imaging_order(*pv))
        log(f"  random: {len(sel)} of the {n_all} flakes imaged at 100x (the first and last imaged, the rest at random): "
            + ", ".join(f"{Path(p).name} F{v['flake_id']}" for p, v in sel[:40]) + (" ..." if len(sel) > 40 else ""))
    n = sum(len(v) for v in out.values())
    log(f"  flakes to remeasure ({which}): {n} imaged at 100x in {len(out)} piece{'s' if len(out) != 1 else ''}"
        + (f"; {len(want - hit)} of the {len(want)} asked for have no 100x picture in this run (or another piece name): "
           + ", ".join(f"{a or '?'} F{b}" for a, b in sorted(want - hit, key=lambda k: (str(k[0]), k[1]))[:12])
           if want is not None and want - hit else ""))
    return out


def calib_spots(pieces, log=print):
    """Candidate spots to centre by hand, for the chips' rough move: [end A spots, end B spots], each best first, at the
    two ends (along the longest direction) of what was scanned (all pieces' 10x images).  A spot: dict(piece (folder),
    canvas, px (full-resolution pixel of the old 10x image), st (its old stage position), score)."""
    import cv2
    cands = []
    for p in pieces:
        try:
            cv = Canvas(p)
        except Exception as e:
            log(f"  {Path(p).name}: no 10x image for the calibration spots ({e})")
            continue
        g = cv.overview()
        ok = np.isfinite(g)
        b = cv2.GaussianBlur(np.nan_to_num(g, nan=float(np.nanmedian(g))), (0, 0), 1.0) - \
            cv2.GaussianBlur(np.nan_to_num(g, nan=float(np.nanmedian(g))), (0, 0), 6.0)
        k = max(8, int(CAL_WIN_UM / (cv.um_px * OVERVIEW_DOWN)))
        thr = 2.5 * max(float(np.median(np.abs(b))), 1e-4)    # (once: inside the loop it took hours on a large chip)
        for y in range(0, g.shape[0] - k, k // 2):
            for x in range(0, g.shape[1] - k, k // 2):
                if not ok[y:y + k, x:x + k].all():
                    continue
                w = b[y:y + k, x:x + k]
                # structure everywhere in the window (not one speck): the spread of the band-pass and how much of the
                # window has edges
                sc = float(w.std()) * float((np.abs(w) > thr).mean()) ** 0.5
                px = np.array([(x + k / 2.0) * OVERVIEW_DOWN, (y + k / 2.0) * OVERVIEW_DOWN])
                cands.append(dict(piece=Path(p), canvas=cv, px=px, st=cv.to_stage(px)[0], score=sc))
    if len(cands) < 2:
        raise ValueError("no structure in the old 10x images to centre on")
    S = np.array([c["st"] for c in cands])
    u = np.linalg.svd(S - S.mean(0), full_matrices=False)[2][0]          # the longest direction
    s_ = (S - S.mean(0)) @ u
    lo, hi = s_.min(), s_.max()
    ends = []
    for side in (0, 1):
        f = (s_ - lo) / max(hi - lo, 1e-9)
        sel = [c for c, q in zip(cands, f) if (q <= CAL_END if side == 0 else q >= 1 - CAL_END)]
        ends.append(sorted(sel, key=lambda c: -c["score"]))
    return ends


def calib_picture(spot, path):
    """The old 10x image around a calibration spot (CAL_PIC_UM wide, colour), the 10x field boxed and a cross at the
    spot, as path (JPEG)."""
    import cv2
    cv, d = spot["canvas"], 2
    n = int(CAL_PIC_UM / (cv.um_px * d))
    x0, y0 = int(spot["px"][0] / d) - n // 2, int(spot["px"][1] / d) - n * 3 // 8
    im = cv.region(x0, y0, n, n * 3 // 4, d, color=True)
    im = np.nan_to_num(im, nan=0.07)
    im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    D = cv.D.get("stage_map") or {}
    fw, fh = (D.get("wh") or (1376, 1038))
    c = np.array([n // 2, n * 3 // 8])
    hw, hh = int(fw / d / 2), int(fh / d / 2)
    cv2.rectangle(im, (int(c[0] - hw), int(c[1] - hh)), (int(c[0] + hw), int(c[1] + hh)), (0, 220, 255), 3)
    a = max(12, n // 40)
    cv2.line(im, (int(c[0] - a), int(c[1])), (int(c[0] + a), int(c[1])), (0, 0, 255), 3)
    cv2.line(im, (int(c[0]), int(c[1] - a)), (int(c[0]), int(c[1] + a)), (0, 0, 255), 3)
    cv2.imwrite(str(path), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return str(path)


def calib_whole(canvases, path, max_w=2400):
    """The whole old 10x scan (every piece's image, put in place by their stage positions in the first one's pixel
    frame, so as the 10x crops are shown), at most max_w pixels wide, as path (JPEG), for the operator to see where on
    the chips the calibration spots are.  dict(path, frame (the first canvas), x0, y0 (its pixel at the picture's
    top left), d (its pixels per picture pixel)); calib_marks gives the spots' boxes in it."""
    import cv2
    cvs = list({id(c): c for c in canvases}.values())
    f = cvs[0]
    corner = lambda c: np.array([[0, 0], [c.W, 0], [c.W, c.H], [0, c.H]], float)
    into = lambda c, px: px if c is f else f.to_px(c.to_stage(px))
    allp = np.vstack([into(c, corner(c)) for c in cvs])
    x0, y0 = allp.min(0)
    x1, y1 = allp.max(0)
    d = max(1.0, (x1 - x0) / max_w, (y1 - y0) / max_w)
    W, H = int(np.ceil((x1 - x0) / d)), int(np.ceil((y1 - y0) / d))
    out = np.zeros((H, W, 3), np.float32)
    for c in cvs:
        k = 1
        while k * 2 <= d and k * 2 <= 2 ** c.L:
            k *= 2
        im = np.nan_to_num(c.region(0, 0, -(-c.W // k), -(-c.H // k), k, color=True), nan=0.0).astype(np.float32)
        src = np.array([[0, 0], [c.W, 0], [0, c.H]], float)
        dst = (into(c, src) - [x0, y0]) / d
        A = cv2.getAffineTransform((src / k).astype(np.float32), dst.astype(np.float32))
        w_ = cv2.warpAffine(im, A, (W, H), flags=cv2.INTER_AREA if k < d else cv2.INTER_LINEAR)
        m = cv2.warpAffine(np.ones(im.shape[:2], np.float32), A, (W, H)) > 0.5
        out[m] = w_[m]
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return dict(path=str(path), frame=f, x0=float(x0), y0=float(y0), d=float(d))


def calib_marks(whole, spots):
    """The calibration spots' 10x fields in calib_whole's picture: [(label, centre x, y, half width, half height,
    current), ...] (pixels of the picture) for spots [(label, spot, current), ...]."""
    f, out = whole["frame"], []
    for lab, sp, cur in spots:
        cv = sp["canvas"]
        c = np.asarray(sp["px"], float)
        q = (c if cv is f else f.to_px(cv.to_stage(c))[0]) - [whole["x0"], whole["y0"]]
        fw, fh = ((cv.D.get("stage_map") or {}).get("wh") or (1376, 1038))
        out.append((lab, float(q[0] / whole["d"]), float(q[1] / whole["d"]), max(6.0, fw / whole["d"] / 2),
                    max(5.0, fh / whole["d"] / 2), bool(cur)))
    return out


CAL100_N = 3                    # the IR-only job's calibration: this many flakes at 100x, centred by hand ...
CAL100_SEP_UM = 400.0           # ... each candidate at least this far from the other corners' flakes


def calib100_flakes(targets, n=CAL100_N):
    """The flakes to centre by hand at 100x for the chips' move (the IR-only job, Avram 2026-11-01: three 100x flakes
    instead of two 10x spots): the n corners of the largest triangle the flakes span (visit_targets dicts; c_old).
    Returns n candidate lists, each its corner's flake first, then the others nearest that corner ("Skip": the next),
    none within CAL100_SEP_UM of another corner.  Fewer than n flakes: []."""
    import cv2
    ts = [t for t in targets if t.get("c_old") is not None]
    if len(ts) < n:
        return []
    S = np.array([t["c_old"] for t in ts], float)
    hull = cv2.convexHull(S.astype(np.float32), returnPoints=False).ravel().tolist()
    if len(hull) < 3:
        hull = list(range(len(ts)))
    if len(hull) > 40:                                  # (the triangle's search: at most 40^3 / 6)
        hull = [hull[i] for i in np.linspace(0, len(hull) - 1, 40).astype(int)]
    best, tri = -1.0, None
    for a in range(len(hull)):
        for b in range(a + 1, len(hull)):
            for c in range(b + 1, len(hull)):
                i, j, k = hull[a], hull[b], hull[c]
                u, w = S[j] - S[i], S[k] - S[i]
                ar = abs(u[0] * w[1] - u[1] * w[0])
                if ar > best:
                    best, tri = ar, (i, j, k)
    if tri is None:
        return []
    order = sorted(tri, key=lambda i: (S[i][0] + S[i][1]))       # ((1): the corner at the smallest x + y)
    out = []
    for i in order:
        others = [S[j] for j in order if j != i]
        d = np.hypot(*(S - S[i]).T)
        c = [ts[j] for j in np.argsort(d) if all(np.hypot(*(S[j] - o)) >= CAL100_SEP_UM for o in others)]
        out.append(c)
    return out


def calib100_marks(whole, items):
    """The calibration flakes' 100x fields in calib_whole's picture: [(label, x, y, half width, half height, current),
    ...] for items [(label, visit target, current), ...] (its c_old, 10x stage um; frame_wh, um_px: the 100x field)."""
    f, out = whole["frame"], []
    for lab, t, cur in items:
        q = f.to_px(np.asarray(t["c_old"], float))[0] - [whole["x0"], whole["y0"]]
        fw, fh = (t.get("frame_wh") or (1376, 1038))
        k = float(t.get("um_px") or 0.114) / f.um_px                 # (100x pixels -> the 10x image's)
        out.append((lab, float(q[0] / whole["d"]), float(q[1] / whole["d"]), max(8.0, fw * k / whole["d"] / 2),
                    max(6.0, fh * k / whole["d"] / 2), bool(cur)))
    return out


def compare_passes(prev, cur, tol=AGREE_UM, frac=None, at_least=False):
    """Two passes {flake key: position (um)} compared: the shift all flakes share taken off (the median), each flake's
    remaining difference.  dict(common, shift, dev {key: um}, max_um, worst, ok, missing (found in one pass only),
    within (the keys within tol), total (flakes found in either pass)).  ok: frac None: every flake of both found in
    both and within tol (the remeasure); else more than frac of all the flakes found in either pass within tol (the
    coordinate check, 2026-11-01: more than half within 5 um; at_least: frac or more, the remeasure since 2026-11-02:
    at least half within 5 um)."""
    keys = sorted(set(prev) & set(cur), key=str)
    missing = sorted((set(prev) | set(cur)) - set(keys), key=str)
    total = len(keys) + len(missing)
    if not keys:
        return dict(common=0, shift=[0.0, 0.0], dev={}, max_um=float("inf"), worst=None, ok=False, missing=missing,
                    within=[], total=total)
    D = np.array([np.asarray(cur[k], float) - np.asarray(prev[k], float) for k in keys])
    sh = np.median(D, axis=0)
    dev = {k: float(np.hypot(*(d - sh))) for k, d in zip(keys, D)}
    worst = max(dev, key=dev.get)
    within = [k for k in keys if dev[k] <= tol]
    if frac is None:
        ok = dev[worst] <= tol and not missing
    else:
        ok = len(within) >= float(frac) * total if at_least else len(within) > float(frac) * total
    return dict(common=len(keys), shift=sh.tolist(), dev=dev, max_um=dev[worst], worst=worst, ok=ok, missing=missing,
                within=within, total=total)


def report_failed(wd, allv, passes, cmps, info, title="Remeasure", kept="The run was not changed."):
    """The passes did not agree: a small report to find out why (<wd>/report/index.html, with a picture per flake and
    pass and a map of the last two passes' differences).  allv: the visits (A: the positions per pass; how / peak /
    thumb: per pass, from Scanner.find100); passes: [{key: position}]; cmps: the compare_passes of each pass with the
    one before (from A2 on); info: what else goes on top (run, spots, focus, ...).  Returns (report folder, summary
    text)."""
    import cv2
    import html as H
    wd = Path(wd)
    rd = wd / "report"
    rd.mkdir(parents=True, exist_ok=True)
    # each flake against the pass before, the shift they share off; and its spread over the passes (each pass's shift
    # against A1 off)
    sh = [np.zeros(2)]
    for c in cmps:
        sh.append(sh[-1] + np.asarray(c["shift"], float))
    spread = {}
    for v in allv:
        pts = [(k, np.asarray(a, float) - sh[k]) for k, a in enumerate(v["A"]) if a is not None and k < len(sh)]
        if len(pts) >= 2:
            P_ = np.array([q for _, q in pts])
            spread[v["key"]] = float(np.hypot(*(P_ - P_.mean(0)).T).max())
    lines = []
    for k, c in enumerate(cmps, 2):
        w = c.get("worst")
        lines.append(f"A{k} against A{k - 1}: {c['common']} flakes in both, common shift {c['shift'][0]:+.2f}, {c['shift'][1]:+.2f} um, "
                     f"largest difference {c['max_um']:.2f} um" + (f" ({w[0]} F{w[1]})" if w else "")
                     + (f", {sum(d > AGREE_UM for d in c['dev'].values())} flakes over {AGREE_UM:g} um" if c["dev"] else ""))
    # what kind of failure: all flakes moving together between passes (the stage slipping: the shift taken off, they
    # agree), a few flakes jumping (wrong matches, a flake found by hand), or many flakes off (the stage slipping
    # during a pass: rows of flakes)
    last = cmps[-1] if cmps else None
    bad = sorted((k_ for k_, d in (last["dev"].items() if last else ()) if d > AGREE_UM), key=lambda k_: -last["dev"][k_])
    by = {v["key"]: v for v in allv}
    hows = {}
    for k_ in bad:
        for h in (by[k_].get("how") or {}).values():
            hows[h] = hows.get(h, 0) + 1
    diag = []
    if last and not bad and last["missing"]:
        diag.append(f"Every flake found in both of the last two passes agrees; {len(last['missing'])} flakes were found in only "
                    "one of them (not recognised, or skipped): see their pictures.")
    elif last and bad:
        frac = len(bad) / max(1, last["common"])
        ys = np.array([float(passes[-1][k_][1]) for k_ in bad if k_ in passes[-1]])
        diag.append(f"{len(bad)} of {last['common']} flakes ({100 * frac:.0f} %) differ by more than {AGREE_UM:g} um between the last two "
                    "passes (after their common shift).")
        if frac > 0.3:
            diag.append("Many flakes off: the stage most likely slips during the passes (look at whether the flakes off are in the "
                        f"same rows: their Y {', '.join(f'{y_:.0f}' for y_ in np.sort(ys)[:12])} um).")
        else:
            diag.append("A few flakes off: most likely a wrong match or a flake that is hard to recognise (see their pictures: "
                        "the cross should be on the same spot of the flake in every pass).")
        if hows:
            diag.append("How those were found (all passes): " + ", ".join(f"{h} {n}" for h, n in sorted(hows.items())) + ".")
    big = [abs(float(np.hypot(*c["shift"]))) for c in cmps]
    if big and max(big) > AGREE_UM:
        diag.append(f"Between passes all flakes moved together by up to {max(big):.1f} um (taken off before comparing: a slip "
                    "between passes does no harm, but it shows the stage slips).")
    # the map: the last pass's flakes, arrows of their difference from the pass before (x 10)
    mp = None
    if last and len(passes) >= 2:
        P_ = np.array([passes[-1][k_] for k_ in last["dev"]], float)
        D_ = np.array([np.asarray(passes[-1][k_], float) - np.asarray(passes[-2][k_], float) - np.asarray(last["shift"], float)
                       for k_ in last["dev"]])
        lo, hi = P_.min(0) - 200, P_.max(0) + 200
        sc = 900.0 / max(float((hi - lo).max()), 1.0)
        W_, H_ = int((hi - lo)[0] * sc) + 40, int((hi - lo)[1] * sc) + 60
        im = np.full((H_, W_, 3), 255, np.uint8)
        for (k_, q), d in zip(last["dev"].items(), D_):
            a = tuple(int(u) for u in ((np.asarray(passes[-1][k_], float) - lo) * sc + 20))
            b = tuple(int(u) for u in (np.array(a) + d * 100 * sc))       # (x 100)
            col = (0, 0, 220) if last["dev"][k_] > AGREE_UM else (0, 150, 0)
            cv2.circle(im, a, 4, col, -1)
            cv2.arrowedLine(im, a, b, col, 2, tipLength=0.2)
            cv2.putText(im, f"F{k_[1]}", (a[0] + 6, a[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
        cv2.putText(im, f"A{len(passes)} - A{len(passes) - 1} (common shift off), arrows x100 of the map's scale; red: over "
                    f"{AGREE_UM:g} um", (10, H_ - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        mp = rd / "map.png"
        cv2.imwrite(str(mp), im)
    # the pictures: per flake its old 100x picture and its frame in each pass (cross: where it was found)
    order = sorted(allv, key=lambda v: -(last["dev"].get(v["key"], -1.0) if last else 0.0))
    trs = []
    for v in order:
        old = rd / f"{v['key'][0]}_F{v['key'][1]}_old.jpg"
        pic = cv2.imread(v["picture"])
        if pic is not None:
            cv2.imwrite(str(old), cv2.resize(pic, (240, int(240 * pic.shape[0] / pic.shape[1]))), [cv2.IMWRITE_JPEG_QUALITY, 80])
        cells = [f'<td><img src="{H.escape(old.name)}"><br>old</td>']
        for k in range(1, len(passes) + 1):
            th = (v.get("thumb") or {}).get(k)
            a = v["A"][k - 1] if k - 1 < len(v["A"]) else None
            txt = ("not found" if a is None else f"{a[0] - v['A'][0][0]:+.1f}, {a[1] - v['A'][0][1]:+.1f}" if v["A"][0] is not None
                   else f"{a[0]:.1f}, {a[1]:.1f}")
            how = (v.get("how") or {}).get(k, "")
            pk = (v.get("peak") or {}).get(k)
            if th and Path(th).is_file():
                shutil.copy2(th, rd / Path(th).name)
                cells.append(f'<td><img src="{H.escape(Path(th).name)}"><br>A{k}: {txt}<br><small>{H.escape(how)}'
                             + (f" {pk:.2f}" if pk else "") + "</small></td>")
            else:
                cells.append(f"<td>A{k}: {txt}<br><small>{H.escape(how)}</small></td>")
        d = last["dev"].get(v["key"]) if last else None
        trs.append(f"<tr><th>{H.escape(v['key'][0])} F{v['key'][1]}<br>last two: {'' if d is None else f'{d:.2f} um'}<br>"
                   f"spread: {spread.get(v['key'], float('nan')):.2f} um<br><small>{H.escape(v.get('why') or '')}</small></th>"
                   + "".join(cells) + "</tr>")
    summary = (f"{title} FAILED: after {len(passes)} passes at 100x the last two still differ by up to "
               f"{last['max_um']:.2f} um" if last else f"{title} FAILED after {len(passes)} passes") + "."
    page = (f"<!doctype html><meta charset='utf-8'><title>{H.escape(title)} failed</title><style>body{{font:13px sans-serif;margin:16px}}"
            "td,th{border:1px solid #ccc;padding:4px;vertical-align:top;text-align:center}th{text-align:left}img{width:200px}"
            "table{border-collapse:collapse}</style>"
            f"<h2>{H.escape(summary)}</h2><p>{H.escape(kept)}  Tolerance {AGREE_UM:g} um.</p>"
            + "".join(f"<p>{H.escape(l)}</p>" for l in diag)
            + "<h3>Passes</h3><ul>" + "".join(f"<li>{H.escape(l)}</li>" for l in lines) + "</ul>"
            + "<h3>Run</h3><ul>" + "".join(f"<li>{H.escape(str(k))}: {H.escape(str(v))}</li>" for k, v in info.items()) + "</ul>"
            + (f"<h3>Map</h3><img src='map.png' style='width:auto;max-width:100%'>" if mp else "")
            + "<h3>Flakes</h3><p>Worst first.  Each pass: the 100x frame where the flake was found (cross), its position from A1 "
              "(um), how (here / mosaic / by hand) and the match.</p><table>" + "".join(trs) + "</table>")
    (rd / "index.html").write_text(page, encoding="utf-8")
    js = dict(summary=summary, diagnosis=diag, passes=lines, info={k: str(v) for k, v in info.items()},
              flakes=[dict(key=list(v["key"]), A=[None if a is None else [round(float(a[0]), 2), round(float(a[1]), 2)] for a in v["A"]],
                           how=v.get("how") or {}, peak=v.get("peak") or {}, last_two_um=last["dev"].get(v["key"]) if last else None,
                           spread_um=spread.get(v["key"]), note=v.get("why") or "") for v in allv])
    (rd / "report.json").write_text(json.dumps(js, indent=1, default=str), encoding="utf-8")
    return rd, summary + "  " + kept + "  " + " ".join(diag)


def apply_positions(old_run, visits, R0, t0, note=None, log=print):
    """The old run moved (in place, saved first to old_coords/<time>/, as remeasure()) to the positions the passes
    agreed on.  visits: {old piece folder: [visit entries]}, each found one with c_new (10x stage um: the last pass's
    100x position less its picture's 10x -> 100x offset).  Each found flake moves to where it was found; the rest of a
    piece by the rigid move fitted to its found flakes (fit_visits; fewer than 3 found: R0 with their mean shift, none:
    R0, t0 the chips' move from the calibration).  Returns the report (also <old run>/remeasure.json)."""
    old_run = Path(old_run)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bk = old_run / BACKUP_DIR / stamp
    report = dict(old_run=str(old_run), date=time.strftime("%Y-%m-%d %H:%M"), backup=str(bk), how="passes", pieces=[],
                  **(note or {}))
    moves = {}
    for piece, vis in visits.items():
        piece = Path(piece)
        found = [v for v in vis if v.get("c_new") is not None]
        vfit = fit_visits(vis, log) if len(found) >= 3 else None
        if vfit:
            R, t, flakes = vfit["R"], vfit["t"], dict(vfit["flakes"])
            near = near_moves(piece, R, t, flakes)
            flakes.update(near)
            how = f"{vfit['n_fit']} of {len(found)} flakes found"
        else:
            R = np.asarray(R0, float)
            sh = [np.asarray(v["c_new"], float) - R @ np.asarray(v["c_old"], float) for v in found]
            t = np.mean(sh, axis=0) if sh else np.asarray(t0, float)
            flakes = {int(v["flake_id"]): q for v, q in zip(found, sh)}
            near = {}
            how = f"{len(found)} flake{'s' if len(found) != 1 else ''} found" + ("" if found else ": the calibration's move")
        rot = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        ctr = np.array([v["c_old"] for v in vis], float).mean(0)
        shift = (R @ ctr + t - ctr).tolist()
        n2 = dict(date=report["date"], rot_deg=round(rot, 4), shift_um=[round(v, 1) for v in shift], backup=f"{BACKUP_DIR}/{stamp}",
                  move_from=how, remeasured=len(vis), found=len(found), how="passes")
        log(f"  {piece.name}: rotated {rot:+.3f} deg, moved {shift[0]:+.1f}, {shift[1]:+.1f} um ({how}"
            + (f", rms {vfit['rms_um']:.2f} um, {len(vfit['outliers'])} away from the chip's move: moved to where they were found, "
               f"{len(near)} more by their neighbours" if vfit else "") + ")")
        n = move_piece(piece, R, t, n2, backup=bk / piece.name, log=log, flakes=flakes)
        moves[piece.name] = piece_move(piece, R, t, flakes)
        report["pieces"].append(dict(old=piece.name, R=np.round(R, 9).tolist(), t=np.round(t, 3).tolist(), rot_deg=rot, shift_um=shift,
                                     move_from=how, files_moved=n, remeasured=len(vis), found=len(found),
                                     own_place=vfit["outliers"] if vfit else [], near=len(near)))
    if moves:
        _finish(old_run, bk, moves, report, log)
    return report


def write_passes(path, vis, n_pass):
    """The flakes' positions in every pass (10x: the rough find; A1..An: 100x stage um) and the coordinates kept."""
    cols = (["flake_id", "picture", "old_x_um", "old_y_um", "found10_x_um", "found10_y_um"]
            + [f"A{k}_{a}_um" for k in range(1, n_pass + 1) for a in "xy"] + ["final_x_um", "final_y_um", "note"])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for v in vis:
            r = dict(flake_id=v["flake_id"], picture=v["stem"], old_x_um=round(v["c_old"][0], 2), old_y_um=round(v["c_old"][1], 2),
                     note=v.get("why") or "")
            if v.get("p10") is not None:
                r.update(found10_x_um=round(v["p10"][0], 2), found10_y_um=round(v["p10"][1], 2))
            for k, a in enumerate(v.get("A") or [], 1):
                if a is not None:
                    r.update({f"A{k}_x_um": round(a[0], 2), f"A{k}_y_um": round(a[1], 2)})
            if v.get("c_new") is not None:
                r.update(final_x_um=round(v["c_new"][0], 2), final_y_um=round(v["c_new"][1], 2))
            w.writerow(r)


def _finish(old_run, bk, moves, report, log=print):
    """After the pieces are moved (move_piece): the run file and the designer's bundle moved too (each saved to the
    backup first), and the report written (<old run>/remeasure.json, and in the backup)."""
    bk.mkdir(parents=True, exist_ok=True)
    for f in old_run.glob("*_run_*.json"):
        shutil.copy2(f, bk / f.name)
        move_run_file(f, f, moves)
    bf = old_run / "designer" / "bundle.json"
    if bf.is_file():                                # (the designer's copy: moved too, no rebuild needed)
        (bk / "designer").mkdir(parents=True, exist_ok=True)
        shutil.copy2(bf, bk / "designer" / "bundle.json")
        try:
            log(f"  designer bundle: {move_bundle(bf, moves)} flakes moved (a page refresh shows them)")
        except Exception as e:
            log(f"  WARNING: designer bundle not moved ({type(e).__name__}: {e}); rebuild the run in the designer")
    if (old_run / "remeasure.json").is_file():          # (an earlier remeasure's report)
        shutil.copy2(old_run / "remeasure.json", bk / "remeasure.json")
    (old_run / "remeasure.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (bk / "remeasure_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


def remeasure(old_run, new_viewers, log=print, plan=None):
    """Match each new piece's 10x image to the old run's pieces, then move the matched pieces' coordinates in the old
    run's folder itself (2026-10-29: in place, no new run).  Every file changed is first copied to
    <old run>/old_coords/<time>/ (restore() puts it back).  new_viewers: {new piece name: its viewer folder}.
    plan: Scanner.remeasure_visit's: the matching (match_pieces) and the visits {old piece: [visit, ...]}: each flake
    found again moves by where it was found, the rest by the rigid move fitted to the visits; without visits (or with
    fewer than 3 found) one rigid move from the matching, as before.  Returns the report (also <old run>/remeasure.json)."""
    old_run = Path(old_run)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bk = old_run / BACKUP_DIR / stamp
    t0 = time.time()
    mp = plan["match"] if plan and plan.get("match") else match_pieces(old_run, new_viewers, log)
    visits = (plan or {}).get("visits") or {}
    olds, regs = mp["olds"], mp["regs"]
    new_viewers = mp.get("new_viewers") or new_viewers
    report = dict(old_run=str(old_run), date=time.strftime("%Y-%m-%d %H:%M"), backup=str(bk), pieces=[])
    moves = {}
    for nk, ok_, reg, rows in regs:                    # (all matched first: nothing changed unless the matching worked)
        nf, nok = len(rows), sum(r["confirmed"] for r in rows)
        R, t, how = reg["R"], reg["t"], "the 10x images"
        flakes, vfit, vis = None, None, visits.get(ok_)
        if vis is not None:
            vfit = fit_visits(vis, log)
            if vfit is None:
                log(f"  {ok_}: only {sum(v.get('c_new') is not None for v in vis)} of {len(vis)} flakes found again on the visits: "
                    "the move from the 10x images is used for all")
            else:
                R, t, how = vfit["R"], vfit["t"], f"{vfit['n_fit']} flakes visited"
                flakes = dict(vfit["flakes"])
                near = near_moves(olds[ok_], R, t, flakes)
                flakes.update(near)
                vfit["near"] = len(near)
        if vfit:                                    # (the move at the visited flakes' centre)
            rot = math.degrees(math.atan2(R[1, 0], R[0, 0]))
            ctr = np.array([v["c_old"] for v in vis], float).mean(0)
            shift = (R @ ctr + t - ctr).tolist()
        else:
            rot, shift = reg["rot_deg"], list(reg["shift_um"])
        note = dict(date=report["date"], rot_deg=round(rot, 4), shift_um=[round(v, 1) for v in shift], rms_um=round(vfit["rms_um"] if vfit else reg["rms_um"], 2),
                    new_scan=nk, flakes_confirmed=f"{nok} of {nf}", backup=f"{BACKUP_DIR}/{stamp}", move_from=how)
        if vfit:
            note.update(visited=len(vis), found=vfit["n_found"], own_place=len(vfit["outliers"]), near=vfit["near"])
        log(f"  {ok_} -> {nk}: rotated {rot:+.3f} deg, moved {shift[0]:+.1f}, {shift[1]:+.1f} um (from {how}"
            + (f", rms {vfit['rms_um']:.2f} um; {vfit['n_found']} of {len(vis)} flakes found again, "
               f"{len(vfit['outliers'])} of them away from the chip's move (moved to where they were found), "
               f"{vfit['near']} more by their visited neighbours" if vfit else
               f"; fit to {reg['n_spots']} of {reg['n_tried']} patches, rms {reg['rms_um']:.2f} um") + ")")
        n = move_piece(olds[ok_], R, t, note, backup=bk / ok_, log=log, flakes=flakes)
        dd = olds[ok_] / "data" if (olds[ok_] / "data").is_dir() else olds[ok_]
        with open(dd / f"{ok_}_remeasure.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["flake_id"])
            w.writeheader()
            w.writerows(rows)
        if vis is not None:
            write_visits(dd / f"{ok_}_remeasure_visits.csv", vis)
        moves[ok_] = piece_move(olds[ok_], R, t, flakes)
        report["pieces"].append(dict(old=ok_, new_scan=nk, new_viewer=str(new_viewers.get(nk, "")), R=np.round(R, 9).tolist(),
                                     t=np.round(t, 3).tolist(), rot_deg=rot, shift_um=shift, move_from=how,
                                     rms_um=vfit["rms_um"] if vfit else reg["rms_um"], n_spots=reg["n_spots"], coarse=reg["coarse"],
                                     files_moved=n, flakes=nf, flakes_confirmed=nok,
                                     flakes_not_confirmed=[r["flake_id"] for r in rows if not r["confirmed"]][:200],
                                     **(dict(visited=len(vis), found=vfit["n_found"], fit=vfit["n_fit"], own_place=vfit["outliers"],
                                             near=vfit["near"], image_move=dict(R=np.round(reg["R"], 9).tolist(), t=np.round(reg["t"], 3).tolist()))
                                        if vfit else {})))
    report["not_matched_old"] = sorted(set(olds) - {r[1] for r in regs})
    report["not_matched_new"] = sorted(set(new_viewers) - {r[0] for r in regs})
    if moves:
        _finish(old_run, bk, moves, report, log)
    if report["not_matched_old"]:
        log(f"  not found in the new scans (left as they were): {', '.join(report['not_matched_old'])}")
    log(f"remeasure done in {time.time() - t0:.0f} s: {len(report['pieces'])} pieces with new coordinates in {old_run}"
        + (f" (the old ones in {bk})" if moves else ""))
    return report


def restore(old_run, stamp=None, log=print):
    """Put the coordinates of a remeasure back: the files saved in <old run>/old_coords/<stamp>/ (default: the latest)
    copied over the current ones.  Returns the number of files restored."""
    old_run = Path(old_run)
    stamps = sorted(p.name for p in (old_run / BACKUP_DIR).iterdir() if p.is_dir()) if (old_run / BACKUP_DIR).is_dir() else []
    if not stamps:
        raise FileNotFoundError(f"no {BACKUP_DIR} in {old_run}")
    bk = old_run / BACKUP_DIR / (stamp or stamps[-1])
    n = 0
    for f in bk.rglob("*"):
        if f.is_file() and f.name != "remeasure_report.json":
            dst = old_run / f.relative_to(bk)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            n += 1
    log(f"restored {n} files from {bk}")
    return n


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m flakeid.scope.remeasure", description=__doc__.split("\n\n")[0])
    ap.add_argument("old_run", help="the old run's results folder (its piece folders with index.html, 100x/, data/): changed in place")
    ap.add_argument("new_viewers", nargs="*", help="the new 10x scan's piece viewer folders")
    ap.add_argument("--restore", nargs="?", const="", metavar="TIME", help="put back the coordinates saved in old_coords/TIME "
                    "(default: the latest)")
    a = ap.parse_args(argv)
    if a.restore is not None:
        restore(a.old_run, a.restore or None)
    else:
        remeasure(a.old_run, {Path(v).name: v for v in a.new_viewers})


if __name__ == "__main__":
    main()
