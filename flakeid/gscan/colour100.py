"""The 100x colour of a flake against the colour model: which of the thicknesses its 10x colour can be.

Interference colours repeat: at 10x (NA 0.25) ~30 and ~316 nm hBN, or ~97 and ~236 nm, have the same colour within the
model's error.  Through the 100x objective (NA 0.9) the light cone is wide and the two no longer look alike.  On the
test chip (2026-11-16) all ten ambiguous regions separated clearly, and the 100x colour alone gave the 10x thickness
within 5 nm for 61 of 70 unambiguous regions (tools/bx53m/check100_colour_trial.py).  Not checked against AFM.

measure(): the colour in the middle of a raw 100x frame (flat field and dark removed) relative to the substrate.
Chooser: the model's contrast curve for the chip's oxide (profile bx53m_dp22_raw_led_100x), clipped where the camera
saturates, and for each candidate thickness the misfit of the measured colour.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .. import real100

PROFILE = "bx53m_dp22_raw_led_100x"
PATCH_UM = 10.0             # the middle of the picture that is measured, um across
MAX_MAD = 0.06              # ... not flat (an edge, dirt): no colour
MIN_CONTRAST = 0.08         # ... bare substrate: no colour
NEAR_NM = 6.0               # a candidate's misfit: the best within this of it
CLEAR = (2.0, 0.03)         # the choice is clear when the second-best misfit > 2 x the best + 0.03


def load_flat(path):
    """(flat at half resolution (h, w, 3), dark per channel) from a 100x flat-field file, or (None, None)."""
    try:
        z = np.load(str(path))
        return z["flat"].astype(np.float32)[::2, ::2], np.median(z["dark"].reshape(-1, 3), axis=0)
    except Exception:
        return None, None


def measure(raw, flat, dark, um_per_px=0.11714):
    """dict(sub=[r, g, b] the frame's commonest colour, med=[r, g, b] the middle patch's colour, mad=its spread) of a
    raw mosaic frame; um_per_px: of the half-resolution channels (2 x 58.57 nm)."""
    ch = [np.asarray(c, np.float32) for c in real100.channels(raw, 0.0)]
    hh, ww = ch[0].shape
    if flat is not None:
        f = cv2.resize(flat, (ww, hh), interpolation=cv2.INTER_AREA) if flat.shape[:2] != (hh, ww) else flat
        ch = [(c - dark[k]) / np.maximum(f[..., k], 0.2) for k, c in enumerate(ch)]
    else:
        ch = [c - 128.0 for c in ch]
    sub = []
    for c in ch:
        hist = np.bincount(np.clip(c[::2, ::2], 0, 4095).astype(np.int64).ravel())
        sub.append(float(np.argmax(np.convolve(hist, np.ones(9), "same"))))
    w = max(8, int(round(PATCH_UM / um_per_px / 2)))
    cy, cx = hh // 2, ww // 2
    patch = np.stack([c[cy - w:cy + w, cx - w:cx + w] for c in ch], 2).reshape(-1, 3)
    med = np.median(patch, 0)
    mad = np.median(np.abs(patch - med), 0) / np.maximum(med, 1.0)
    return dict(sub=sub, med=[float(v) for v in med], mad=float(mad.max()))


class Chooser:
    """The model's 100x colours for one chip (its oxide) and exposure (substrate: the run's substrate colour in counts,
    for where the camera clips)."""

    def __init__(self, material, oxide_nm, substrate, profile=PROFILE):
        from ..optics import ContrastModel
        from ..thickness import camera_from_profile, load_profile
        prof = load_profile(profile)
        self.sub = np.asarray(substrate, float)
        self.clip = float(prof.get("saturation_counts", 3540.0)) / np.maximum(self.sub, 1.0) - 1.0
        self.grid = np.arange(0.0, 400.5, 0.5)
        m = ContrastModel(material, float(oxide_nm), na=float(prof.get("na", 0.88)), camera=camera_from_profile(prof), n_angles=8)
        self.curve = np.minimum(m.contrast(self.grid), self.clip[None])

    def contrast(self, meas):
        """The middle's contrast (r, g, b) against the run's substrate, or None when it has no usable colour."""
        c = np.asarray(meas["med"], float) / self.sub - 1.0
        if meas["mad"] >= MAX_MAD or np.linalg.norm(c) <= MIN_CONTRAST:
            return None
        return np.minimum(c, self.clip)

    def choose(self, meas, candidates):
        """dict(contrast, alone_nm: the 100x colour's own best thickness, misfit: per candidate, pick: the candidate it
        fits best, clear) or None."""
        c = self.contrast(meas)
        if c is None:
            return None
        dist = np.linalg.norm(self.curve - c[None], axis=1)
        out = dict(contrast=[round(float(v), 3) for v in c], alone_nm=float(self.grid[int(np.argmin(dist))]),
                   alone_misfit=round(float(dist.min()), 3))
        cand = [float(t) for t in candidates if t is not None and t == t]
        if cand:
            r = [float(dist[np.abs(self.grid - t) <= NEAR_NM].min()) for t in cand]
            k = int(np.argmin(r))
            s = sorted(r)
            out.update(misfit=[round(v, 3) for v in r], pick=cand[k],
                       clear=len(r) == 1 or s[1] > CLEAR[0] * s[0] + CLEAR[1])
        return out


REGION_BIN = 4              # the region finder works on 4 x 4 blocks of the half-resolution channels (0.47 um)
REGION_EDGE = 0.012         # an edge: the contrast changes by more than this per block
REGION_MIN_CONTRAST = 0.08  # not substrate: contrast further than this from it. 2 nm hBN is ~0.115 on 288 nm oxide, the
                            # substrate drifts by up to ~0.07 across a frame: thinner than ~1.5 nm is not found here
                            # (that is left to the 10x thin-flake search)
REGION_MIN_STEP = 0.045     # ... and differing by this from a ring just outside it (an edge, not a drift of the substrate)
REGION_MIN_UM2 = 20.0       # smaller areas are not reported
REGION_MAX_SPREAD = 0.03    # an area's median contrast spread above this: mottled (residue), not reported
REGION_MAX_MISFIT = 0.12    # an area whose colour fits no thickness this well: no thickness


def regions(raw, flat, dark, chooser, um_per_px=0.11714):
    """The flat, one-colour areas of a raw 100x frame that are not bare substrate, largest first: dicts with
    thickness_nm (from the 100x colour alone; None when it fits no thickness), misfit, size_um (largest inscribed circle;
    a lower bound when the area runs off the picture: cut True), area_um2, dx_um / dy_um (its middle from the cross),
    at_cross.  The 10x picture can miss a faint area beside the target (a thin flake next to a thicker piece): this is
    what the 100x picture itself shows."""
    ch = [np.asarray(c, np.float32) for c in real100.channels(raw, 0.0)]
    hh, ww = ch[0].shape
    if flat is not None:
        f = cv2.resize(flat, (ww, hh), interpolation=cv2.INTER_AREA) if flat.shape[:2] != (hh, ww) else flat
        ch = [(c - dark[k]) / np.maximum(f[..., k], 0.2) for k, c in enumerate(ch)]
    C = np.stack([c / max(float(chooser.sub[k]), 1.0) - 1.0 for k, c in enumerate(ch)], 2).astype(np.float32)
    C = cv2.resize(C, (ww // REGION_BIN, hh // REGION_BIN), interpolation=cv2.INTER_AREA)
    um = um_per_px * REGION_BIN
    G = cv2.GaussianBlur(C, (0, 0), 1.2)
    grad = np.zeros(C.shape[:2], np.float32)
    for k in range(3):
        grad = np.maximum(grad, np.hypot(cv2.Sobel(G[..., k], cv2.CV_32F, 1, 0, ksize=3),
                                         cv2.Sobel(G[..., k], cv2.CV_32F, 0, 1, ksize=3)) / 8)
    mask = ((grad <= REGION_EDGE) & (np.linalg.norm(G, axis=2) > REGION_MIN_CONTRAST)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    h, w = lab.shape
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA] * um * um
        if area < REGION_MIN_UM2:
            continue
        r = lab == i
        px = C[r]
        med = np.median(px, 0)
        spread = float(np.median(np.abs(px - med), 0).max())
        if spread > REGION_MAX_SPREAD:
            continue
        # a flake ends in a step against what surrounds it; a patch of substrate whose colour drifts across the frame
        # (flat field, oxide) fades into it.  The step: the area against a ring just outside it.
        r8 = r.astype(np.uint8)
        ring = cv2.dilate(r8, np.ones((11, 11), np.uint8)).astype(bool) & ~cv2.dilate(r8, np.ones((5, 5), np.uint8)).astype(bool)
        if ring.sum() < 10 or np.linalg.norm(med - np.median(G[ring], 0)) < REGION_MIN_STEP:
            continue
        dt = cv2.distanceTransform(r8, cv2.DIST_L2, 5)
        c = np.minimum(med, chooser.clip)
        dist = np.linalg.norm(chooser.curve - c[None], axis=1)
        k = int(np.argmin(dist))
        x, y, bw, bh = stats[i, :4]
        cut = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        out.append(dict(thickness_nm=float(chooser.grid[k]) if dist[k] <= REGION_MAX_MISFIT else None,
                        misfit=round(float(dist[k]), 3), size_um=round(float(dt.max()) * 2 * um, 1), area_um2=round(area, 1),
                        dx_um=round((cent[i][0] - w / 2) * um, 1), dy_um=round((cent[i][1] - h / 2) * um, 1),
                        at_cross=bool(r[h // 2, w // 2]), cut=bool(cut), contrast=[round(float(v), 3) for v in c]))
    out.sort(key=lambda d: -d["area_um2"])
    return out


def region_range(r):
    """(min, max) thickness of a 100x area for the sorting."""
    t = r["thickness_nm"]
    w = max(2.0, 0.1 * t)
    return max(0.0, t - w), t + w


def pick_flat(flat100, oxide_nm):
    """The 100x flat-field file for an oxide from the setting flat100 (("285=path", ...)): the nearest oxide's."""
    best = None
    for e in flat100 or ():
        k, _, p = str(e).partition("=")
        try:
            d = abs(float(k) - float(oxide_nm))
        except ValueError:
            continue
        if Path(p).is_file() and (best is None or d < best[0]):
            best = (d, p)
    return best[1] if best else None
