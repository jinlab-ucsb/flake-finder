"""High-level flake identification pipeline.

    from flakeid import find_flakes
    result = find_flakes("g31.jpg", material="graphite", oxide_nm=90)
    result.to_csv("g31_flakes.csv")

The result carries everything later stages need (e.g. to drive the
microscope to each flake for high-magnification imaging): positions in
image pixels, micrometres, and — when a stage origin is known — stage
coordinates, plus thickness estimates with uncertainties and quality flags.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import cv2
import os as _os
import numpy as np

from . import __version__
from .background import compute_contrast
from .io import ImageData, load_image
from scipy import ndimage

from .optics import resolve_material
from .flakes import SPLIT_RATIO, build as build_flakes, crowding, focus_width_um, split_by_thickness as split_flakes
from .scorer import Scorer, scorer_path
from .thickness import DEFAULT_PROFILE
from .segment import SegmentParams, segment
from .thickness import (ThicknessEstimator, OxideError, describe_substrate_oxide, load_profile, oxide_from_substrate,
                        refine_oxide, substrate_oxide_problem)

# Default thickness window of interest (nm) per material
DEFAULT_RANGE_NM = {"graphite": (0.3, 25.0), "hbn": (0.0, 100.0)}
try:        # (environment FLAKEID_HBN_RANGE_NM "min,max", the scan setting hbn_range_nm: another window for hBN)
    if _os.environ.get("FLAKEID_HBN_RANGE_NM"):
        DEFAULT_RANGE_NM["hbn"] = tuple(float(v) for v in _os.environ["FLAKEID_HBN_RANGE_NM"].split(","))[:2]
except ValueError:
    pass

# flags from the thin-film model's fit, which rhombohedral screening does not use (its layer number comes from the
# measured colour path, flakeid.rhombo); enclosed_in_thick: thin graphite next to thicker thin graphite is not a facet
RHOMBO_MODEL_FLAGS = {"out_of_range", "poor_colour_match", "off_model", "ambiguous_thickness", "possibly_thick",
                      "colour_mismatch", "bulk", "indistinct", "enclosed_in_thick", "two_branch", "thick_context",
                      "low_confidence"}
# rhombohedral screening: soft there (in the score instead): a large thin flake often has a wide edge at 10x (a step
# or a tapering rim along part of it; flake 3 of LH_G_048_noAsh_noHeat_MT_3's crop, 5000 um2, was lost to it)
RHOMBO_SOFT = {"blurry_edges", "non_uniform", "not_flat", "in_haze", "residue_like"}
# (Avram, 2026-10-28: "the outline doesn't matter that much, what counts is a large amount of pixels together of the right
# colour": in rhombohedral screening only the colour removes a region (thick_or_bulk, low_contrast, no_signal); shape,
# edge and flatness only lower the score, through the clean pixels it counts: score10 core x fill)
RHOMBO_MIN_LAYERS = 0.7         # fainter than a monolayer on the colour path: residue, haze, not graphite

# Minimum |contrast| for a reliable identification: graphene monolayers on
# ~90 nm SiO2 give ~0.06-0.09 (gamma-encoded); hBN on 285 nm SiO2 reaches
# 0.15 at ~3.5 nm.  Thin hBN is separated from residue by the full-resolution
# flatness (fill) and shape (fullres_solidity) checks.
DEFAULT_MIN_CONTRAST = {"graphite": 0.06, "hbn": 0.15}

# thin hBN first (Avram, 2026-10-29: "a 1 nm BN that would now have a score of 0.6 would get 0.9"): the 10x score of
# hBN of THIN_HBN_FULL_NM or less becomes 1 - (1 - score) / 4, the boost fading out linearly by THIN_HBN_END_NM
THIN_HBN_FULL_NM = 2.0
THIN_HBN_END_NM = 4.0
THIN_HBN_KEEP = 0.25                # the share of (1 - score) a thin hBN keeps
# the thin-hBN search (opt-in: find_flakes thin_search, flakeid.live --thin-hbn N; 2026-10-30): regions of THIN_SEARCH_*
# nm in the image's thin-hBN channel (flakeid.segment.thin_channel: ~ blue - red), away from the normal detections, are
# measured and screened like any region but for the low-contrast / indistinct rules (made for faint residue: here the
# shape, flatness and haze rules do that); flagged "thin_search", numbered after the normal flakes (theirs unchanged)
THIN_SEARCH_MIN_NM = 1.0           # (20261027_run_13: the background is 0.25 - 0.4 nm rms in the channel; below ~1 nm what
THIN_SEARCH_MAX_NM = 5.0           # is found is haze, not flakes)
THIN_SEARCH_STEP = 0.7             # a flake's edge: the channel just outside (1.5 - 4 px) falls by >= 70 % of its inside
                                   # (a haze patch fades into more haze: the strongest test on run_13, 17084 -> 532 regions)



try:
    STRICT_UNIFORM_D_UM = float(_os.environ.get("FLAKEID_MIN_UNIFORM_D_UM", "0") or 0)
except ValueError:
    STRICT_UNIFORM_D_UM = 0.0
# A faint region in a field of similar haze is still a flake if it is compact itself (environment
# FLAKEID_RESIDUE_RING_SOLIDITY, the scan setting residue_ring_solidity; 0 = off).  Glovebox BX53M, 2026-11-16: a clean
# 12 nm hBN flake beside tape glue (raw_3 100x picture 21) was left out as residue for its surroundings alone (ring
# 0.18); of the 110 regions of that chip left out only as residue, the 7 with solidity >= 0.85 and sharp edges held that
# flake and no labelled glue, the other 103 held 15 labelled glue pictures and no flake (glue: solidity 0.3-0.77)
try:
    RESIDUE_RING_SOLIDITY = float(_os.environ.get("FLAKEID_RESIDUE_RING_SOLIDITY", "0") or 0)
except ValueError:
    RESIDUE_RING_SOLIDITY = 0.0
# Faint flat flakes (environment FLAKEID_FAINT_FLAKES=1, the scan setting faint_flakes; glovebox BX53M, 2026-11-17): a
# transparent-material region left out only as low_contrast and/or residue_like is kept, flagged "faint_flake", when it
# is clearly a film and not glue: the colour of FAINT_MIN_NM or more of the material (fit residual <= FAINT_MAX_RESIDUAL),
# flat at full resolution (fill >= FAINT_MIN_FILL), a sharp edge (<= FAINT_MAX_EDGE_UM) and no similar haze around it
# (ring_similar <= FAINT_MAX_RING).  20261117_1: a flat 2 nm hBN flake 25 um across (photographed on another microscope)
# was left out as low_contrast + residue_like; of the 562 regions this camera's analysis leaves out that way, the rule
# keeps 7 (6 um or wider), that flake among them.
FAINT_FLAKES = _os.environ.get("FLAKEID_FAINT_FLAKES", "") not in ("", "0")
FAINT_SOFT = {"low_contrast", "residue_like"}
FAINT_MIN_NM, FAINT_MAX_RESIDUAL, FAINT_MIN_FILL, FAINT_MAX_EDGE_UM, FAINT_MAX_RING = 1.5, 0.05, 0.95, 2.5, 0.2

def _thin_step(r, cs, labels, w):
    """How much the thin-hBN channel (weights w) falls at a region's edge: 1 - (median just outside) / (median inside);
    outside: 1.5 - 4 working px out, not in another region."""
    if w is None:
        return 0.0
    pad = 6
    H, W = labels.shape
    sl = (slice(max(r.slices[0].start - pad, 0), min(r.slices[0].stop + pad, H)),
          slice(max(r.slices[1].start - pad, 0), min(r.slices[1].stop + pad, W)))
    m = np.zeros((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start), np.uint8)
    oy, ox = r.slices[0].start - sl[0].start, r.slices[1].start - sl[1].start
    m[oy:oy + r.mask.shape[0], ox:ox + r.mask.shape[1]] = r.mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ring = cv2.dilate(m, k, iterations=4).astype(bool) & ~cv2.dilate(m, k, iterations=1).astype(bool) & (labels[sl] == 0)
    if ring.sum() < 10:
        return 0.0
    ch = cs[sl] @ np.asarray(w, np.float32)
    inside = float(np.median(ch[m.astype(bool)]))
    return 1.0 - float(np.median(ch[ring])) / inside if inside > 0 else 0.0


def thin_hbn_score(score, t_nm):
    """The 10x score of an hBN terrace t_nm thick, with the thin-hBN boost."""
    if not (score == score and t_nm == t_nm) or t_nm >= THIN_HBN_END_NM:
        return score
    w = min(max((THIN_HBN_END_NM - t_nm) / (THIN_HBN_END_NM - THIN_HBN_FULL_NM), 0.0), 1.0)
    return 1.0 - (1.0 - score) * (1.0 - w * (1.0 - THIN_HBN_KEEP))


@dataclass
class Terrace:
    """A region of uniform thickness (region of interest) inside a flake."""
    terrace_id: int
    flake_id: int                   # the flake (connected region of material) it belongs to
    x_px: float                     # centroid, original image pixels
    y_px: float
    target_x_px: float              # point deepest inside the region (best for imaging)
    target_y_px: float
    x_um: float                     # centroid in um from the image origin (top-left)
    y_um: float
    target_x_um: float
    target_y_um: float
    stage_x_um: float | None
    stage_y_um: float | None
    thickness_nm: float
    thickness_min_nm: float
    thickness_max_nm: float
    layers: float                   # thickness / interlayer spacing
    alt_thickness_nm: str           # other thickness solutions that fit the colour
    length_um: float                # long side of the minimum-area bounding rectangle
    width_um: float                 # short side
    orientation_deg: float          # angle of the long side, degrees from +x (image)
    area_um2: float
    inscribed_diameter_um: float    # largest circle fully inside the region
    solidity: float                 # area / convex hull area (ragged debris is low)
    fill: float                     # full-res fraction of the interior matching the flake colour
    fullres_solidity: float         # solidity of the matching full-res pixels (dendrites are low)
    ring_similar: float             # fraction of surroundings with the same colour (residue haze fields)
    bbox_x_px: int
    bbox_y_px: int
    bbox_w_px: int
    bbox_h_px: int
    contrast_r: float
    contrast_g: float
    contrast_b: float
    fit_residual: float
    uniformity: float
    edge_width_um: float            # 20-80 % width of the flake edge (sharp ~2-3 um; smudges/defocus larger)
    saturated_frac: float
    # --- the flake this terrace belongs to (whole connected region of material)
    flake_area_um2: float
    flake_inscribed_d_um: float
    flake_focus_um: float           # edge width of the flake outline at full resolution (focus; ~1-2 um sharp)
    flake_crowding: float           # fraction of the 5-30 um surroundings covered by other material
    flake_frac_window: float        # fraction of the flake whose thickness is inside the window
    flake_frac_thick: float         # fraction thicker than the window / bulk / unidentifiable dark
    score: float                    # learned P(worth imaging) from user labels (profiles/scorer.json)
    quality: str                    # "good", or ';'-separated warning flags
    # --- context for thin / thick hBN (transparent materials; 0 otherwise): colours only thick films show
    edge_thick_um2: float = 0.0     # edge pixels (+-2 um of the outline, not in another region) of a thick-only colour
    thick_nb_um2: float = 0.0       # thick-only colour nearby: touching regions + pixels 2-8 um out (> ~60 nm)
    core_um2: float = float("nan")  # rhombohedral screening: the clean core (the region less 2.5 um along its edge)
    flake_core_um2: float = float("nan")  # ... the flake's largest connected clean core of 3-25 layer colour (the score)


Flake = Terrace                      # backwards compatibility


@dataclass
class FlakeSummary:
    """One flake: a whole connected region of material, with its terraces."""
    flake_id: int                   # rank by score: 1 = best candidate
    score: float                    # best terrace score
    material: str
    n_terraces: int
    terrace_ids: str                # ';'-separated, best first
    thicknesses_nm: str             # ';'-separated, same order
    best_terrace_id: int
    best_thickness_nm: float
    best_terrace_d_um: float
    x_px: float                     # flake centroid
    y_px: float
    target_x_px: float              # centre of the best terrace (aim here)
    target_y_px: float
    x_um: float
    y_um: float
    target_x_um: float
    target_y_um: float
    stage_x_um: float | None        # of the target point
    stage_y_um: float | None
    area_um2: float
    inscribed_diameter_um: float
    length_um: float
    width_um: float
    bbox_x_px: int
    bbox_y_px: int
    bbox_w_px: int
    bbox_h_px: int
    focus_um: float
    crowding: float
    frac_window: float
    frac_thick: float
    quality: str                    # flags of the best terrace


@dataclass
class FlakeResult:
    image: str
    material: str
    oxide_nm: float
    oxide_nm_used: float
    pixel_um: float
    pixel_um_source: str
    flakes: list                              # FlakeSummary, ranked
    terraces: list = field(default_factory=list)   # Terrace
    params: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    # working arrays (not serialised) for annotation / further analysis
    _labels: np.ndarray | None = None
    _scale: float = 1.0
    _region_of: dict = field(default_factory=dict)   # terrace_id -> region label
    _flake_lab: np.ndarray | None = None              # flake label image (working res)
    _flake_of: dict = field(default_factory=dict)     # flake_id -> label in _flake_lab
    _contrast: np.ndarray | None = None      # working-resolution contrast map (for the viewer)
    _substrate_rgb: np.ndarray | None = None
    _oxide_samples: np.ndarray | None = None  # contrast samples used to refine the oxide (live: pooled over blocks)
    _clip_max: np.ndarray | None = None
    _clean: list = field(default_factory=list)  # clean areas (find_flakes clean_areas=...; flakeid.clean), else []
    _material: np.ndarray | None = None       # rhombohedral screening: all material (thick too; the flakes are thin only)

    @staticmethod
    def _write(path, rows, cls):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cls.__dataclass_fields__))
            w.writeheader()
            for f in rows:
                row = asdict(f)
                for k, v in row.items():
                    if isinstance(v, float):
                        row[k] = round(v, 4)
                w.writerow(row)

    def to_csv(self, path, terraces_path=None):
        """Flake table to `path`; terrace table to `terraces_path` (default: *_terraces.csv next to it)."""
        path = Path(path)
        self._write(path, self.flakes, FlakeSummary)
        tp = Path(terraces_path) if terraces_path else path.with_name(path.stem.replace("_flakes", "") + "_terraces.csv")
        self._write(tp, self.terraces, Terrace)
        return tp

    def to_json(self, path):
        d = {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}
        d["flakes"] = [asdict(fl) for fl in self.flakes]
        d["terraces"] = [asdict(t) for t in self.terraces]
        Path(path).write_text(json.dumps(d, indent=2, default=_json_default))


def _region_from_mask(label, m, sl, cs, chip, saturated):
    """A segment.Region from a mask in a bounding box (rhombohedral screening: the right-colour pixels of a flake in
    which the segmentation found no terrace)."""
    from .segment import Region
    px = cs[sl][m]
    med = np.median(px, axis=0)
    dev = np.abs(px - med).max(axis=1)
    sat = saturated[sl][m] if saturated is not None and saturated.ndim == 3 else np.zeros((len(px), 3), bool)
    inner = cv2.erode(chip.astype(np.uint8), np.ones((5, 5), np.uint8))[sl].astype(bool)
    return Region(label=label, parent=-label, slices=sl, mask=m, median=med.astype(np.float32),
                  spread=float(np.linalg.norm(np.median(np.abs(px - med), axis=0))),
                  uniformity=float((dev < max(0.045, 0.08 * float(np.linalg.norm(med)))).mean()),
                  saturated_frac=float(sat.any(axis=1).mean()) if len(sat) else 0.0,
                  saturated_ch=sat.mean(axis=0) if len(sat) else np.zeros(3), touches_chip_edge=bool((m & ~inner).any()),
                  edge_width=float("nan"), thick_border=0.0, ring_similar=0.0)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _collect_uniform_samples(contrast, chip, saturated, n=20000, seed=0):
    """Contrast samples from flat, non-substrate, unsaturated areas (for oxide refinement)."""
    cb = cv2.blur(contrast, (5, 5))
    c2 = cv2.blur(contrast * contrast, (5, 5))
    sd = np.sqrt(np.maximum(c2 - cb * cb, 0)).max(axis=2)
    satd = cv2.dilate(saturated.any(axis=2).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    m = chip & ~satd & (sd < 0.02) & (np.abs(cb).max(axis=2) > 0.05)
    s = cb[m]
    if len(s) > n:
        s = s[np.random.default_rng(seed).choice(len(s), n, replace=False)]
    return s


def fullres_check(rgb, r, work_scale, tol_abs=0.07, tol_rel=0.12):
    """Verify a region on the original full-resolution pixels.

    Returns (fill, solidity):
      fill      fraction of the region interior whose contrast matches the
                region's median (flat flakes ~0.9; dendritic residue or clusters
                of tiny fragments, which only look uniform after down-sampling,
                are much lower)
      solidity  area / convex hull of the matching pixels (simple flake
                geometry ~0.8-1; thin dendrites low)
    """
    k3 = np.ones((3, 3), np.uint8)
    y0, x0 = int(r.slices[0].start / work_scale), int(r.slices[1].start / work_scale)
    h, w = int(round(r.mask.shape[0] / work_scale)), int(round(r.mask.shape[1] / work_scale))
    pad = 16
    ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
    yb, xb = min(y0 + h + pad, rgb.shape[0]), min(x0 + w + pad, rgb.shape[1])
    crop = rgb[ya:yb, xa:xb].astype(np.float32)
    M = np.zeros(crop.shape[:2], np.uint8)
    mm = cv2.resize(r.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    M[y0 - ya:y0 - ya + h, x0 - xa:x0 - xa + w] = mm[:M.shape[0] - (y0 - ya), :M.shape[1] - (x0 - xa)]
    ring = cv2.dilate(M, k3, iterations=12).astype(bool) & ~cv2.dilate(M, k3, iterations=5).astype(bool)
    inner = cv2.erode(M, k3, iterations=2).astype(bool)
    if ring.sum() < 20 or inner.sum() < 12:
        return float("nan"), float("nan")
    ref = np.median(crop[ring], axis=0)
    ok = np.all(np.abs(crop[ring] / np.maximum(ref, 1) - 1) < 0.05, axis=1)   # substrate-like ring pixels
    if ok.sum() >= 20:
        ref = np.median(crop[ring][ok], axis=0)
    c = crop / np.maximum(ref, 1.0) - 1.0
    med = np.median(c[inner], axis=0)
    tol = max(tol_abs, tol_rel * float(np.linalg.norm(med)))
    on = np.linalg.norm(c - med, axis=2) < tol
    fill = float(on[inner].mean())
    cand = on & cv2.dilate(M, k3, iterations=2).astype(bool)
    lab, n = ndimage.label(cand)
    if n == 0:
        return fill, 0.0
    big = lab == (1 + np.argmax(np.bincount(lab.ravel())[1:]))
    pts = np.argwhere(big)[:, ::-1].astype(np.float32)
    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull) + 0.5 * cv2.arcLength(hull, True) + 1.0
    return fill, float(min(1.0, big.sum() / hull_area))


def _solidity(mask):
    """Area / convex-hull area of a region mask."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(np.vstack(cnts).reshape(-1, 2).astype(np.float32))
    # polygon through boundary-pixel centres misses ~half a pixel all round
    hull_area = cv2.contourArea(hull) + 0.5 * cv2.arcLength(hull, True) + 1.0
    return float(min(1.0, mask.sum() / max(hull_area, 1.0)))


def thick_context(regions, labels, contrast, est, px_um, thin_hi=60.0, band_um=2.0, near_um=8.0, min_c=0.15):
    """region label -> (edge_thick_um2, thick_nb_um2): thick-only colours at the region's edge and in touching regions.

    A colour is thick-only if it is within 0.10 of the model curve beyond thin_hi and more than 0.25 from every
    colour of films up to thin_hi (curve points closer than 0.30 to a thin colour are left out: shared colours)."""
    from scipy.spatial import cKDTree
    d, cv = est.d, est.curve
    thin = cKDTree(cv[d <= thin_hi])
    thick = cv[d > thin_hi]
    only = thick[thin.query(thick)[0] > 0.30]
    if len(only) == 0:
        return {}
    tonly = cKDTree(only)

    def is_thick(v):
        return (tonly.query(v)[0] < 0.10) & (thin.query(v)[0] > 0.25)
    rmed = {r.label: r for r in regions}
    # thick-only colour, or clipped in a channel no thin film's colour clips (the camera saturates on thick hBN)
    thin_max = cv[d <= thin_hi].max(axis=0)
    def thick_region(r):
        if r.saturated_frac < 0.5:
            return bool(is_thick(r.median[None, :])[0])
        clip_ch = r.saturated_ch > 0.5
        return bool(np.any(clip_ch & (thin_max < (est.clip_max if est.clip_max is not None else np.inf) - 0.05)))
    thick_reg = {r.label for r in regions if thick_region(r)}
    H, W = labels.shape
    b = max(1, int(round(band_um / px_um)))
    bn = max(b + 1, int(round(near_um / px_um)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * b + 1, 2 * b + 1))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bn + 1, 2 * bn + 1))
    # the bands and rings of all regions first, then is_thick once on the pixels any of them needs (neighbouring
    # regions' rings overlap: the per-pixel test was a tenth of a 10x block's analysis), then the sums
    todo = []
    need = np.zeros((H, W), bool)
    for r in regions:
        if float(np.linalg.norm(r.median)) < min_c:
            continue
        y0, y1 = max(r.slices[0].start - bn - 1, 0), min(r.slices[0].stop + bn + 1, H)
        x0, x1 = max(r.slices[1].start - bn - 1, 0), min(r.slices[1].stop + bn + 1, W)
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        m[r.slices[0].start - y0:r.slices[0].stop - y0, r.slices[1].start - x0:r.slices[1].stop - x0][r.mask] = 1
        outer = cv2.dilate(m, ker).astype(bool)
        inner = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
        lab = labels[y0:y1, x0:x1]
        band = outer & ~inner & ((lab == 0) | (lab == r.label))
        nb = set(np.unique(lab[outer & ~m.astype(bool)])) - {0, r.label}
        a = sum(float(rmed[k].mask.sum()) for k in nb if k in thick_reg) * px_um ** 2
        # thick material nearby that is no uniform region (colour gradients of thick crystals stay unlabelled):
        # thick-only colour pixels 2-8 um out, not in this region's own band
        ring = cv2.dilate(m, kern).astype(bool) & ~outer & (lab != r.label)
        need[y0:y1, x0:x1] |= band | ring
        todo.append((r.label, y0, y1, x0, x1, band, ring, a))
    T = np.zeros((H, W), bool)
    if need.any():
        T[need] = is_thick(contrast[need])
    out = {}
    for lbl, y0, y1, x0, x1, band, ring, a in todo:
        Tw = T[y0:y1, x0:x1]
        e = float(Tw[band].sum()) * px_um ** 2 if band.any() else 0.0
        if ring.any():
            a += float(Tw[ring].sum()) * px_um ** 2
        out[lbl] = (e, a)
    return out


def find_flakes(image, material: str, oxide_nm: float, *,
                thickness_range_nm: tuple | None = None,
                min_area_um2: float = 100.0,
                pixel_um: float | None = None,
                work_scale: float = 0.5,
                profile: str | dict | None = None,
                refine_oxide_thickness: bool = True,
                oxide_entered_nm: float | None = None,
                max_residual: float = 0.20,
                min_uniformity: float = 0.6,
                max_edge_width_um: float = 5.0,
                indistinct_ratio: float = 1.5,
                min_solidity: float = 0.6,
                min_contrast: float | None = None,
                thin_search: bool = False,
                max_blue_green: float = 0.8,
                ambiguous_residual: float = 0.25,
                prefer_thin_ambiguous: bool = True,
                thick_margin: float = 0.08,
                bulk_margin: float = 0.03,
                min_fill: float = 0.75,
                min_fullres_solidity: float = 0.6,
                faint_contrast: float = 0.4,
                min_fill_faint: float = 0.9,
                min_fullres_solidity_faint: float = 0.75,
                max_ring_similar: float = 0.3,
                haze_contrast: float = 0.6,
                residue_ring_similar: float = 0.15,
                residue_solidity: float = 0.75,
                residue_edge_um: float = 3.0,
                low_confidence_contrast: float = 0.25,
                thick_edge_um2: float = 12.0,
                thick_near_um2: float = 50.0,
                max_saturated_frac: float = 0.2,
                stage_origin_um: tuple | None = None,
                stage_axis_sign: tuple = (1.0, 1.0),
                include_rejected: bool = False,
                clean_areas: tuple | None = None,
                keep_contrast: bool = False,
                segment_params: SegmentParams | None = None,
                substrate_hint=None,
                oxide_only: bool = False,
                rhombo: bool = False,
                verbose: bool = False) -> FlakeResult:
    """Identify uniform-contrast flakes of a given material in an overview image.

    image:              path or ImageData
    material:           'graphite' or 'hbn' (see optics.MATERIALS)
    oxide_nm:           SiO2 thickness in the thin-film model (thickness.model_oxide_nm of the entered one)
    refine_oxide_thickness: raw Micro-Manager scans with oxide_entered_nm: the oxide from the substrate's colour
                        (thickness.oxide_from_substrate; OxideError if it is more than 20 % from the entered one or fits no
                        oxide); other images: fitted to the colours of the flat flake regions (thickness.refine_oxide)
    oxide_entered_nm:   the oxide as entered (nominal), for the substrate-colour oxide
    thickness_range_nm: (min, max) thickness of interest; default per material
    min_area_um2:       smallest region reported
    work_scale:         processing resolution relative to the original image
    stage_origin_um:    stage (x, y) of the image's top-left pixel; defaults to
                        the origin found in the .vsi metadata, if available
    stage_axis_sign:    (+1/-1, +1/-1) mapping image +x/+y to stage +x/+y
    clean_areas:        (min, max) nm: also the clean areas of that thickness, whatever the flakes' outlines
                        (flakeid.clean; FlakeResult._clean: their middle in image pixels and stage um, area, thickness,
                        and the identified flake there, if any)
    prefer_thin_ambiguous: inclusive about thickness (2026-10-25): a region whose best colour fit is thicker than the
                        window but whose colour an in-window film also matches acceptably (ambiguous_thickness) is
                        kept and reported at the in-window (thinner) thickness, the thick one as its alternative, and
                        is not counted as thick around the other regions; False: as before 2026-10-25
    include_rejected:   also report regions outside the thickness range or
                        failing quality cuts (flagged in the 'quality' column)
    substrate_hint:     known substrate RGB level (image units) for chip detection (live analysis)
    oxide_only:         stop after the oxide refinement: no flakes, only the oxide, its samples and the substrate level
                        (live analysis: the oxide is fixed from the first blocks before any block is analysed in full)
    rhombo:             rhombohedral screening of graphite on ~285 nm oxide (flakeid.rhombo): thin or thick and the layer
                        number from the measured colour path instead of the thin-film model; a flake is the connected
                        thin graphite only (thick graphite and bulk attached to it are neither in its outline nor in its
                        size); the score is the size of each terrace's clean core within 3-25 layers
    """
    t0 = time.time()
    log = (lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a)) if verbose else (lambda *a: None)

    material = resolve_material(material)
    img = image if isinstance(image, ImageData) else load_image(image, pixel_um=pixel_um)
    log(f"loaded {img.path.name} {img.rgb.shape[1]}x{img.rgb.shape[0]} px, "
        f"{img.pixel_um:.4f} um/px ({img.pixel_um_source})")
    if profile is None and img.metadata.get("source") == "micromanager" \
            and img.metadata.get("data_format", "").upper().startswith("RAW"):
        profile = "bx51_xc10_mm_raw16"          # linear raw sensor data (Stream profiles include its processing)
        if _os.environ.get("FLAKEID_PROFILE_RAW"):  # (another camera's raw data: its own profile, the scan setting
            profile = _os.environ["FLAKEID_PROFILE_RAW"]    # profile_raw; one profile for every material there)
        if material == "graphite":              # (its own fit: the hBN-calibrated one put 2-12 nm graphite at 8-11 nm)
            profile = "bx51_xc10_mm_raw16_graphite"
            log("graphite: profile fitted on graphite_20261019_run_1 (1-4 layer anchors + the run's colours); above "
                "1.3 nm not yet checked against AFM")
    prof = profile if isinstance(profile, dict) else load_profile(profile)
    log(f"profile: {Path(prof['_path']).stem if '_path' in prof else 'custom'}")
    tmin, tmax = thickness_range_nm or DEFAULT_RANGE_NM.get(material, (0.0, 100.0))
    seg_p = segment_params or SegmentParams()
    if min_contrast is None:
        min_contrast = DEFAULT_MIN_CONTRAST.get(material, 0.08)

    cm = compute_contrast(img.rgb, img.pixel_um, work_scale=work_scale, max_value=img.max_value,
                          substrate_hint=substrate_hint)
    px_um = img.pixel_um / work_scale
    log(f"background flattened; substrate noise (RGB) = {np.round(cm.noise, 4)}")

    # largest contrast each channel can record before the camera clips
    clip_level = img.clip_value if img.clip_value is not None else img.max_value - 0.5
    clip_max = clip_level / np.maximum(cm.substrate_rgb, 1.0) - 1.0
    oxide_used, oxide_gain = float(oxide_nm), 0.0
    samples = None
    oxide_source = "entered"
    raw = img.metadata.get("source") == "micromanager" and str(img.metadata.get("data_format", "")).upper().startswith("RAW")
    if refine_oxide_thickness and raw and oxide_entered_nm:
        sr = oxide_from_substrate(cm.substrate_rgb, oxide_entered_nm)
        log(describe_substrate_oxide(sr, oxide_entered_nm))
        why = substrate_oxide_problem(sr, oxide_entered_nm)
        if why:
            raise OxideError(f"analysis stopped: {why}.  The thicknesses would be wrong (--no-refine-oxide: the entered oxide)")
        oxide_used, oxide_source = sr["model_nm"], "substrate"
    elif refine_oxide_thickness:
        oxide_source = "flakes"
        samples = _collect_uniform_samples(cm.contrast, cm.chip_mask, cm.saturated)
        if len(samples) > 500:
            ox, gain = refine_oxide(samples, material, oxide_nm, prof, clip_max=clip_max)
            if gain > 0.05:
                oxide_used, oxide_gain = ox, gain
        log(f"oxide thickness used: {oxide_used:.1f} nm (nominal {oxide_nm}, gain {oxide_gain:.2f})")
    if oxide_only:
        return FlakeResult(image=str(img.path), material=material, oxide_nm=float(oxide_nm), oxide_nm_used=oxide_used,
                           pixel_um=img.pixel_um, pixel_um_source=img.pixel_um_source, flakes=[],
                           params=dict(oxide_refine_gain=oxide_gain, profile=prof.get("_path")), metadata=dict(img.metadata),
                           _substrate_rgb=cm.substrate_rgb, _oxide_samples=samples, _clip_max=clip_max)

    est = ThicknessEstimator(material, oxide_used, prof, clip_max=clip_max)
    mono = est.monolayer_contrast()

    min_px = max(seg_p.min_region_px, int(min_area_um2 / px_um ** 2))
    thin_cfg = None
    if thin_search and material == "hbn" and not rhombo:
        thin_cfg = dict(g=np.asarray(est.model.contrast([1.0])[0], float), lo=THIN_SEARCH_MIN_NM, hi=THIN_SEARCH_MAX_NM)
    seg = segment(cm.contrast, cm.chip_mask, cm.saturated, cm.noise, seg_p, min_parent_px=min_px, thin=thin_cfg)
    regions, cs, labels = seg[:3]
    thin_labels = set()
    if thin_cfg is not None:
        th = seg[3]
        thin_labels = th["labels"]
        log(f"thin-hBN search: {len(thin_labels)} regions in the thin-hBN channel (weights R G B {th['w']}, background "
            f"{th['noise_nm']} nm rms, from {th['lo_nm']} nm)")
    log(f"segmented {len(regions)} uniform regions")
    rh_shift = 0.0
    if rhombo:                          # (the colour paths moved for the chip's oxide offset: flakeid.rhombo)
        from . import rhombo as RH
        rh_shift = float(rhombo.get("oxide_shift_nm") or 0.0) if isinstance(rhombo, dict) else 0.0
        RH.use_oxide_shift(rh_shift)
    rh_layers, thin_px, all_material = {}, None, None
    if rhombo:
        from . import rhombo as RH
        from .flakes import material_mask
        thin_px, n_px = RH.layers_of(cs, "10x")
        in_win = thin_px & (n_px >= RH.WINDOW_LAYERS[0]) & (n_px <= RH.WINDOW_LAYERS[1])     # (the right colour)
        all_material = material_mask(cs, cm.noise, cm.chip_mask, material) | (labels > 0)
        rh_layers = {r.label: RH.region_layers(r.median, "10x") for r in regions}
        log(f"rhombohedral screening: {sum(v is not None for v in rh_layers.values())} of {len(regions)} regions thin "
            "graphite (on the measured colour path), the rest thick, bulk or not graphite")


    meta = dict(img.metadata)
    if stage_origin_um is None and "origin_x_um" in meta:
        stage_origin_um = (meta["origin_x_um"], meta["origin_y_um"])
    sx, sy = stage_axis_sign

    def to_orig(v):  # working pixel coordinate -> original pixel coordinate
        return (v + 0.5) / work_scale - 0.5

    # pass 1: colour fit of every region, and per-parent share of material
    # that is thicker than the window or not identifiable (bulk, shadows, ...)
    fits = {}
    par_area, par_thick = {}, {}
    absorbing = est.model._n_fl.imag.mean() > 0.3        # graphite-like: colour saturates to bulk
    cmin = est.curve.min(axis=0)
    from .branch import load as load_branch
    branch = None if absorbing else load_branch(Path(prof.get("_path", "")).stem, material, oxide_used, log)
    branch_p = {}                                       # region label -> P(thick) where two branches fit

    def bulk_like(c):
        # thick (bulk or lifted) graphite: neutral grey and darker than any thin film can be on this oxide;
        # the nearest-curve fit would otherwise put it at the darkest point of the curve (~5 nm)
        return bool(absorbing and c[1] < cmin[1] - bulk_margin and c[2] / c[1] > max_blue_green)

    for r in regions:
        # channels clipped in most pixels only give a lower bound on the contrast
        fit = est.fit(r.median, censored=r.saturated_ch > 0.5)
        dec = branch.decide(est, r.median, censored=r.saturated_ch > 0.5) if branch else None
        if dec is not None:
            # colour fits both a thin and a thick branch: reported thin with the thick alternative; thin or thick is
            # decided over the whole chip once it is analysed (branch.resolve, at the target list), since the
            # boundary between the two colour groups shifts slightly from chip to chip
            p_k, tn, dn, tk, dk = dec
            branch_p[r.label] = p_k
            fit.thickness_nm, fit.residual = tn, dn
            fit.range_nm = (tn, tn)
            fit.alternatives_nm = [round(tk, 1)]
        fits[r.label] = fit
        a = float(r.mask.sum())
        par_area[r.parent] = par_area.get(r.parent, 0.0) + a
        thick_ = (fit.thickness_nm > tmax or r.saturated_frac > 0.5
                  or fit.residual > max(max_residual, 0.25 * float(np.linalg.norm(r.median)))
                  or bulk_like(r.median))
        if thick_ and prefer_thin_ambiguous and not absorbing:
            # inclusive (2026-10-25): a colour that a film inside the window also has (an acceptable match) is not
            # counted as thick material around the other regions (enclosed_in_thick, the flake's thick fraction)
            t_in, r_in = est.best_in_range(r.median, tmin, tmax, censored=r.saturated_ch > 0.5)
            thick_ = not r_in <= max(max_residual, 0.25 * float(np.linalg.norm(r.median)))
        if thick_:
            par_thick[r.parent] = par_thick.get(r.parent, 0.0) + a

    # context for thin / thick (transparent materials): a ~150 nm hBN flake has the 10x colour of a ~20 nm one, but its
    # tapered edges and thinner steps pass through colours no film under ~60 nm shows (orange, red, purple on ~300 nm
    # oxide), and it often touches thick material whose colour is unambiguous
    ctx = {}
    if not absorbing:
        ctx = thick_context(regions, labels, cm.contrast, est, px_um, thin_hi=(branch.cal["thin_range_nm"][1] if branch else 60.0))

    # tape residue (transparent materials): a faint region (|C| < haze_contrast) in a field of similar haze, irregular
    # or with diffuse edges is residue film, not a thin flake (10x residue labels, multitest chip_1, 2026-10-02: caught
    # 30 of 31 residue terraces, lost 2 of 9 faint flake terraces, both ~2 nm with score ~0.02).  It is flagged, and
    # kept out of the flake masks: the film otherwise joins a real flake next to it and the flake outline spreads
    # over the residue field (a flake mask 6x its terraces on mixed detections)
    residue = set()
    if not absorbing:
        for r in regions:
            if float(np.linalg.norm(r.median)) >= haze_contrast:
                continue
            sol, wide = _solidity(r.mask), r.edge_width * px_um > residue_edge_um
            ring = r.ring_similar > residue_ring_similar and not (0 < RESIDUE_RING_SOLIDITY <= sol)
            if ring or sol < residue_solidity or wide:
                residue.add(r.label)

    def in_window(r, fit):
        """(thickness, residual) of the region's best colour match inside the window, and whether it fits nearly as
        well as the best fit (an out-of-window fit: interference colours repeat)."""
        t_in, r_in = est.best_in_range(r.median, tmin, tmax, censored=r.saturated_ch > 0.5)
        cn = float(np.linalg.norm(r.median))
        near = r_in <= max(ambiguous_residual, 1.5 * fit.residual) and cn >= 0.25
        if prefer_thin_ambiguous:           # (inclusive: any acceptable colour match inside the window)
            near = r_in <= max(ambiguous_residual, 1.5 * fit.residual, max_residual, 0.25 * cn)
        return t_in, r_in, near

    # flakes: whole connected regions of material; terraces are regions of interest inside
    # (rhombohedral screening: flakes of thin graphite only, thick regions are part of no flake)
    off_path = {lb for lb, n in rh_layers.items() if n is None}
    # (the thin-hBN search's regions are left out of the normal flakes, which are built exactly as without the search,
    # and make flakes of their own, numbered after them: flake_lab_n, the normal flakes alone, for their crowding)
    labels_n = np.where(np.isin(labels, np.fromiter(thin_labels, np.int32)), 0, labels) if thin_labels else labels
    FLK, flake_lab, region_flake = build_flakes([r for r in regions if r.label not in thin_labels], labels_n, cs, cm.noise,
                                                cm.chip_mask, material, px_um, work_scale, exclude=residue | off_path,
                                                allow=thin_px)
    if not absorbing and not rhombo:
        # (the thickness each region will be reported at: an out-of-window fit whose in-window match is nearly as good is
        # reported thin, below; 20261103_4: ~25 nm terraces fitted ~156 nm at this point)
        t_rep = {}
        for r in regions:
            fit = fits.get(r.label)
            if fit is None:
                continue
            t_rep[r.label] = fit.thickness_nm
            if branch_p.get(r.label) is None and not (tmin <= fit.thickness_nm <= tmax) and prefer_thin_ambiguous:
                t_in, _, near = in_window(r, fit)
                if near and fit.thickness_nm > tmax and t_in < min(fit.thickness_nm, tmax - 5.0):
                    t_rep[r.label] = float(t_in)
        # touching flakes of very different thickness are separate flakes (flakes.split_by_thickness: one at least
        # twice the other, 2026-11-04)
        n_sp = split_flakes(FLK, flake_lab, region_flake, [r for r in regions if r.label not in thin_labels],
                            t_rep, px_um, work_scale)
        if n_sp:
            log(f"{n_sp} connected flake{'s' if n_sp != 1 else ''} split by thickness (one part at least "
                f"{SPLIT_RATIO:g}x the other)")
    flake_lab_n, thin_fids = flake_lab, set()
    if thin_labels:
        labels_t = seg[3]["labels_img"]               # (the thin regions whole, also where they lie over normal ones)
        Ft, lab_t, rf_t = build_flakes([r for r in regions if r.label in thin_labels], labels_t, cs, cm.noise, cm.chip_mask,
                                       material, px_um, work_scale, allow=labels_t > 0)
        off = int(flake_lab.max()) + 1
        for fid, fo in Ft.items():
            fo.fid = fid + off
            FLK[fid + off] = fo
            thin_fids.add(fid + off)
        flake_lab = np.where((lab_t > 0) & (flake_lab == 0), lab_t + off, flake_lab).astype(flake_lab.dtype)
        region_flake.update({k: v + off for k, v in rf_t.items()})
    rh_core = {}
    if rhombo:
        # what counts is a large amount of pixels together of the right colour (Avram, 2026-10-28): each flake's largest
        # connected clean core of 3-25 layer colour, whatever its terraces (a large flake whose thickness varies a
        # little, 6-10 layers, was split into one terrace and unlabelled rest, and ranked by the terrace alone); a flake
        # in which the segmentation found no terrace at all gets one from those pixels (a 5000 um2 area had no flake)
        have = set(region_flake.values())
        nxt = n_new = int(labels.max()) + 1
        for fid, fl in FLK.items():
            sl = fl.slices
            rc = fl.mask & in_win[sl]
            rh_core[fid], comp_m = RH.largest_core(rc, px_um)
            if fid in have or rh_core[fid] < RH.FILL_CORE_UM2:
                continue
            r = _region_from_mask(nxt, comp_m, sl, cs, cm.chip_mask, cm.saturated)
            labels[sl][comp_m & (labels[sl] == 0)] = nxt
            regions.append(r)
            region_flake[nxt] = fid
            fl.regions.append(nxt)
            rh_layers[nxt] = RH.region_layers(r.median, "10x")
            fits[nxt] = est.fit(r.median, censored=r.saturated_ch > 0.5)
            par_area[r.parent] = float(r.mask.sum())
            nxt += 1
        if nxt > n_new:
            log(f"rhombohedral screening: {nxt - n_new} flakes without a terrace given one from their 3-25 layer pixels")
    comp = {fid: [0.0, 0.0] for fid in FLK}
    for r in regions:
        fid = region_flake.get(r.label)
        if fid is None:
            continue
        fit = fits[r.label]
        a = float(r.mask.sum()) * px_um ** 2
        ts = [fit.thickness_nm] + list(fit.alternatives_nm)
        bad = fit.residual > max(max_residual, 0.25 * float(np.linalg.norm(r.median)))
        if not bad and not bulk_like(r.median) and any(tmin <= t <= tmax for t in (ts if not absorbing else ts[:1])):
            comp[fid][0] += a
        elif bulk_like(r.median) or fit.thickness_nm > tmax or r.saturated_frac > 0.5 or (bad and float(np.mean(r.median)) < -0.3):
            comp[fid][1] += a
    fmetric = {}

    def flake_metrics(fid):
        if fid not in fmetric:
            fl = FLK[fid]
            fmetric[fid] = dict(area=fl.area_um2, d=fl.inscribed_d_um,
                                focus=focus_width_um(img.rgb, fl, work_scale, img.pixel_um, cm.substrate_rgb),
                                crowd=crowding(fl, flake_lab if fid in thin_fids else flake_lab_n, px_um),
                                win=min(1.0, comp[fid][0] / fl.area_um2), thick=min(1.0, comp[fid][1] / fl.area_um2))
        return fmetric[fid]

    flakes, region_of = [], {}
    kept_labels = set()                  # (normal regions kept: a thin-search region over one of them is that flake again)
    for r in regions:
        npx = int(r.mask.sum())
        area = npx * px_um ** 2
        if area < min_area_um2:
            continue
        if r.label in thin_labels and kept_labels & set(seg[3]["overlap"].get(r.label, ())):
            continue
        fit = fits[r.label]
        flags = []
        bp = branch_p.get(r.label)
        if bp is not None:
            flags.append("two_branch")
        elif not (tmin <= fit.thickness_nm <= tmax):
            # interference colours repeat; keep regions that also have an
            # acceptable solution inside the window, flagged as ambiguous
            t_in, r_in, near = in_window(r, fit)
            if not absorbing and near:
                flags.append("ambiguous_thickness")
                # (not for a colour just past tmax: its in-window match is the window's end, so it would be reported at
                # tmax exactly; it is a little thick, not the repeated thin colour; multitest chip_1, 2026-10-25: ~30
                # terraces fitted 100-106 nm would all have read 100.0)
                if prefer_thin_ambiguous and fit.thickness_nm > tmax and t_in < min(fit.thickness_nm, tmax - 5.0):
                    # the thinner, in-window thickness fits nearly as well: reported as the thickness (exfoliated flakes
                    # of that size are more often the thinner one; 2026-10-25), the thick one as the alternative
                    t_out = round(float(fit.thickness_nm), 1)
                    fit.thickness_nm, fit.residual, fit.range_nm = float(t_in), float(r_in), (float(t_in), float(t_in))
                    fit.alternatives_nm = sorted((set(fit.alternatives_nm) | {t_out}) - {round(t_in, 1)})
                else:
                    fit.alternatives_nm = sorted(set(fit.alternatives_nm) | {round(t_in, 1)})
            else:
                flags.append("out_of_range")
        elif not absorbing and tmax < est.d[-1]:
            # interference colours repeat: on ~300 nm oxide a ~150 nm hBN flake has nearly the colour of a ~20 nm
            # one at 10x.  A thicker-than-window branch that fits almost as well -> the flake may well be thick
            t_k, r_k = est.best_in_range(r.median, tmax, float(est.d[-1]), censored=r.saturated_ch > 0.5)
            if r_k <= fit.residual + thick_margin:
                flags.append("possibly_thick")
                fit.alternatives_nm = sorted(set(fit.alternatives_nm) | {round(t_k, 1)})
        if float(np.max(r.median)) < -0.97:
            flags.append("no_signal")          # black: outside the imaged area or the chip (edge slivers)
        elif bulk_like(r.median):
            flags.append("bulk")
        # model errors grow with contrast, so the tolerance scales with |C|
        if fit.residual > max(max_residual, 0.25 * float(np.linalg.norm(r.median))):
            flags.append("poor_colour_match")
        if getattr(fit, "off_model", False):
            flags.append("off_model")          # (soft: the thickness is a wide range, flakeid.thickness.OFF_MODEL_RESIDUAL)
        # colour barely distinguishable from bare substrate relative to the fit
        # quality (e.g. faint residue that happens to match a near-invisible
        # thickness): not a reliable identification
        thin_r = r.label in thin_labels
        if thin_r:
            flags.append("thin_search")          # (the thin-hBN search: below the per-channel limits by design)
            if _thin_step(r, cs, labels, seg[3]["w"]) < THIN_SEARCH_STEP:
                flags.append("haze_edge")
        elif np.linalg.norm(r.median) < indistinct_ratio * fit.residual:
            flags.append("indistinct")
        if np.linalg.norm(r.median) < min_contrast and not thin_r:
            flags.append("low_contrast")
        # Few-layer graphite (absorbing films) on ~90 nm SiO2 is green-dominated
        # (observed B/G ~0.35-0.65).  Faint neutral-grey regions (B/G ~0.85-1)
        # are facets of thick crystals, dirt or shadows, not thin flakes.
        if absorbing and fit.thickness_nm < 5 and r.median[1] < 0:
            bg_ratio = float(r.median[2] / r.median[1])
            if bg_ratio > max_blue_green:
                flags.append("colour_mismatch")
        if r.label in residue:
            flags.append("residue_like")
        # thick by context: a colour that may be thin or thick, with clear thick-only colours at its edge or around it
        # (user thin/thick labels, multitest chip_1, 2026-10-03: 34 of 83 thick flakes, 0 of 55 thin)
        e_k, n_k = ctx.get(r.label, (0.0, 0.0))
        if (e_k >= thick_edge_um2 or n_k >= thick_near_um2) and any(
                f in flags for f in ("two_branch", "possibly_thick", "ambiguous_thickness")):
            flags.append("thick_context")
        if r.uniformity < min_uniformity:
            flags.append("non_uniform")
        if r.edge_width * px_um > max_edge_width_um:        # nan (no free edge) passes
            flags.append("blurry_edges")
        if r.saturated_frac > max_saturated_frac:
            flags.append("saturated")
        if r.touches_chip_edge:
            flags.append("chip_edge")
        if (any(tmin <= a <= tmax for a in fit.alternatives_nm) and "out_of_range" not in flags
                and "ambiguous_thickness" not in flags):
            flags.append("ambiguous_thickness")
        # context: faint facets inside thick crystals resemble few-layer films
        thick_parent = par_thick.get(r.parent, 0.0) / par_area[r.parent] > 0.2
        if (absorbing and thick_parent and "out_of_range" not in flags
                and fit.thickness_nm < 0.6 * tmax and r.thick_border > 0.5):
            flags.append("enclosed_in_thick")
        rh_n = None
        if rhombo:
            # the measured colour path decides: thin graphite (its layer number replaces the model's thickness, whose
            # flags do not apply) or not (thick, bulk, dirt: no terrace)
            rh_n = rh_layers.get(r.label)
            flags = [f for f in flags if f not in RHOMBO_MODEL_FLAGS]
            if rh_n is None:
                flags.append("thick_or_bulk")
            elif rh_n < RHOMBO_MIN_LAYERS:
                flags.append("low_contrast")
            else:
                fit.thickness_nm = rh_n * est.layer_nm
                fit.range_nm = (0.8 * fit.thickness_nm, 1.25 * fit.thickness_nm)
                fit.alternatives_nm = []
        hard = {"out_of_range", "poor_colour_match", "non_uniform", "blurry_edges", "haze_edge",
                "indistinct", "low_contrast", "enclosed_in_thick", "colour_mismatch", "bulk", "no_signal",
                "residue_like", "thick_or_bulk"} - (RHOMBO_SOFT if rhombo else set())
        # user labels: strongly coloured regions with an imperfect colour fit are
        # as often worth imaging as the rest -> only a soft flag (scored)
        if float(np.linalg.norm(r.median)) >= 0.4:
            hard = hard - {"poor_colour_match"}
        if STRICT_UNIFORM_D_UM > 0:
            hard = hard - {"residue_like"}      # (strict uniformity, below: the width of the uniform area decides)
        # faint flat flake (FAINT_FLAKES): only the faint-residue rules reject it, and its colour, edge and surroundings
        # are a film's; flatness is checked below, at full resolution
        faint_try = (FAINT_FLAKES and not absorbing and not rhombo and bool(hard & set(flags)) and hard & set(flags) <= FAINT_SOFT
                     and fit.thickness_nm >= FAINT_MIN_NM and fit.residual <= FAINT_MAX_RESIDUAL
                     and r.edge_width * px_um <= FAINT_MAX_EDGE_UM and r.ring_similar <= FAINT_MAX_RING)
        if not include_rejected and hard & set(flags) and not faint_try:
            continue

        # geometry
        m8 = r.mask.astype(np.uint8)
        ys, xs = np.nonzero(r.mask)
        y0, x0 = r.slices[0].start, r.slices[1].start
        cy, cx = ys.mean() + y0, xs.mean() + x0
        dist = cv2.distanceTransform(np.pad(m8, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        ty, tx = np.unravel_index(np.argmax(dist), dist.shape)
        inscribed = 2 * float(dist.max()) * px_um
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = np.vstack(cnts).reshape(-1, 2).astype(np.float32)
        (_, _), (w_, h_), ang = cv2.minAreaRect(pts)
        solidity = _solidity(r.mask)
        fill, fr_sol = fullres_check(img.rgb, r, work_scale)
        # faint regions are where residue (dendritic, patchy) is confused with
        # thin flakes: demand flatter contrast and simpler geometry there
        faint = float(np.linalg.norm(r.median)) < faint_contrast
        if faint and not (fill == fill):      # could not be verified at full resolution (nan)
            fill, fr_sol = 0.0, 0.0
        if solidity < min_solidity or fr_sol < (min_fullres_solidity_faint if faint else min_fullres_solidity):
            flags.append("irregular_shape")
        if fill < (min_fill_faint if faint else min_fill):
            flags.append("not_flat")
        cmag = float(np.linalg.norm(r.median))
        if cmag < haze_contrast and r.ring_similar > max_ring_similar and not rhombo:
            # (rhombohedral screening: not for graphite on the thin-graphite colour path; the terraces of a large flake
            # surround each other in similar colours, and the rule dropped the best flakes of LH_G_048, 9-40 thousand um2)
            flags.append("in_haze")
        if cmag < low_confidence_contrast:
            flags.append("low_confidence")      # faint: identification less certain
        if faint_try:
            if fill >= FAINT_MIN_FILL and "in_haze" not in flags:
                flags = [f for f in flags if f not in FAINT_SOFT] + ["faint_flake"]
            elif not include_rejected:
                continue
        strict = STRICT_UNIFORM_D_UM > 0 and not rhombo
        if not include_rejected and not rhombo and ({"in_haze"} if strict else {"not_flat", "in_haze"}) & set(flags):
            continue                                                                         # irregular_shape: soft (labels)
        # strict uniformity (environment FLAKEID_MIN_UNIFORM_D_UM, the scan setting strict_uniformity_um; glovebox BX53M,
        # 2026-11-13): tape glue has the colour of thin hBN in that camera's picture and only its shape tells it apart:
        # patchy, no uniform area wider than 5-7 um (12.6 um in one large patch; the good flakes there: 27-36 um).  A
        # region must hold a uniform disc of this diameter, else it is left out.  The rules not_flat, residue_like and
        # irregular_shape do not reject there: their tolerances are for the XC10's raw data, and on the DP22's 8-bit
        # picture they dropped a flat, faint 36 um flake (the thin flakes wanted)
        if not include_rejected and strict and inscribed < STRICT_UNIFORM_D_UM:
            continue
        if w_ < h_:
            w_, h_, ang = h_, w_, ang + 90.0
        length, width = (w_ + 1) * px_um, (h_ + 1) * px_um

        x_px, y_px = to_orig(cx), to_orig(cy)
        tx_px, ty_px = to_orig(tx + x0), to_orig(ty + y0)
        pxo = img.pixel_um
        stage_x = stage_y = None
        if img.stage_map is not None:           # Micro-Manager mosaic: exact per-tile mapping
            stage_x, stage_y = img.stage_map.to_stage(tx_px, ty_px)
        elif stage_origin_um is not None:
            stage_x = stage_origin_um[0] + sx * tx_px * pxo
            stage_y = stage_origin_um[1] + sy * ty_px * pxo

        fid = region_flake.get(r.label)
        fmx = flake_metrics(fid) if fid is not None else dict(area=area, d=inscribed, focus=float("nan"),
                                                               crowd=0.0, win=1.0, thick=0.0)
        flakes.append(Terrace(
            terrace_id=0, flake_id=fid if fid is not None else -r.label,
            x_px=x_px, y_px=y_px, target_x_px=tx_px, target_y_px=ty_px,
            x_um=x_px * pxo, y_um=y_px * pxo, target_x_um=tx_px * pxo, target_y_um=ty_px * pxo,
            stage_x_um=stage_x, stage_y_um=stage_y,
            thickness_nm=round(fit.thickness_nm, 1),
            thickness_min_nm=round(fit.range_nm[0], 1), thickness_max_nm=round(fit.range_nm[1], 1),
            layers=round(fit.thickness_nm / est.layer_nm, 1),
            alt_thickness_nm=";".join(f"{a:g}" for a in fit.alternatives_nm),
            length_um=length, width_um=width, orientation_deg=(-ang) % 180.0,
            area_um2=area, inscribed_diameter_um=inscribed, solidity=solidity,
            fill=fill, fullres_solidity=fr_sol, ring_similar=r.ring_similar,
            bbox_x_px=int(to_orig(x0 - 0.5) + 0.5), bbox_y_px=int(to_orig(y0 - 0.5) + 0.5),
            bbox_w_px=int(round(r.mask.shape[1] / work_scale)),
            bbox_h_px=int(round(r.mask.shape[0] / work_scale)),
            contrast_r=float(r.median[0]), contrast_g=float(r.median[1]), contrast_b=float(r.median[2]),
            fit_residual=fit.residual, uniformity=r.uniformity,
            edge_width_um=r.edge_width * px_um, saturated_frac=r.saturated_frac,
            flake_area_um2=fmx["area"], flake_inscribed_d_um=fmx["d"], flake_focus_um=fmx["focus"],
            flake_crowding=fmx["crowd"], flake_frac_window=fmx["win"], flake_frac_thick=fmx["thick"],
            score=float("nan"), quality=";".join(flags) if flags else "good",
            edge_thick_um2=round(ctx.get(r.label, (0.0, 0.0))[0], 1), thick_nb_um2=round(ctx.get(r.label, (0.0, 0.0))[1], 1),
            core_um2=round(RH.core_area_um2(r.mask, px_um), 1) if rhombo else float("nan"),
            flake_core_um2=round(rh_core.get(fid, float("nan")), 1) if rhombo else float("nan")))
        region_of[id(flakes[-1])] = r.label
        kept_labels.add(r.label)

    # score terraces; a flake scores as its best terrace.  Flakes are ranked by
    # score (flake 1 = best candidate), terraces grouped by flake, best first.
    scorer = Scorer(scorer_path(Path(prof.get("_path", "")).stem, material))
    terr = flakes
    # the default scorer was trained on Stream images, whose contrasts are compressed by
    # its processing; with another camera profile its contrast feature is taken
    # from the Stream model at the fitted thickness instead of the measured value.
    # A scorer fitted on this camera's own labels (profiles/scorer_<profile>_<material>.json) uses the measured contrast.
    ref_est = None
    if scorer.ok and not scorer.native and Path(prof.get("_path", "")).stem != DEFAULT_PROFILE:
        ref_est = ThicknessEstimator(material, oxide_nm, load_profile(None))
    for t in terr:
        if rhombo:
            t.score = RH.score10(t.core_um2, t.layers, t.uniformity, t.flake_focus_um, t.quality.split(";"), t.fill,
                                 flake_core_um2=t.flake_core_um2)
            continue
        cmag = None
        if ref_est is not None and t.thickness_nm == t.thickness_nm:
            cmag = float(np.linalg.norm(ref_est.model.contrast([t.thickness_nm])[0]))
        t.score = round(scorer(t, material, tmin, tmax, contrast_mag=cmag), 4) if scorer.ok else float("nan")
        if material == "hbn":
            t.score = round(thin_hbn_score(t.score, t.thickness_nm), 4)
    key = (lambda t: -t.score) if scorer.ok or rhombo else (lambda t: -t.area_um2)
    groups = {}
    for t in terr:
        groups.setdefault(t.flake_id, []).append(t)
    for g in groups.values():
        g.sort(key=key)
    # (flakes of the thin-hBN search after all the others: the normal flakes keep their numbers)
    order = sorted(groups, key=lambda k: (all("thin_search" in t.quality for t in groups[k]), key(groups[k][0])))
    terraces, summaries, reg_map, flake_of = [], [], {}, {}
    for rank, old in enumerate(order, start=1):
        g = groups[old]
        for t in g:
            reg_map[len(terraces) + 1] = region_of[id(t)]
            t.terrace_id = len(terraces) + 1
            t.flake_id = rank
            terraces.append(t)
        best = g[0]
        fl = FLK.get(old)
        if fl is not None:
            flake_of[rank] = old
            sl = fl.slices
            fx, fy = fl.centroid_px
            area, dd, L_, W_ = fl.area_um2, fl.inscribed_d_um, fl.length_um, fl.width_um
            bx, by = int(to_orig(sl[1].start - 0.5) + 0.5), int(to_orig(sl[0].start - 0.5) + 0.5)
            bw, bh = int(round(fl.mask.shape[1] / work_scale)), int(round(fl.mask.shape[0] / work_scale))
        else:
            fx, fy, area, dd, L_, W_ = best.x_px, best.y_px, best.area_um2, best.inscribed_diameter_um, best.length_um, best.width_um
            bx, by, bw, bh = best.bbox_x_px, best.bbox_y_px, best.bbox_w_px, best.bbox_h_px
        pxo = img.pixel_um
        summaries.append(FlakeSummary(
            flake_id=rank, score=best.score, material=material, n_terraces=len(g),
            terrace_ids=";".join(str(t.terrace_id) for t in g),
            thicknesses_nm=";".join(f"{t.thickness_nm:g}" for t in g),
            best_terrace_id=best.terrace_id, best_thickness_nm=best.thickness_nm,
            best_terrace_d_um=best.inscribed_diameter_um,
            x_px=fx, y_px=fy, target_x_px=best.target_x_px, target_y_px=best.target_y_px,
            x_um=fx * pxo, y_um=fy * pxo, target_x_um=best.target_x_um, target_y_um=best.target_y_um,
            stage_x_um=best.stage_x_um, stage_y_um=best.stage_y_um,
            area_um2=area, inscribed_diameter_um=dd, length_um=L_, width_um=W_,
            bbox_x_px=bx, bbox_y_px=by, bbox_w_px=bw, bbox_h_px=bh,
            focus_um=best.flake_focus_um, crowding=best.flake_crowding,
            frac_window=best.flake_frac_window, frac_thick=best.flake_frac_thick, quality=best.quality))
    flakes = summaries
    log(f"{len(flakes)} flakes with {len(terraces)} terraces reported")
    clean = []
    if clean_areas is not None:
        from .clean import clean_areas as find_clean
        rank_of = {lab_: rank for rank, lab_ in flake_of.items()}
        fsum = {f.flake_id: f for f in flakes}
        for c in find_clean(cm.contrast, cm.chip_mask, est, px_um, clean_areas, min_contrast):
            tx_px, ty_px = to_orig(c["tx"]), to_orig(c["ty"])
            st = (None, None)
            if img.stage_map is not None:
                st = img.stage_map.to_stage(tx_px, ty_px)
            elif stage_origin_um is not None:
                st = (stage_origin_um[0] + sx * tx_px * img.pixel_um, stage_origin_um[1] + sy * ty_px * img.pixel_um)
            fid = rank_of.get(int(flake_lab[c["ty"], c["tx"]])) if flake_lab is not None else None
            f_ = fsum.get(fid)
            clean.append(dict(c, target_x_px=float(tx_px), target_y_px=float(ty_px), stage_x_um=st[0], stage_y_um=st[1],
                              flake_id=fid, flake_stage=[f_.stage_x_um, f_.stage_y_um] if f_ is not None else None))
        log(f"{len(clean)} clean areas of {clean_areas[0]:g}-{clean_areas[1]:g} nm" + (f" (largest {clean[0]['area_um2']:.0f} um2)" if clean else ""))

    params = dict(material=material, thickness_range_nm=[tmin, tmax], min_area_um2=min_area_um2,
                  **({"rhombo_oxide_shift_nm": rh_shift} if rhombo else {}),
                  work_scale=work_scale, max_residual=max_residual, min_uniformity=min_uniformity,
                  max_edge_width_um=max_edge_width_um, min_solidity=min_solidity,
                  indistinct_ratio=indistinct_ratio, min_contrast=min_contrast, max_blue_green=max_blue_green, ambiguous_residual=ambiguous_residual,
                  prefer_thin_ambiguous=prefer_thin_ambiguous,
                  thick_margin=thick_margin, bulk_margin=bulk_margin,
                  min_fill=min_fill, min_fullres_solidity=min_fullres_solidity,
                  faint_contrast=faint_contrast, min_fill_faint=min_fill_faint,
                  min_fullres_solidity_faint=min_fullres_solidity_faint,
                  max_ring_similar=max_ring_similar, haze_contrast=haze_contrast,
                  residue_ring_similar=residue_ring_similar, residue_solidity=residue_solidity, residue_edge_um=residue_edge_um,
                  low_confidence_contrast=low_confidence_contrast, thick_edge_um2=thick_edge_um2, thick_near_um2=thick_near_um2,
                  refine_oxide=refine_oxide_thickness, oxide_refine_gain=oxide_gain, oxide_source=oxide_source,
                  profile=prof.get("_path"), scorer="rhombo" if rhombo else scorer.path.name if scorer.ok else None,
                  rhombo=bool(rhombo), segment=asdict(seg_p),
                  **({"thin_search": {k: v for k, v in seg[3].items() if k in ("w", "noise_nm", "lo_nm")}} if thin_cfg is not None else {}),
                  substrate_noise=cm.noise.tolist(), substrate_rgb=cm.substrate_rgb.tolist(),
                  monolayer_contrast=mono.tolist(), stage_origin_um=stage_origin_um,
                  stage_axis_sign=list(stage_axis_sign), version=__version__,
                  runtime_s=round(time.time() - t0, 1))
    return FlakeResult(image=str(img.path), material=material, oxide_nm=float(oxide_nm),
                       oxide_nm_used=oxide_used, pixel_um=img.pixel_um,
                       pixel_um_source=img.pixel_um_source, flakes=flakes, terraces=terraces, params=params,
                       metadata=meta, _labels=labels, _scale=work_scale, _region_of=reg_map,
                       _flake_lab=flake_lab, _flake_of=flake_of,
                       _contrast=cm.contrast if keep_contrast else None,
                       _substrate_rgb=cm.substrate_rgb, _oxide_samples=samples, _clip_max=clip_max, _clean=clean,
                       _material=all_material)
