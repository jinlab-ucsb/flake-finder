"""What flakes are wanted, and what that means for the scan.

The rule of this program: the 10x step is generous and the criteria are applied at the end.  The 10x thickness is not
calibrated against AFM on this microscope, tape glue looks like thin flakes at 10x, and a flake not photographed at
100x is lost; a picture too many costs about 7 s.  So at 10x a flake is only left out when it cannot be wanted (its
flat area much smaller than the smallest wanted size; optionally: clearly thicker than the thickest wanted), and the
thin end is never cut.  After the 100x pictures every flake is sorted (collect.classify):

    match      a real flake, and thickness, size and cleanliness are all inside the criteria
    possible   a real flake that may be inside them: its thickness or size is not known (no flat region under the
               middle of the picture), or the same colour is also a thickness inside the window
    other      a real flake outside the criteria
    doubtful   the real / glue sorter is not sure (probability between DOUBT and the cut)
    glue       tape residue

Nothing is deleted: a scan can be sorted again with other criteria (the window's "Sort again"; no microscope needed),
and the results page has the same boxes to try values on the spot.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DOUBT = 0.15                # below this probability of a real flake: glue; from here to the cut: doubtful
SIZE_MARGIN = 0.6           # 10x: flat areas down to this share of the smallest wanted size are photographed
THICK_MARGIN = (1.5, 10.0)  # 10x, "skip the clearly thicker ones": thicker than max x 1.5 + 10 nm is left out
MAX_NM = 400.0              # the colour model's range


@dataclass
class Criteria:
    material: str = "hbn"
    oxide_nm: float = 285.0
    thickness_min_nm: float = 0.0
    thickness_max_nm: float = 400.0
    size_min_um: float = 15.0           # the flat area's width: the largest circle that fits in it, um
    size_max_um: float = 0.0            # 0 = no upper limit
    clean_min: float = 0.0              # least share of the flake that is flat and clean at 100x, 0..1 (0 = any)
    real_cut: float = 0.5               # least probability of a real flake (the 100x sorter); lower keeps more
    thin_search: bool = True            # 10x: also the thin-hBN search (flakes of ~1-5 nm, below the normal detection)
    thin_max_targets: int = 60          # ... at most this many of its finds per chip are photographed at 100x
    pictures_per_cm2: float = 600.0     # at most this many 100x pictures per cm2 of chip (the best-scoring first)
    skip_thick_at_10x: bool = False     # 10x: flakes clearly thicker than the window are not photographed (faster)
    copy_to: str = ""                   # a folder the results are copied to at the end ("" = they stay on this PC)

    def clean(self):
        """The values put in order (a typed minimum above the maximum, a share above 1 ...)."""
        c = Criteria(**asdict(self))
        c.thickness_min_nm = max(0.0, float(c.thickness_min_nm))
        c.thickness_max_nm = float(c.thickness_max_nm) if float(c.thickness_max_nm) > 0 else MAX_NM
        if c.thickness_max_nm < c.thickness_min_nm:
            c.thickness_min_nm, c.thickness_max_nm = c.thickness_max_nm, c.thickness_min_nm
        c.size_min_um, c.size_max_um = max(0.0, float(c.size_min_um)), max(0.0, float(c.size_max_um))
        if 0 < c.size_max_um < c.size_min_um:
            c.size_min_um, c.size_max_um = c.size_max_um, c.size_min_um
        v = float(c.clean_min)
        c.clean_min = min(max(v / 100.0 if v > 1.0 else v, 0.0), 1.0)      # (above 1: typed as a percentage)
        c.real_cut = min(max(float(c.real_cut), 0.01), 0.99)
        c.pictures_per_cm2 = max(1.0, float(c.pictures_per_cm2))
        c.thin_max_targets = max(0, int(c.thin_max_targets))
        c.material = str(c.material).lower().strip() or "hbn"
        return c

    def describe(self):
        c = self.clean()
        return (f"{c.material} {c.thickness_min_nm:g}-{c.thickness_max_nm:g} nm, flat area "
                + (f"{c.size_min_um:g}-{c.size_max_um:g} um" if c.size_max_um else f"{c.size_min_um:g} um or more") + " across, "
                + (f"at least {c.clean_min * 100:.0f} % clean" if c.clean_min else "any cleanliness")
                + f", real-flake probability >= {c.real_cut:g}")


def load(path) -> Criteria:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Criteria()
    names = {f.name for f in fields(Criteria)}
    return Criteria(**{k: v for k, v in d.items() if k in names})


def save(path, c: Criteria):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(c), indent=1), encoding="utf-8")


def scan_settings(c: Criteria, run_name: str) -> dict:
    """The original scan program's settings (flakeid/scope/settings.py) for one scan with these criteria: generous at
    10x (see the module text), every question that has an answer here answered, the pictures kept for sorting."""
    c = c.clean()
    d = dict(
        mode="multi", ask_mode=False, chip_type="polygon", run_name=run_name,
        material=c.material, oxide_nm=float(c.oxide_nm),
        ask_results_folder=False, copy_results_to=c.copy_to or "", ask_chip_notes=False,
        export_in_background=False,                 # the job ends when the analysis has: the sorting follows at once
        # 10x: every region that passes the 10x rules is a candidate, whatever its score; the flat area at least
        # SIZE_MARGIN of the smallest wanted size (the 10x outline of a small flake is rough), 6 um at the least
        target_min_score=0.0, min_targets=0,
        target_min_d_um=round(max(6.0, SIZE_MARGIN * c.size_min_um), 1),
        targets_per_cm2=float(c.pictures_per_cm2), targets_max_piece=0, hbn_max_targets=0, hbn_imaged_per_cm2=0.0,
        thin_hbn_targets=int(c.thin_max_targets) if c.thin_search and c.material == "hbn" else 0,
        faint_flakes=True,          # faint flat films the residue rules would drop (a 2 nm, 25 um flake on 20261117_1)
        # 100x: judged, nothing moved or deleted (collect.py sorts; "Sort again" needs the raw pictures)
        real100="judge", real100_cut=float(c.real_cut), flake_min_flat=0.0, flake_min_area_um2=0.0,
        keep_tiles=True,
    )
    if c.material == "hbn":
        top = MAX_NM
        if c.skip_thick_at_10x:
            top = min(MAX_NM, c.thickness_max_nm * THICK_MARGIN[0] + THICK_MARGIN[1])
        d["hbn_range_nm"] = [0.0, float(top)]        # (the thin end is never cut at 10x)
    return d


def tri(lo, hi, v_lo, v_hi=None):
    """"yes" / "no" for a value (or a range v_lo..v_hi: yes if it overlaps) against lo..hi (hi 0 or None: no limit);
    "maybe" for an unknown value."""
    if v_lo is None or v_lo != v_lo:
        return "maybe"
    v_hi = v_lo if v_hi is None or v_hi != v_hi else v_hi
    return "yes" if v_hi >= lo and (not hi or v_lo <= hi) else "no"


BIGGER = 1.5    # a wanted 100x area this many times wider than the 10x region heads the card (headline)


def headline(f: dict, c: Criteria):
    """The 100x area (collect.py regions100) whose numbers head a wanted flake's card and name its picture, or None for
    the 10x region's: the area the flake is wanted for, or a wanted area much larger than the 10x region (the 10x
    analysis saw only a small piece of the flake)."""
    if f.get("class") != "match":
        return None
    r = area100(f, c)
    if r is not None and (str(f.get("why", "")).startswith("wanted for an area") or r["size_um"] > BIGGER * (f.get("size_um") or 0)):
        return r
    return None


def area100(f: dict, c: Criteria):
    """The largest flat area of the flake's 100x picture (collect.py regions100) whose 100x-colour thickness and size are
    inside the criteria, or None."""
    for r in f.get("regions100") or ():
        if r.get("thickness_nm") is None:
            continue
        if (tri(c.thickness_min_nm, c.thickness_max_nm, r.get("thickness_min_nm", r["thickness_nm"]),
                r.get("thickness_max_nm", r["thickness_nm"])) == "yes"
                and tri(c.size_min_um, c.size_max_um, r["size_um"]) == "yes"):
            return r
    return None


def classify(f: dict, c: Criteria) -> tuple:
    """(class, why) of one flake record (collect.py) under the criteria: see the module text.  The same rule is written
    in the results page's script (page.py) for its boxes."""
    c = c.clean()
    p = f.get("p_real")
    if p is not None and p < min(DOUBT, c.real_cut):
        return "glue", f"sorter: real-flake probability {p:.2f}"
    if p is not None and p < c.real_cut:
        return "doubtful", f"sorter not sure: real-flake probability {p:.2f}"
    why = []
    t = f.get("thickness_nm")
    th = tri(c.thickness_min_nm, c.thickness_max_nm, f.get("thickness_min_nm") if t is not None else None, f.get("thickness_max_nm"))
    if th == "no" and any(c.thickness_min_nm <= a <= c.thickness_max_nm for a in f.get("alt_nm") or ()):
        th = "maybe"
        why.append("the same colour is also a thickness inside the window")
    elif th == "maybe":
        why.append("thickness not known")
    elif th == "no":
        why.append(f"thickness {t:.0f} nm")
    sz = tri(c.size_min_um, c.size_max_um, f.get("size_um"))
    if sz == "maybe":
        why.append("size not known")
    elif sz == "no":
        why.append(f"flat area {f['size_um']:.0f} um across")
    fl = f.get("flat_frac")
    cl = "yes" if not c.clean_min or (fl is not None and fl >= c.clean_min) else "no"
    if cl == "no":
        why.append(f"{(fl or 0) * 100:.0f} % clean")
    if (th, sz) != ("yes", "yes") and cl == "yes":
        # the 10x region does not (surely) fit, but the 100x picture shows a flat area that does (a faint thin flake
        # beside the thicker piece the 10x analysis saw)
        r = area100(f, c)
        if r is not None:
            return "match", (f"wanted for an area seen at 100x: {r['thickness_nm']:.1f} nm, {r['size_um']:.0f} um across, "
                             + ("at the cross" if r.get("at_cross") else f"{(r['dx_um'] ** 2 + r['dy_um'] ** 2) ** 0.5:.0f} um from the cross"))
    if "no" in (th, sz, cl):
        return "other", "; ".join(w for w in why if "not known" not in w and "same colour" not in w)
    if "maybe" in (th, sz):
        return "possible", "; ".join(why)
    return "match", ""
