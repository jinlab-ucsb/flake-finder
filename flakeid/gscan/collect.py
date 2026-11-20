"""After a scan: every 100x picture joined to what the 10x analysis found under its middle, judged real flake or glue,
its thickness checked against the 100x colour, and sorted by the criteria.  No microscope is needed: a finished scan
can be sorted again at any time.

    python -m flakeid.gscan.collect RUN_FILE [--criteria FILE] [--results DIR] [--terraces-from FOLDER] [--open]

RUN_FILE          the scan's run file (C:/FlakeScans/<name>_run_<n>.json), or its name without the folder
--results         the analysis results of that run (default <run folder>/results/<run name>)
--terraces-from   a results folder of another analysis of the same chip, not moved in between (its <piece> folders, or
                  one piece's folder used for every piece): a re-analysis with other settings

Writes <results>/sorted/: index.html (page.py), flakes.csv, flakes.json, img/, and the full-size colour pictures of the
wanted flakes in match/ and possible/.

How a picture gets its numbers (the steps done by hand on 2026-11-16, tools/bx53m/thickness_gallery.py):
  - the middle of a 100x picture is its stage position less the 10x -> 100x offset (the 100x list holds positions with
    the offset, the 10x regions without);
  - its region is the flat 10x region whose largest inscribed circle holds that point; thickness and size are that
    region's.  No such region: thickness and size are not known (never another region's);
  - the same region photographed more than once (provisional and final target list, two runs): one entry, the picture
    nearest the region's middle, a real one before a glue one;
  - real or glue: flakeid.real100 on the raw frame, with the run's substrate colour;
  - a region whose 10x colour fits several thicknesses: the one its 100x colour fits (colour100.py), when clear.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .. import real100
from . import colour100
from .criteria import Criteria, classify, headline

CLASSES = ("match", "possible", "other", "doubtful", "glue")
SPOT_UM = 20.0              # pictures without a region within this of each other: the same spot
DISAGREE = (15.0, 0.30)     # the 100x colour alone against a single 10x reading: further apart than 15 nm and 30 %
AIM_TOL_UM = 12.0           # the aimed region still counts with the cross this far outside its flat circle (re-centring)


def rows(p):
    with open(p, newline="", encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def _f(v, default=None):
    try:
        x = float(v)
        return x if x == x else default
    except (TypeError, ValueError):
        return default


def find_run(run, root="C:/FlakeScans"):
    p = Path(run)
    if not p.is_file():
        p = Path(root) / (p.name if p.suffix == ".json" else p.name + ".json")
    if not p.is_file():
        raise FileNotFoundError(f"no run file {run} (in {root})")
    return p


def runs(root="C:/FlakeScans"):
    """The run files of root that have 100x pictures, newest first: [(run name, path, pieces)]."""
    out = []
    for p in sorted(Path(root).glob("*_run_*.json"), key=lambda q: -q.stat().st_mtime):
        try:
            man = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        names = [q["name"] for q in man.get("pieces", []) if (p.parent / f"{q['name']}_100x").is_dir()]
        if names:
            out.append((p.stem, p, names))
    return out


def _piece_results(results, terraces_from, name):
    """(terraces csv, flakes json) of a piece, or (None, None)."""
    for base in filter(None, (terraces_from, results)):
        base = Path(base)
        for d in (base / name, base):
            t = next(iter(sorted(d.glob("*_live_terraces.csv"))), None) if d.is_dir() else None
            if t is not None:
                j = Path(str(t).replace("_terraces.csv", "_flakes.json"))
                return t, (j if j.is_file() else None)
    return None, None


def _display(raw, sub):
    """A colour picture from a raw frame (no colour copy was saved): the substrate at mid grey."""
    r, g, b = real100.channels(raw, 128.0)
    a = np.stack([b / max(sub[2], 1), g / max(sub[1], 1), r / max(sub[0], 1)], 2) * 110.0
    return np.clip(a, 0, 255).astype(np.uint8)


def collect_piece(root, name, results, terraces_from, offset, flat100, crit, img_dir, log=print):
    """The flake records of one piece (one per region / spot), its pictures written to img_dir."""
    root = Path(root)
    h = root / f"{name}_100x"
    idx = next(iter(h.glob("*_100x_index.csv")), None)
    if idx is None:
        log(f"{name}: no 100x pictures")
        return []
    tcsv, fjson = _piece_results(results, terraces_from, name)
    T = rows(tcsv) if tcsv else []
    T = [t for t in T if _f(t.get("stage_x_um")) is not None and _f(t.get("inscribed_diameter_um")) is not None]
    P = np.array([[float(t["stage_x_um"]), float(t["stage_y_um"])] for t in T]).reshape(-1, 2)
    D = np.array([float(t["inscribed_diameter_um"]) for t in T])
    meta = {}
    try:
        meta = json.loads(Path(fjson).read_text(encoding="utf-8")) if fjson else {}
    except (OSError, ValueError):
        pass
    material = str(meta.get("material") or crit.material)
    oxide = _f(meta.get("oxide_nm_used")) or _f(meta.get("oxide_nm")) or crit.oxide_nm
    if not T:
        log(f"{name}: no table of 10x regions ({results}): the pictures are judged, without thickness or size")
    aimed = {}
    for f in (root / f"{name}_targets_partial.csv", root / f"{name}_targets.csv"):
        if f.is_file():
            aimed.update({r["label"]: r for r in rows(f) if r.get("label")})
    # pass 1: each frame's commonest colour (the run's substrate) and the colour of its middle
    flat, dark = colour100.load_flat(colour100.pick_flat(flat100, oxide)) if flat100 else (None, None)
    I, t0 = [], time.time()
    for r in rows(idx):
        f = next((d / r["file"] for d in (h, h / "rejected") if (d / r["file"]).is_file()), None)
        raw = cv2.imread(str(f), cv2.IMREAD_UNCHANGED) if f else None
        if raw is None or raw.ndim != 2:
            continue                            # (a colour picture instead of a raw frame: not this camera's raw mode)
        I.append(dict(row=r, file=f, mode=real100.frame_mode(raw), meas=colour100.measure(raw, flat, dark)))
    if not I:
        log(f"{name}: no raw 100x frames")
        return []
    sub = np.median(np.array([i["mode"] for i in I]), axis=0)
    chooser = None
    if T and flat is not None:
        try:
            chooser = colour100.Chooser(material, oxide, np.median(np.array([i["meas"]["sub"] for i in I]), axis=0))
        except Exception as e:
            log(f"{name}: no 100x colour check ({type(e).__name__}: {e})")
    # pass 2: real or glue, the region, the pictures
    model = real100.load_model()
    E = []
    for i in I:
        r, f = i["row"], i["file"]
        raw = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        ft = real100.features(raw, sub)
        order = int(_f(r.get("order"), 0))
        key = f"{name}_{order:03d}"
        e = dict(piece=name, order=order, label=r.get("label", ""), file=f.name, key=key,
                 p_real=round(real100.p_real(ft, model), 3), flat_frac=round(ft["flat_frac"], 3), mottle=round(ft["mottle"], 4),
                 cover=round(ft["cover"], 3), stage_x_um=_f(r.get("stage_x_um")), stage_y_um=_f(r.get("stage_y_um")),
                 z_um=_f(r.get("z_um")), focus=r.get("focus", ""), notes=[], alt_nm=[])
        if e["stage_x_um"] is None:
            continue
        q = np.array([e["stage_x_um"] - offset[0], e["stage_y_um"] - offset[1]])
        e["x10_um"], e["y10_um"] = round(float(q[0]), 1), round(float(q[1]), 1)
        j = None
        if len(T):
            dist = np.hypot(*(P - q).T)
            inside = np.flatnonzero(dist < D / 2)
            if len(inside):
                j = int(inside[np.argmin(dist[inside])])
                e["dist_um"] = round(float(dist[j]), 1)
        a = aimed.get(e["label"])
        if a:
            e["aimed"] = dict(thickness_nm=_f(a.get("thickness_nm")), size_um=_f(a.get("terrace_d_um")), flake_id=a.get("flake_id"))
            if j is None and len(T):
                # small regions: the 100x re-centring (up to ~10 um) can put the cross just outside the region's inscribed
                # circle. Then take the region the scan aimed this picture at (same flake, same inscribed diameter).
                d_aim = _f(a.get("terrace_d_um"))
                c = [k for k in range(len(T)) if T[k].get("flake_id") == a.get("flake_id") and d_aim is not None
                     and abs(D[k] - d_aim) < 0.01]
                if len(c) == 1 and dist[c[0]] < D[c[0]] / 2 + AIM_TOL_UM:
                    j = c[0]
                    e["dist_um"] = round(float(dist[j]), 1)
                    e["notes"].append(f"the cross is {dist[j] - D[j] / 2:.0f} um outside this region's flat circle "
                                      "(the region the scan aimed at): check the picture")
        e["j"] = j
        if j is not None:
            t = T[j]
            e.update(region=t["terrace_id"], flake_id=t["flake_id"], thickness_nm=_f(t.get("thickness_nm")),
                     thickness_min_nm=_f(t.get("thickness_min_nm")), thickness_max_nm=_f(t.get("thickness_max_nm")),
                     alt_nm=[float(x) for x in str(t.get("alt_thickness_nm") or "").split(";") if _f(x) is not None],
                     size_um=_f(t.get("inscribed_diameter_um")), area_um2=_f(t.get("area_um2")), quality=t.get("quality", ""),
                     contrast=[_f(t.get("contrast_" + k)) for k in "rgb"], fit_residual=_f(t.get("fit_residual")),
                     thickness_10x_nm=_f(t.get("thickness_nm")))
            if chooser is not None and e["thickness_nm"] is not None:
                ch = chooser.choose(i["meas"], [e["thickness_nm"]] + e["alt_nm"])
                if ch:
                    e["colour100"] = ch
                    if e["alt_nm"] and ch.get("clear"):
                        if ch["pick"] != e["thickness_nm"]:
                            old = e["thickness_nm"]
                            w = max(2.0, 0.05 * ch["pick"])
                            e.update(thickness_nm=ch["pick"], thickness_min_nm=ch["pick"] - w, thickness_max_nm=ch["pick"] + w)
                            e["notes"].append(f"thickness from the 100x colour (the 10x colour read {old:.0f} nm, or this)")
                        else:
                            e["notes"].append("100x colour agrees (not " + ", ".join(f"{x:.0f}" for x in e["alt_nm"]) + " nm)")
                        e["alt_ruled_out_nm"], e["alt_nm"] = [x for x in [e["thickness_10x_nm"]] + e["alt_nm"] if x != e["thickness_nm"]], []
                    elif not e["alt_nm"] and abs(ch["alone_nm"] - e["thickness_nm"]) > max(DISAGREE[0], DISAGREE[1] * e["thickness_nm"]):
                        e["alt_nm"] = [ch["alone_nm"]]
                        e["notes"].append(f"the 100x colour alone reads {ch['alone_nm']:.0f} nm: check by eye")
        if chooser is not None:
            try:
                e["regions100"] = [dict(r, thickness_min_nm=round(colour100.region_range(r)[0], 1),
                                        thickness_max_nm=round(colour100.region_range(r)[1], 1))
                                   if r["thickness_nm"] is not None else r
                                   for r in colour100.regions(raw, flat, dark, chooser)[:6]]
            except Exception as ex:                 # (the 10x region still sorts the picture)
                log(f"{key}: no 100x areas ({type(ex).__name__}: {ex})")
        # the pictures
        col = next((d / (f.name[:-4] + s) for d in (h, h / "rejected") for s in ("_colour.png", "_colour.tif")
                    if (d / (f.name[:-4] + s)).is_file()), None)
        im = cv2.imread(str(col), cv2.IMREAD_COLOR) if col else None
        if im is None:
            im = _display(raw, sub)
        e["colour_file"] = str(col) if col else ""
        small = cv2.resize(im, (800, 600), interpolation=cv2.INTER_AREA)
        cv2.drawMarker(small, (400, 300), (255, 255, 255), cv2.MARKER_CROSS, 22, 1)
        cv2.imwrite(str(img_dir / f"{key}_100x.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 90])
        crop = root / f"{name}_targets_crops" / f"{e['label']}.jpg"
        if crop.is_file():
            shutil.copyfile(str(crop), str(img_dir / f"{key}_10x.jpg"))
            e["crop"] = True
        e["q"] = q
        E.append(e)
    # one entry per region; pictures without a region: one per spot
    groups = {}
    for e in E:
        if e["j"] is not None:
            groups.setdefault(e["j"], []).append(e)
        else:
            k = next((k for k, g in groups.items() if isinstance(k, tuple) and np.hypot(*(g[0]["q"] - e["q"])) < SPOT_UM), None)
            groups.setdefault(k if k is not None else ("spot", len(groups)), []).append(e)
    out = []
    for g in groups.values():
        g.sort(key=lambda e: (e["p_real"] < crit.real_cut, e.get("dist_um", 0.0)))
        e = g[0]
        e["also"] = [f"{x['piece']} #{x['order']:03d}" for x in g[1:]]
        for k in ("j", "q"):
            e.pop(k, None)
        out.append(e)
    log(f"{name}: {len(I)} 100x pictures, {len(out)} different places ({sum(1 for e in out if e.get('region'))} on a 10x region), "
        f"oxide {oxide:.1f} nm" + ("" if chooser else "; no 100x colour check") + f"; {time.time() - t0:.0f} s")
    return out


CSV_COLUMNS = ("class", "why", "piece", "order", "label", "thickness_nm", "thickness_min_nm", "thickness_max_nm", "alt_nm", "size_um",
               "area_um2", "flat_frac", "p_real", "stage_x_um", "stage_y_um", "z_um", "x10_um", "y10_um", "region", "flake_id", "quality",
               "thickness_10x_nm", "areas_100x", "notes", "also", "file", "picture")


def collect(run, crit: Criteria, root=None, results=None, terraces_from=None, offset=(21.1, -38.4), flat100=(), pieces=None,
            out=None, log=print):
    """Sort a finished scan.  Returns (the output folder, the records)."""
    crit = crit.clean()
    run = find_run(run, root or "C:/FlakeScans")
    root = Path(root) if root else run.parent
    man = json.loads(run.read_text(encoding="utf-8"))
    names = [p["name"] for p in man.get("pieces", []) if pieces is None or p["name"] in pieces]
    results = Path(results) if results else root / "results" / run.stem
    out = Path(out) if out else results / "sorted"
    if not terraces_from:                       # (sorted before from a re-analysis: the same one again)
        try:
            prev = json.loads((out / "flakes.json").read_text(encoding="utf-8")).get("terraces_from") or ""
            if prev and Path(prev).is_dir():
                terraces_from = prev
                log(f"10x regions from {prev} (as when this scan was sorted before)")
        except (OSError, ValueError):
            pass
    img = out / "img"
    if out.is_dir():
        for d in ("img", "match", "possible"):
            shutil.rmtree(out / d, ignore_errors=True)
    img.mkdir(parents=True, exist_ok=True)
    log(f"sorting {run.stem}: {crit.describe()}")
    F = []
    for name in names:
        F += collect_piece(root, name, results, terraces_from, offset, flat100, crit, img, log)
    for f in F:
        f["class"], f["why"] = classify(f, crit)
    order = {c: k for k, c in enumerate(CLASSES)}
    F.sort(key=lambda f: (order[f["class"]], f.get("thickness_nm") if f.get("thickness_nm") is not None else 1e9, f["piece"], f["order"]))
    # the wanted flakes' full-size colour pictures: <thickness>nm_<size>um_<piece>_<n>_x<stage x>_y<stage y>.png
    for f in F:
        f["picture"] = ""
        f["areas_100x"] = "; ".join(
            f"{'?' if r['thickness_nm'] is None else format(r['thickness_nm'], '.1f')} nm {r['size_um']:.0f} um"
            f"{' (at the cross)' if r['at_cross'] else ''}" for r in f.get("regions100") or ())
        if f["class"] in ("match", "possible") and f.get("colour_file"):
            a = headline(f, crit)
            thk, siz = (a["thickness_nm"], a["size_um"]) if a else (f.get("thickness_nm"), f.get("size_um"))
            t = f"{thk:05.1f}nm" if thk is not None else "unknown"
            s = f"_{siz:.0f}um" if siz is not None else ""
            dst = out / f["class"] / f"{t}{s}_{f['key']}_x{f['stage_x_um']:.0f}_y{f['stage_y_um']:.0f}.png"
            dst.parent.mkdir(exist_ok=True)
            im = cv2.imread(f["colour_file"], cv2.IMREAD_COLOR)
            if im is not None and cv2.imwrite(str(dst), im):
                f["picture"] = f"{f['class']}/{dst.name}"
    n = {c: sum(1 for f in F if f["class"] == c) for c in CLASSES}
    with open(out / "flakes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        for f in F:
            w.writerow([";".join(map(str, v)) if isinstance(v, list) else ("" if v is None else v) for v in (f.get(k) for k in CSV_COLUMNS)])
    info = dict(run=run.stem, sorted=time.strftime("%Y-%m-%d %H:%M"), criteria=asdict(crit), counts=n, pieces=names,
                offset_100x_um=list(offset), results=str(results), terraces_from=str(terraces_from or ""))
    (out / "flakes.json").write_text(json.dumps(dict(info, flakes=F), indent=1), encoding="utf-8")
    from .page import write_page
    write_page(out, info, F)
    log(f"{run.stem}: " + ", ".join(f"{n[c]} {c}" for c in CLASSES) + f"  ->  {out / 'index.html'}")
    return out, F


def main(argv=None):
    import argparse
    from . import criteria as C
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--root", default=None, help="the scans' folder (default: the run file's, else C:/FlakeScans)")
    ap.add_argument("--criteria", help="a criteria file (default: <root>/scanner_criteria.json, the window's)")
    ap.add_argument("--results")
    ap.add_argument("--terraces-from")
    ap.add_argument("--pieces", nargs="+")
    ap.add_argument("-o", "--output")
    ap.add_argument("--settings", help="the scan settings file (the 10x -> 100x offset, the 100x flat fields)")
    ap.add_argument("--open", action="store_true", help="open the page when done")
    a = ap.parse_args(argv)
    run = find_run(a.run, a.root or "C:/FlakeScans")
    root = Path(a.root) if a.root else run.parent
    crit = C.load(a.criteria or root / "scanner_criteria.json")
    offset, flat100 = scan_calibration(a.settings)
    out, _ = collect(run, crit, root=root, results=a.results, terraces_from=a.terraces_from, offset=offset, flat100=flat100,
                     pieces=a.pieces, out=a.output)
    if a.open:
        import webbrowser
        webbrowser.open((out / "index.html").as_uri())
    return 0


def scan_calibration(settings_file=None):
    """(10x -> 100x offset, 100x flat fields) from the scan settings file (default: scan_settings.json in the program
    folder; without one: the glovebox microscope's defaults)."""
    from ..scope.settings import load
    p = Path(settings_file) if settings_file else Path(__file__).resolve().parents[2] / "scan_settings.json"
    try:
        s = load(str(p)) if p.is_file() else load(None, hardware="bx53m")
    except Exception:
        s = load(None, hardware="bx53m")
    return tuple(float(v) for v in s.offset_100x_um), tuple(s.flat100 or ())


if __name__ == "__main__":
    sys.exit(main())
