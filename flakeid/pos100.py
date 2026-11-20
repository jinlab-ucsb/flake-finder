"""Flake positions from the 100x pictures (Avram, 2026-10-29): every flake imaged at 100x gets the stage position where
its 100x picture found it, in place of the one read from the 10x picture.

Why: the 10x picture is stitched from tiles placed at the positions the stage was sent to.  When a tile was not where
the stage reported (20261027_run_13: whole bands of tiles 80-150 um off in Y), every flake read from it is off by as
much.  The 100x imaging finds each target again (it matches the 100x frame against the target's 10x picture and moves
onto it, scan.Scanner.centre_on_target), so the stage position of its 100x picture is a direct measurement: run_13's
100x pictures put 434 flakes within a median 1.4 um of what that re-centring measured, while the 10x picture had them
up to 150 um away.

Measured here, after the 100x export (hundred.export): each 100x picture (<viewer>/100x/<file>.jpg) against its target's
10x picture (<piece>_targets_crops/<label>.jpg, the one the microscope re-centred on; else cut from the viewer's tiles):
the target's offset d from the frame centre (retarget.match_crop), so its stage position is

    100x stage position - 10x->100x offset + M d          (M: the 10x camera's um per pixel)

and its correction the difference to the 10x position of its target row.  A flake's correction is the median of its
targets'.  The flake of a picture is the one hundred.attach gave it (the viewer's H100, by position: a provisional
list's labels number the flakes differently).  A flake whose picture does not match confidently (and every flake not
imaged at 100x) gets the median of the measured flakes of its own 10x tile when at least 2 agree and it is over 10 um
(a tile is placed as a whole: its flakes share its error); else it keeps its position from the 10x picture.

Applied to every stage coordinate of the flake read from the 10x picture: the viewer's flakes and terraces (sx, sy), the
flake / terrace tables, the target lists and their .pos files.  The 100x pictures' own positions (100x index, regions,
mosaics) are measurements already and stay; nothing in pixels changes.  Each changed row keeps the correction it holds
(pos_dx_um, pos_dy_um, pos_from: "100x" measured / "10x tile" estimated), so the position read from the 10x picture is
always stage - pos_d, and doing it again (a second export, a re-image) only applies what changed.  The measurements:
<piece>_positions100.csv next to the flake tables.

    python -m flakeid.pos100 <piece>_100x --viewer <viewer>      (a piece exported before: measure and apply)
"""
from __future__ import annotations

from flakeid.hwconst import OFFSET_100X, RATIO_100X  # noqa: F401  (the microscope's 100x ratio and offset)

import csv
import io
import json
import math
import re
from pathlib import Path

import numpy as np

MIN_PEAK = 0.5            # correlation a 100x picture needs against its target's 10x picture (as the re-centring:
MAX_SECOND = 0.85         # scope.settings recentre_min_peak / recentre_max_second)
AGREE_UM = 15.0           # the measurements of one flake (or of one tile's flakes) must agree this well
MAX_UM = 300.0            # a larger correction is not believed (the re-centring itself moves at most 150 um)
TILE_MIN = 2              # a tile's estimate needs this many measured flakes in it ...
TILE_MIN_UM = 10.0        # ... and is only applied when larger than this (smaller: the 10x position is as good)
CROP_HALF = 180           # a target's 10x picture: 2 x this 10x pixels across (live.write_target_crops)
SAME_UM = 0.1             # a change smaller than this is none (the viewer keeps positions to 0.1 um: doing it again
                          # measures from them and can differ by a few hundredths)
REPORT = "positions100.csv"
POS_COLS = ("pos_dx_um", "pos_dy_um", "pos_from")
SKIP = ("_100x_index", "_100x_mosaic", "_domains_", "_scan.pos", "_remeasure.csv", REPORT)


def _flake_of(label):
    """(flake id, thickness nm) of a target label such as 16_F143_65nm, or (None, None)."""
    m = re.search(r"_F(\d+)(?:_(\d+(?:\.\d+)?)nm)?", label or "")
    return (int(m.group(1)), float(m.group(2)) if m.group(2) else None) if m else (None, None)


def _num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def piece_name(folder100):
    """<piece>_100x, <piece>_retake2_100x -> <piece>."""
    return re.sub(r"(_retake\d+)?_100x$", "", Path(folder100).name)


def data_dirs(folder100, vdir):
    """The folders that hold a piece's tables: the run folder (target lists, .pos), the results folder (the viewer's
    parent: flake and terrace tables) and, in the share's layout, <piece>/data."""
    out = []
    for d in (Path(folder100).parent, Path(vdir).parent, Path(vdir) / "data"):
        if d.is_dir() and d.resolve() not in [q.resolve() for q in out]:
            out.append(d)
    return out


def piece_files(dirs, name):
    """The piece's files that may hold its flakes' stage coordinates (tables, target lists, .pos, the flake json)."""
    out = []
    for d in dirs:
        for f in sorted(d.iterdir()):
            nm = f.name
            if not f.is_file() or not nm.startswith(name + "_") or any(s in nm for s in SKIP):
                continue
            if nm.endswith(".csv") or nm.endswith(".pos") or nm.endswith("_live_flakes.json"):
                out.append(f)
    return out


def _read_csv(path):
    raw = Path(path).read_bytes().decode("utf-8-sig")
    rd = csv.DictReader(io.StringIO(raw))
    return list(rd.fieldnames or []), list(rd), ("\r\n" if "\r\n" in raw else "\n")


def target_rows(files):
    """label -> (flake id, (stage x, y of the target as read from the 10x picture, in its list's frame (10x + the 100x
    offset))), from the target lists (the corrections they already hold taken off)."""
    out = {}
    for f in files:
        if not f.name.endswith(".csv"):
            continue
        try:
            fields, rows, _ = _read_csv(f)
        except (OSError, ValueError):
            continue
        if not {"label", "stage_x_um", "stage_y_um"} <= set(fields):
            continue
        for r in rows:
            x, y = _num(r.get("stage_x_um")), _num(r.get("stage_y_um"))
            if x is None or y is None or not r.get("label") or r["label"] in out:
                continue
            dx, dy = _num(r.get("pos_dx_um")) or 0.0, _num(r.get("pos_dy_um")) or 0.0
            fid = _num(r.get("flake_id"))
            out[r["label"]] = (int(fid) if fid is not None else None, (x - dx, y - dy))
    return out


def _crop_file(path, frame_wh):
    """A target's 10x picture as grey 0..1, the marks live.write_target_crops draws on it painted out (as
    scan.Scanner._crop_grey)."""
    import cv2
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    fw, fh = frame_wh[0] / RATIO_100X, frame_wh[1] / RATIO_100X
    marks = np.zeros((h, w), np.uint8)
    cv2.rectangle(marks, (int(w / 2 - fw / 2), int(h / 2 - fh / 2)), (int(w / 2 + fw / 2), int(h / 2 + fh / 2)), 1, 6)
    cv2.drawMarker(marks, (w // 2, h // 2), 1, cv2.MARKER_CROSS, 18, 5)
    bgr = cv2.inpaint(bgr, marks, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def _crop_canvas(cv, s10):
    """The same picture cut from the viewer's tiles around 10x stage position s10 (no crop file): (picture, the 10x
    stage position of its centre (a whole pixel: up to half a pixel from s10)), or (None, None)."""
    p = np.round(cv.to_px([s10])[0])
    g = cv.region(int(p[0]) - CROP_HALF, int(p[1]) - CROP_HALF, 2 * CROP_HALF, 2 * CROP_HALF)
    if np.isnan(g).mean() > 0.3:
        return None, None
    return np.nan_to_num(g, nan=float(np.nanmedian(g))), cv.to_stage([p])[0]


def pictures(vdir):
    """{picture stem: flake id} from the viewer's 100x data (const H100, written by hundred.attach, which matched each
    picture to its flake by its position: a provisional list's labels number the flakes differently)."""
    html = (Path(vdir) / "index.html").read_text(encoding="utf-8")
    m = re.search(r"const H100=(.*?);\n", html, re.S)
    try:
        H = json.loads(m.group(1)) if m else {}
    except ValueError:
        H = {}
    out = {}
    for fid, es in (H or {}).items():
        for e in es:
            if not e.get("domain") and e.get("f"):
                out[Path(e["f"]).stem] = int(fid)
    return out


def measure(folder100, vdir, offset=None, log=print):
    """Each 100x picture of the piece against its target's 10x picture.  Returns rows, one per picture: dict(label,
    flake_id, file, peak, second, ok, dx, dy (the target's correction, um; None when not measured), how)."""
    import cv2
    from .hundred import OFFSET_100X, _M100
    from .scope.remeasure import Canvas
    from .scope.retarget import match_crop
    folder100, vdir = Path(folder100), Path(vdir)
    idx = next(folder100.glob("*_100x_index.csv"), None)
    if idx is None:
        return []
    name = piece_name(folder100)
    off = np.asarray(offset if offset is not None else OFFSET_100X, float)
    lists = target_rows(piece_files(data_dirs(folder100, vdir), name))
    crops = next((d / f"{name}_targets_crops" for d in data_dirs(folder100, vdir) if (d / f"{name}_targets_crops").is_dir()), None)
    cv = Canvas(vdir)
    M10 = cv.M
    px100 = float(np.sqrt(abs(np.linalg.det(_M100()))))
    pics = pictures(vdir)
    terr = {}                                       # flake id -> its terraces' 10x positions in the viewer
    T = cv.D.get("terraces")
    for e in (T.values() if isinstance(T, dict) else T or []):
        if e.get("sx") is not None and e.get("sy") is not None and e.get("fl") is not None:
            p = e.get("pos") or [0.0, 0.0]
            terr.setdefault(e["fl"], []).append((e["sx"] - p[0], e["sy"] - p[1]))
    _, rows, _ = _read_csv(idx)
    out = []
    for r in rows:
        lab = r.get("label") or ""
        stem = Path(r.get("file", "")).stem
        fid = pics.get(stem)                        # (the flake the picture was attached to; its label's number else)
        if fid is None and not pics:
            fid = _flake_of(lab)[0]
        rec = dict(label=lab, flake_id=fid, file=r.get("file", ""), peak=None, second=None, ok=False, dx=None, dy=None, how="")
        out.append(rec)
        x, y = _num(r.get("stage_x_um")), _num(r.get("stage_y_um"))
        if fid is None or x is None or y is None:
            rec["how"] = "not attached to a flake" if fid is None else "no stage position"
            continue
        listed = lab in lists and lists[lab][0] == fid
        if listed:
            s10 = np.asarray(lists[lab][1], float) - off              # the target's 10x position (10x frame)
        elif terr.get(fid):
            # a label of a list rewritten since (a provisional list's first block, its own flake numbers): the
            # flake's terrace nearest the picture, and its 10x picture cut there (the crop file is the old row's)
            here = np.array([x, y]) - off
            s10 = np.asarray(min(terr[fid], key=lambda q: math.hypot(q[0] - here[0], q[1] - here[1])), float)
        else:
            rec["how"] = "not in the target lists"
            continue
        pic = vdir / "100x" / f"{stem}.jpg"
        reg = vdir / "100x" / f"{stem}_regions.json"
        wh, pu = (1376, 1038), px100
        try:
            R = json.loads(reg.read_text(encoding="utf-8")) if reg.is_file() else {}
            wh, pu = tuple(R.get("frame_wh") or wh), float(R.get("px_um") or pu)
        except (OSError, ValueError):
            pass
        img = cv2.imread(str(pic))
        if img is None:
            rec["how"] = "no 100x picture"
            continue
        use_file = listed and crops is not None and (crops / f"{lab}.jpg").is_file()
        if use_file:                                # (centred on the row's position, to the pixel)
            crop, centre = _crop_file(crops / f"{lab}.jpg", wh), s10
        else:
            crop, centre = _crop_canvas(cv, s10)
        if crop is None:
            rec["how"] = "no 10x picture of the target"
            continue
        k = (wh[0] * pu / img.shape[1]) / cv.um_px                    # the picture's pixels -> 10x pixels
        small = cv2.resize(img.astype(np.float32).mean(axis=2) / 255.0, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        m = match_crop(small, crop, um_px=cv.um_px)
        if m is None:
            rec["how"] = "no structure to match"
            continue
        rec.update(peak=round(m["peak"], 3), second=round(m["second"], 3))
        if m["peak"] < MIN_PEAK or m["second"] > MAX_SECOND * m["peak"]:
            rec["how"] = "no confident match"
            continue
        true = np.array([x, y]) - off + M10 @ np.asarray(m["d"], float)     # (where the picture's centre really is)
        c = true - centre
        if math.hypot(*c) > MAX_UM:
            rec.update(dx=round(float(c[0]), 2), dy=round(float(c[1]), 2), how=f"more than {MAX_UM:g} um: not used")
            continue
        rec.update(ok=True, dx=round(float(c[0]), 2), dy=round(float(c[1]), 2), how="100x")
    return out


def corrections(recs, vdir, log=print):
    """Per flake: dict(dx, dy, how, n, spread) from the pictures' measurements; flakes without one take their 10x tile's
    (the median of its measured flakes, when they agree within AGREE_UM)."""
    from .scope.remeasure import Canvas, read_viewer
    by = {}
    for r in recs:
        if r["ok"]:
            by.setdefault(r["flake_id"], []).append((r["dx"], r["dy"]))
    corr = {}
    for fid, v in by.items():
        a = np.array(v, float)
        med = np.median(a, axis=0)
        spread = float(np.hypot(*(a - med).T).max())
        corr[fid] = dict(dx=round(float(med[0]), 2), dy=round(float(med[1]), 2), how="100x", n=len(a), spread=round(spread, 1))
        if spread > AGREE_UM:
            log(f"  F{fid}: its {len(a)} 100x pictures disagree by up to {spread:.0f} um: the median is used")
    # the 10x tile of every flake (the tile whose centre is nearest its target pixel): a tile is placed as a whole
    D, _ = read_viewer(vdir)
    cv = Canvas(vdir)
    tile = {}
    for f in D.get("flakes", []):
        px = (f.get("tx"), f.get("ty")) if f.get("tx") is not None else (f.get("x"), f.get("y"))
        if px[0] is None or px[1] is None:
            continue
        tile[f["id"]] = int(np.argmin(((cv.C - np.array(px, float)) ** 2).sum(1)))
    per = {}
    for fid, c in corr.items():
        if fid in tile:
            per.setdefault(tile[fid], []).append((c["dx"], c["dy"]))
    est = {}
    for t, v in per.items():
        a = np.array(v, float)
        med = np.median(a, axis=0)
        if len(a) >= TILE_MIN and float(np.hypot(*(a - med).T).max()) <= AGREE_UM and math.hypot(*med) > TILE_MIN_UM:
            est[t] = (round(float(med[0]), 2), round(float(med[1]), 2), len(a))
    for fid, t in tile.items():
        if fid not in corr and t in est:
            corr[fid] = dict(dx=est[t][0], dy=est[t][1], how="10x tile", n=est[t][2], spread=None)
    return corr


# ---- applying the corrections (each row keeps the one it holds: only the change is applied)
def _delta(fields_get, c):
    """(ddx, ddy) to apply to a row holding correction fields_get(pos_dx_um, ...) to give it correction c; None: no
    change (less than SAME_UM, same source)."""
    ox, oy = _num(fields_get("pos_dx_um")) or 0.0, _num(fields_get("pos_dy_um")) or 0.0
    d = (c["dx"] - ox, c["dy"] - oy)
    if math.hypot(*d) < SAME_UM and fields_get("pos_from") == c["how"]:
        return None
    return d


def _fmt(x, like):
    s = str(like)
    nd = max(2, len(s.split(".")[1])) if "." in s else 2
    return f"{x:.{min(nd, 6)}f}"


def apply_csv(path, corr):
    """Rows of corrected flakes (flake_id): stage_x_um / stage_y_um and the terrace_poly_um outline.  Returns the
    number of rows changed (the file is written only then)."""
    fields, rows, nl = _read_csv(path)
    if not {"flake_id", "stage_x_um", "stage_y_um"} <= set(fields):
        return 0
    n = 0
    for r in rows:
        try:
            c = corr.get(int(float(r.get("flake_id") or "nan")))
        except ValueError:
            c = None
        x, y = _num(r.get("stage_x_um")), _num(r.get("stage_y_um"))
        if c is None or x is None or y is None:
            continue
        d = _delta(r.get, c)
        if d is None:
            continue
        ddx, ddy = d
        r["stage_x_um"], r["stage_y_um"] = _fmt(x + ddx, r["stage_x_um"]), _fmt(y + ddy, r["stage_y_um"])
        if r.get("terrace_poly_um"):
            pts = []
            for q in r["terrace_poly_um"].split(";"):
                a = q.split()
                if len(a) == 2 and _num(a[0]) is not None and _num(a[1]) is not None:
                    pts.append(f"{float(a[0]) + ddx:.1f} {float(a[1]) + ddy:.1f}")
            r["terrace_poly_um"] = ";".join(pts)
        r["pos_dx_um"], r["pos_dy_um"], r["pos_from"] = f"{c['dx']:.2f}", f"{c['dy']:.2f}", c["how"]
        n += 1
    if n:
        for k in POS_COLS:
            if k not in fields:
                fields.append(k)
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=fields, lineterminator=nl)
        w.writeheader()
        w.writerows(rows)
        with open(path, "w", encoding="utf-8", newline="") as fh:      # (Path.write_text has no newline= before 3.10)
            fh.write(out.getvalue())
    return n


def apply_live_json(path, corr):
    """<piece>_tiles_live_flakes.json: its flakes and terraces (flake_id)."""
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    n = 0
    for k in ("flakes", "terraces"):
        for r in d.get(k) or []:
            c = corr.get(r.get("flake_id"))
            if c is None or _num(r.get("stage_x_um")) is None or _num(r.get("stage_y_um")) is None:
                continue
            d_ = _delta(r.get, c)
            if d_ is None:
                continue
            r["stage_x_um"], r["stage_y_um"] = r["stage_x_um"] + d_[0], r["stage_y_um"] + d_[1]
            r["pos_dx_um"], r["pos_dy_um"], r["pos_from"] = c["dx"], c["dy"], c["how"]
            n += 1
    if n:
        p.write_text(json.dumps(d), encoding="utf-8")
    return n


def apply_pos(path, corr):
    """A target list's Micro-Manager .pos: the positions whose Properties name a corrected flake_id."""
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    n = 0
    for e in ((d.get("map") or {}).get("StagePositions") or {}).get("array") or []:
        props = (e.get("Properties") or {}).get("scalar") or {}
        try:
            c = corr.get(int(float(((props.get("flake_id") or {}).get("scalar")) or "nan")))
        except ValueError:
            c = None
        if c is None:
            continue
        d_ = _delta(lambda k: (props.get(k) or {}).get("scalar"), c)
        if d_ is None:
            continue
        ddx, ddy = d_
        for dp in (e.get("DevicePositions") or {}).get("array") or []:
            a = (dp.get("Position_um") or {}).get("array")
            if "XY" in str((dp.get("Device") or {}).get("scalar", "")).upper() and isinstance(a, list) and len(a) == 2:
                dp["Position_um"]["array"] = [round(a[0] + ddx, 4), round(a[1] + ddy, 4)]
        props.update(pos_dx_um=dict(type="STRING", scalar=f"{c['dx']:.2f}"), pos_dy_um=dict(type="STRING", scalar=f"{c['dy']:.2f}"),
                     pos_from=dict(type="STRING", scalar=c["how"]))
        e.setdefault("Properties", {"type": "PROPERTY_MAP"})["scalar"] = props
        n += 1
    if n:
        p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return n


def apply_viewer(vdir, corr, note):
    """The viewer's flakes and terraces (sx, sy); pos = [dx, dy, how] on each, and a summary (D.pos100)."""
    from .scope.remeasure import read_viewer, write_viewer_data
    D, _ = read_viewer(vdir)
    n = 0
    T = D.get("terraces")
    items = [(f, f.get("id")) for f in D.get("flakes", [])] + [(e, e.get("fl")) for e in (T.values() if isinstance(T, dict) else T or [])]
    for e, fid in items:
        c = corr.get(fid)
        if c is None or e.get("sx") is None or e.get("sy") is None:
            continue
        old = e.get("pos") or [0.0, 0.0, ""]
        if math.hypot(c["dx"] - old[0], c["dy"] - old[1]) < SAME_UM and old[2] == c["how"]:
            continue
        e["sx"], e["sy"] = round(e["sx"] + c["dx"] - old[0], 1), round(e["sy"] + c["dy"] - old[1], 1)
        e["pos"] = [c["dx"], c["dy"], c["how"]]
        n += 1
    if n or D.get("pos100") is None:
        D["pos100"] = note
        write_viewer_data(vdir, D)
    return n


def write_report(path, recs, corr):
    """<piece>_positions100.csv: one row per 100x picture (its measurement), then the flakes estimated from their tile."""
    cols = ["flake_id", "label", "file", "peak", "second", "picture_dx_um", "picture_dy_um", "picture", "flake_dx_um",
            "flake_dy_um", "flake_from", "flake_pictures", "flake_spread_um"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\r\n")
        w.writeheader()
        seen = set()
        for r in recs:
            c = corr.get(r["flake_id"]) or {}
            seen.add(r["flake_id"])
            w.writerow(dict(flake_id=r["flake_id"], label=r["label"], file=r["file"], peak=r["peak"], second=r["second"],
                            picture_dx_um=r["dx"], picture_dy_um=r["dy"], picture=r["how"], flake_dx_um=c.get("dx"),
                            flake_dy_um=c.get("dy"), flake_from=c.get("how"), flake_pictures=c.get("n"), flake_spread_um=c.get("spread")))
        for fid in sorted(k for k in corr if k not in seen):
            c = corr[fid]
            w.writerow(dict(flake_id=fid, flake_dx_um=c["dx"], flake_dy_um=c["dy"], flake_from=c["how"], flake_pictures=c["n"]))


def correct(folder100, vdir, offset=None, log=print):
    """Measure the piece's 100x pictures and apply the corrections to its files (see the module's text).  Returns a
    summary dict (also the viewer's D.pos100), or None (no 100x index or nothing measured)."""
    import time
    folder100, vdir = Path(folder100), Path(vdir)
    name = piece_name(folder100)
    t0 = time.time()
    recs = measure(folder100, vdir, offset, log)
    if not recs:
        return None
    corr = corrections(recs, vdir, log)
    meas = [c for c in corr.values() if c["how"] == "100x"]
    if not meas:
        log(f"{name}: positions from the 100x pictures: none of {len(recs)} pictures matched its 10x picture; "
            "the positions from the 10x picture are kept")
        return None
    mag = np.array([math.hypot(c["dx"], c["dy"]) for c in meas])
    note = dict(date=time.strftime("%Y-%m-%d %H:%M"), pictures=len(recs), measured=sum(r["ok"] for r in recs),
                flakes_100x=len(meas), flakes_tile=sum(c["how"] == "10x tile" for c in corr.values()),
                median_um=round(float(np.median(mag)), 1), max_um=round(float(mag.max()), 1),
                over_20um=int((mag > 20).sum()))
    files = piece_files(data_dirs(folder100, vdir), name)
    rd = next((d for d in (vdir.parent, vdir / "data") if d.is_dir() and any(d.glob(f"{name}_*_live_flakes.csv"))), vdir.parent)
    write_report(rd / f"{name}_{REPORT}", recs, corr)
    nrow = 0
    for f in files:
        try:
            if f.name.endswith(".csv"):
                nrow += apply_csv(f, corr)
            elif f.name.endswith(".pos"):
                nrow += apply_pos(f, corr)
            else:
                nrow += apply_live_json(f, corr)
        except Exception as e:                       # (one unreadable file does not stop the others; it is reported)
            log(f"  WARNING: {f.name}: positions not updated ({type(e).__name__}: {e})")
    nv = apply_viewer(vdir, corr, note)
    try:
        from .zoomview import upgrade_page                # (the current page: it says where a position comes from)
        upgrade_page(vdir / "index.html", backup=False)
    except Exception:
        pass
    log(f"{name}: positions from the 100x pictures: {note['measured']} of {len(recs)} pictures matched their 10x picture; "
        f"{len(meas)} flakes measured (the 10x picture had them off by a median {note['median_um']} um, at most "
        f"{note['max_um']} um), {note['flakes_tile']} more estimated from their 10x tile; {nv} viewer entries and "
        f"{nrow} table rows updated ({time.time() - t0:.0f} s)")
    if note["over_20um"]:
        log(f"  WARNING: {note['over_20um']} flakes were more than 20 um from where the 10x picture showed them: "
            f"tiles of the 10x scan were not where the stage reported (their flakes now have the 100x positions)")
    return note


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m flakeid.pos100", description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", help="the piece's 100x folder (<piece>_100x with its 100x index)")
    ap.add_argument("--viewer", required=True, help="the piece's viewer folder (index.html, tiles/, 100x/)")
    ap.add_argument("--measure-only", action="store_true", help="only write <piece>_positions100.csv next to the viewer")
    a = ap.parse_args(argv)
    if a.measure_only:
        recs = measure(a.folder, a.viewer)
        corr = corrections(recs, a.viewer)
        out = Path(a.viewer) / f"{piece_name(a.folder)}_{REPORT}"
        write_report(out, recs, corr)
        print(f"{sum(r['ok'] for r in recs)} of {len(recs)} pictures measured -> {out}")
        return 0
    return 0 if correct(a.folder, a.viewer) else 1


if __name__ == "__main__":
    raise SystemExit(main())
