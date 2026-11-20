"""A page to check the thicknesses by eye: every 100x picture of scanned pieces with the thickness of the flat region
in its middle, real flakes first (thinnest first), then pictures without a region, then glue.

    py -3.9 tools/bx53m/thickness_gallery.py TERRACES.csv PIECE [PIECE ...] -o C:/FlakeScans/thickness_check
            [--root C:/FlakeScans] [--labels LABELS.json] [--offset 21.1 -38.4]

TERRACES.csv  <piece>_tiles_live_terraces.csv of the scan's analysis (flakeid.live) with the profile to check: the
              thicknesses, and each region's stage position
PIECE         pieces of the same chip, not moved since that scan, with 100x pictures (<root>/<piece>_100x)

The picture of a region: the 100x picture whose middle is inside the region's largest inscribed circle.  The middle of
a 100x picture is its stage position less the 10x -> 100x offset (the 100x list holds positions with the offset; the
regions without).  A picture can therefore only get the thickness of what is in its middle; a picture with no region
there is listed without a thickness.  The 10x picture beside it is the scan's own target crop, by the target's label.
Real or glue: LABELS.json (tools/fit_real100.py's format: by eye) where the picture is listed, else the 100x sorter.
"""
import argparse
import csv
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np

BINS = ((0, 5), (5, 20), (20, 50), (50, 100), (100, 200), (200, 400.1))


def rows(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("terraces")
    ap.add_argument("pieces", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--root", default="C:/FlakeScans")
    ap.add_argument("--labels")
    ap.add_argument("--offset", nargs=2, type=float, default=(21.1, -38.4), help="10x -> 100x offset, um (setting offset_100x_um)")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.output)
    if (out / "img").is_dir():
        for f in (out / "img").glob("*.jpg"):
            f.unlink()
    (out / "img").mkdir(parents=True, exist_ok=True)
    T = rows(a.terraces)
    P = np.array([[float(t["stage_x_um"]), float(t["stage_y_um"])] for t in T])
    D = np.array([float(t["inscribed_diameter_um"]) for t in T])
    eye = {Path(s["folder"]).name: s for s in json.loads(Path(a.labels).read_text())} if a.labels else {}
    real, glue, none = {b: [] for b in BINS}, [], []
    E = []                      # every picture: (region index or None, distance, real?, name, pictures, foot, position)
    for name in a.pieces:
        h = root / f"{name}_100x"
        verdict = {r["file"]: r for r in rows(h / "real100.csv")} if (h / "real100.csv").is_file() else {}
        lab = eye.get(h.name, {})
        for r in rows(next(h.glob("*_100x_index.csv"))):
            order = int(r["order"])
            f = next((d / (r["file"][:-4] + s) for d in (h, h / "rejected") for s in ("_colour.png", "_colour.tif")
                      if (d / (r["file"][:-4] + s)).is_file()), None)
            if f is None:
                continue
            q = np.array([float(r["stage_x_um"]) - a.offset[0], float(r["stage_y_um"]) - a.offset[1]])
            dist = np.hypot(*(P - q).T)
            inside = np.flatnonzero(dist < D / 2)
            t = T[int(inside[np.argmin(dist[inside])])] if len(inside) else None
            key = f"{name}_{order:03d}"
            im = cv2.resize(cv2.imread(str(f), cv2.IMREAD_COLOR), (800, 600), interpolation=cv2.INTER_AREA)
            cv2.drawMarker(im, (400, 300), (255, 255, 255), cv2.MARKER_CROSS, 26, 1)
            cv2.imwrite(str(out / "img" / f"{key}_100x.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            crop, c10 = root / f"{name}_targets_crops" / f"{r['label']}.jpg", ""
            if crop.is_file():
                cv2.imwrite(str(out / "img" / f"{key}_10x.jpg"), cv2.imread(str(crop)), [cv2.IMWRITE_JPEG_QUALITY, 92])
                c10 = f'<a href="img/{key}_10x.jpg"><img src="img/{key}_10x.jpg" loading="lazy"></a>'
            v = verdict.get(r["file"], {})
            if order in lab.get("real", []):
                is_real, src = True, "real (by eye)"
            elif order in lab.get("glue", []):
                is_real, src = False, "glue (by eye)"
            elif lab.get("default") == "glue":
                is_real, src = False, "glue (by eye)"
            elif v:
                is_real, src = v.get("real") == "1", f"{'real' if v.get('real') == '1' else 'glue'} (sorter, p {float(v['p_real']):.2f}; not checked by eye)"
            else:
                is_real, src = True, "not judged"
            pics = f'<div class="p"><a href="img/{key}_100x.jpg"><img src="img/{key}_100x.jpg" loading="lazy"></a>{c10}</div>'
            foot = f'{html.escape(name)} 100x picture #{order:03d} &middot; {html.escape(src)} &middot; old label {html.escape(r["label"])}'
            j = int(inside[np.argmin(dist[inside])]) if len(inside) else None
            E.append(dict(j=j, d=float(dist[j]) if j is not None else 0.0, real=is_real, pic=f"{name} #{order:03d}", pics=pics, foot=foot, q=q))
    # one card per region (the same flake photographed in two runs, or twice in one: the picture nearest its middle, a
    # real one before a glue one); pictures without a region: one per spot (20 um)
    groups = {}
    for e in E:
        if e["j"] is not None:
            groups.setdefault(e["j"], []).append(e)
        else:
            k = next((k for k, g in groups.items() if isinstance(k, tuple) and np.hypot(*(g[0]["q"] - e["q"])) < 20.0), None)
            groups.setdefault(k if k is not None else ("spot", len(groups)), []).append(e)
    for k, g in groups.items():
        g.sort(key=lambda e: (not e["real"], e["d"]))
        e = g[0]
        is_real, pics = e["real"], e["pics"]
        foot = e["foot"] + (f'<br>also photographed as {", ".join(html.escape(x["pic"]) for x in g[1:])}' if len(g) > 1 else "")
        if True:
            t = T[k] if not isinstance(k, tuple) else None
            if t is None:
                card = (f'<div class="c"><div class="t">no thickness <span>(no flat region found in the middle of this picture)</span></div>{pics}'
                        f'<div class="s">{foot}</div></div>')
                (none if is_real else glue).append((0.0, card))
                continue
            th = float(t["thickness_nm"])
            alt = f", or {html.escape(t['alt_thickness_nm'])}" if t.get("alt_thickness_nm") else ""
            flags = "" if t["quality"] in ("good", "") else f' &middot; <b>{html.escape(t["quality"])}</b>'
            card = (f'<div class="c"><div class="t">{th:.1f} nm <span>({float(t["thickness_min_nm"]):.0f}&ndash;{float(t["thickness_max_nm"]):.0f}{alt}){flags}</span></div>{pics}'
                    f'<div class="s">{foot}<br>region {t["terrace_id"]} of flake {t["flake_id"]}: {float(t["area_um2"]):.0f} um2, flat disc {float(t["inscribed_diameter_um"]):.0f} um, '
                    f'{e["d"]:.0f} um from the picture&#39;s middle &middot; contrast R {float(t["contrast_r"]):+.2f} G {float(t["contrast_g"]):+.2f} '
                    f'B {float(t["contrast_b"]):+.2f} &middot; fit {float(t["fit_residual"]):.2f}</div></div>')
            if is_real:
                real[next((b for b in BINS if b[0] <= th < b[1]), BINS[-1])].append((th, card))
            else:
                glue.append((th, card))
    n_real = sum(len(c) for c in real.values())
    join = lambda c: "".join(x for _, x in sorted(c, key=lambda q: q[0]))
    nav = (" ".join(f'<a href="#b{int(b[0])}">{b[0]:g}&ndash;{int(b[1])} nm ({len(c)})</a>' for b, c in real.items())
           + f' <a href="#none">no region ({len(none)})</a> <a href="#glue">glue ({len(glue)})</a>')
    body = "".join(f'<h2 id="b{int(b[0])}">{b[0]:g}&ndash;{int(b[1])} nm &middot; {len(c)} pictures</h2><div class="g">{join(c)}</div>' for b, c in real.items())
    body += f'<h2 id="none">Real, but no flat region in the middle &middot; {len(none)} pictures</h2><div class="g">{join(none)}</div>'
    body += f'<h2 id="glue">Glue &middot; {len(glue)} pictures (a thickness here means nothing)</h2><div class="g">{join(glue)}</div>'
    (out / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Thickness check</title>
<style>body{{background:#16171a;color:#e6e6e6;font:14px system-ui,sans-serif;margin:16px}}a{{color:#8ab4ff}}
nav{{position:sticky;top:0;background:#16171a;padding:8px 0;display:flex;gap:16px;flex-wrap:wrap}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:14px}}.c{{background:#22242a;padding:8px;border-radius:6px}}
.p{{display:flex;gap:6px;align-items:flex-start;margin:4px 0}}.p a:first-child{{flex:3}}.p a+a{{flex:1.6}}.p img{{width:100%;display:block}}
.t{{font-size:20px;font-weight:600}}.t span{{font-size:13px;font-weight:400;color:#aaa}}.t b{{color:#ffb86b}}.s{{color:#aaa;font-size:12px}}</style>
<h1>Thickness check</h1>
<p>{n_real} pictures of real flakes with a thickness, thinnest first; then {len(none)} without one and {len(glue)} glue.
Left: the 100x picture. <b>The thickness is of what lies under the white cross in its middle</b>, not of other flakes in view.
Right: the scan's 10x target (raw data in false colour; the box is the 100x field). In brackets: the range, and other
thicknesses the same colour could be. "Old label" is the name the scan gave the picture, with the old, wrong thickness.
Not checked against AFM.</p><nav>{nav}</nav>{body}""", encoding="utf-8")
    print(f"{n_real} real with a thickness, {len(none)} real without, {len(glue)} glue:", out / "index.html")


if __name__ == "__main__":
    main()
