"""The results page of a sorted scan: index.html beside img/ (collect.py).  One card per flake: the 100x picture, the
scan's 10x picture, thickness, size, cleanliness, where it is.  The boxes at the top are the criteria the scan was
sorted with; changing them re-sorts the page on the spot (the same rule as criteria.classify; the files in match/ and
possible/ change only with the window's "Sort again").  "Real" / "Glue" on a card records what you see (kept in the
browser; "Save my marks" downloads them as a labels file for tools/fit_real100.py, which trains the sorter)."""
from __future__ import annotations

import json
from pathlib import Path

from .criteria import BIGGER, DOUBT

KEEP = ("piece", "order", "label", "key", "p_real", "flat_frac", "stage_x_um", "stage_y_um", "z_um", "x10_um", "y10_um", "region",
        "flake_id", "thickness_nm", "thickness_min_nm", "thickness_max_nm", "alt_nm", "alt_ruled_out_nm", "size_um", "area_um2", "quality",
        "contrast", "fit_residual", "notes", "also", "crop", "picture", "aimed", "dist_um", "thickness_10x_nm", "regions100")

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__RUN__ flakes</title>
<style>
:root{--bg:#15161a;--card:#21232a;--line:#33363f;--fg:#e8e8ea;--dim:#a3a6b0;--acc:#7fb2ff;--ok:#6fd08c;--warn:#f0b35a;--bad:#e2777a}
*{box-sizing:border-box}body{background:var(--bg);color:var(--fg);font:14px system-ui,Segoe UI,sans-serif;margin:0}
header{padding:14px 18px 6px}h1{font-size:20px;margin:0 0 4px}p.lead{color:var(--dim);margin:2px 0;max-width:1100px}
#bar{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 18px;display:flex;gap:18px;flex-wrap:wrap;align-items:flex-end}
#bar label{display:flex;flex-direction:column;font-size:12px;color:var(--dim);gap:3px}
#bar input{width:74px;background:#0f1013;color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:5px 6px;font:inherit}
#bar .pair{display:flex;gap:4px;align-items:center}
button{background:#2b2e37;color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:6px 10px;font:inherit;cursor:pointer}
button:hover{border-color:var(--acc)}
#tabs{display:flex;gap:6px;flex-wrap:wrap;padding:10px 18px 0}
#tabs button.on{background:var(--acc);color:#0b0c0f;border-color:var(--acc);font-weight:600}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:14px;padding:14px 18px 40px}
.c{background:var(--card);border-radius:6px;padding:9px;border-left:4px solid var(--line)}
.c.match{border-left-color:var(--ok)}.c.possible{border-left-color:var(--warn)}.c.glue,.c.doubtful{border-left-color:var(--bad)}
.t{font-size:20px;font-weight:600;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}.t small{font-size:13px;font-weight:400;color:var(--dim)}
.p{display:flex;gap:6px;align-items:flex-start;margin:6px 0}.p a:first-child{flex:3}.p a+a{flex:1.6}.p img{width:100%;display:block;border-radius:3px}
.s{color:var(--dim);font-size:12px;line-height:1.5}.s b{color:var(--warn);font-weight:500}
.m{display:flex;gap:6px;margin-top:6px;align-items:center}.m button{padding:3px 9px;font-size:12px}
.m button.on.r{background:var(--ok);color:#0b0c0f}.m button.on.g{background:var(--bad);color:#0b0c0f}
.tag{font-size:11px;border:1px solid var(--line);border-radius:10px;padding:1px 8px;color:var(--dim)}
#empty{padding:30px 18px;color:var(--dim)}
@media(max-width:560px){#grid{grid-template-columns:1fr;padding:10px}.p{flex-direction:column}}
</style></head><body>
<header><h1>__RUN__ &middot; sorted flakes</h1>
<p class="lead">Left: the 100x picture; thickness and size are of what lies <b>under the white cross</b> in its middle. Right: the scan's 10x picture of the target (the box is the 100x field).
The thickness comes from the colour and is <b>not checked against AFM</b>. Stage positions are with the 100x objective in.</p>
<p class="lead">Scan sorted __WHEN__ with: __CRIT__. Change the boxes to re-sort this page; the picture folders (match/, possible/) change with &ldquo;Sort again&rdquo; in the scanner window.</p></header>
<div id="bar">
<label>Thickness, nm<span class="pair"><input id="tmin" type="number" step="any" min="0"> to <input id="tmax" type="number" step="any" min="0"></span></label>
<label>Flat area across, &micro;m (0 = no limit)<span class="pair"><input id="smin" type="number" step="any" min="0"> to <input id="smax" type="number" step="any" min="0"></span></label>
<label>At least this clean, %<input id="clean" type="number" step="5" min="0" max="100"></label>
<label>Real-flake probability at least<input id="cut" type="number" step="0.05" min="0.01" max="0.99"></label>
<label>Order<select id="sort" style="background:#0f1013;color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:5px"><option value="t">thinnest first</option><option value="s">largest first</option><option value="c">cleanest first</option><option value="n">as photographed</option></select></label>
<button id="reset">Back to the scan's values</button><button id="save">Save my marks</button>
</div>
<div id="tabs"></div><div id="grid"></div><div id="empty" hidden>Nothing in this group.</div>
<script>
const DATA = __DATA__, CRIT0 = __CRIT_JSON__, DOUBT = __DOUBT__, BIGGER = __BIGGER__, RUN = "__RUN__";
const NAMES = {match:"Wanted", possible:"Possibly wanted (check by eye)", other:"Other real flakes", doubtful:"Not sure: flake or glue", glue:"Glue"};
const $ = id => document.getElementById(id);
let tab = "match", marks = {};
try { marks = JSON.parse(localStorage.getItem("marks_" + RUN) || "{}"); } catch (e) {}
function crit() { const v = id => parseFloat($(id).value) || 0;
  return {tmin: v("tmin"), tmax: v("tmax") || 400, smin: v("smin"), smax: v("smax"), clean: Math.min(v("clean"), 100) / 100, cut: Math.min(Math.max(v("cut") || 0.5, 0.01), 0.99)}; }
function tri(lo, hi, a, b) { if (a == null) return "maybe"; if (b == null) b = a; return (b >= lo && (!hi || a <= hi)) ? "yes" : "no"; }
function area100(f, c) {             // criteria.py area100(): the largest flat area of the 100x picture inside the criteria
  return (f.regions100 || []).find(r => r.thickness_nm != null
    && tri(c.tmin, c.tmax, r.thickness_min_nm ?? r.thickness_nm, r.thickness_max_nm ?? r.thickness_nm) == "yes"
    && tri(c.smin, c.smax, r.size_um) == "yes") || null;
}
function classify(f, c) {            // the rule of flakeid/gscan/criteria.py classify(); a mark by eye overrides the sorter
  const mk = marks[f.key];
  if (mk === "glue") return ["glue", "marked glue by eye"];
  if (mk !== "real") { if (f.p_real < Math.min(DOUBT, c.cut)) return ["glue", "sorter: real-flake probability " + f.p_real.toFixed(2)];
                       if (f.p_real < c.cut) return ["doubtful", "sorter not sure: real-flake probability " + f.p_real.toFixed(2)]; }
  const why = []; let th = tri(c.tmin, c.tmax, f.thickness_nm == null ? null : f.thickness_min_nm, f.thickness_max_nm);
  if (th == "no" && (f.alt_nm || []).some(a => a >= c.tmin && a <= c.tmax)) { th = "maybe"; why.push("the same colour is also a thickness inside the window"); }
  else if (th == "maybe") why.push("thickness not known"); else if (th == "no") why.push("thickness " + f.thickness_nm.toFixed(0) + " nm");
  const sz = tri(c.smin, c.smax, f.size_um);
  if (sz == "maybe") why.push("size not known"); else if (sz == "no") why.push("flat area " + f.size_um.toFixed(0) + " µm across");
  const cl = (!c.clean || f.flat_frac >= c.clean) ? "yes" : "no"; if (cl == "no") why.push((f.flat_frac * 100).toFixed(0) + " % clean");
  if ((th != "yes" || sz != "yes") && cl == "yes") { const r = area100(f, c);
    if (r) return ["match", `wanted for an area seen at 100x: ${r.thickness_nm.toFixed(1)} nm, ${r.size_um.toFixed(0)} µm across, `
                   + (r.at_cross ? "at the cross" : Math.hypot(r.dx_um, r.dy_um).toFixed(0) + " µm from the cross"), r]; }
  if ([th, sz, cl].includes("no")) return ["other", why.filter(w => !w.includes("not known") && !w.includes("same colour")).join("; ")];
  if (th == "maybe" || sz == "maybe") return ["possible", why.join("; ")];
  const r = area100(f, c);               // criteria.py headline(): a wanted 100x area much larger than the 10x region heads the card
  return ["match", "", r && r.size_um > BIGGER * (f.size_um || 0) ? r : null];
}
const esc = s => String(s).replace(/[&<>"]/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[ch]));
function card(f, cls, why, r) {      // r: the 100x area the flake is wanted for (its numbers head the card)
  let t, rng = "", size;
  if (r) { t = r.thickness_nm.toFixed(1) + ' nm'; rng = "(100x colour)";
    size = `<small>flat area ${r.size_um.toFixed(0)} µm across${r.cut ? " or more" : ""}, ${Math.round(r.area_um2)} µm² (100x)</small>`; }
  else { t = f.thickness_nm == null ? 'thickness not known' : f.thickness_nm.toFixed(1) + ' nm';
    if (f.thickness_nm != null) { rng = "(" + f.thickness_min_nm.toFixed(0) + "–" + f.thickness_max_nm.toFixed(0);
      if ((f.alt_nm || []).length) rng += ", or " + f.alt_nm.map(a => a.toFixed(0)).join(", "); rng += ")"; }
    size = f.size_um == null ? "" : `<small>flat area ${f.size_um.toFixed(0)} µm across, ${Math.round(f.area_um2)} µm²</small>`; }
  const pics = `<a href="${f.picture || 'img/' + f.key + '_100x.jpg'}" target="_blank"><img loading="lazy" src="img/${f.key}_100x.jpg" alt="100x picture"></a>`
             + (f.crop ? `<a href="img/${f.key}_10x.jpg" target="_blank"><img loading="lazy" src="img/${f.key}_10x.jpg" alt="10x picture"></a>` : "");
  const lines = [`${esc(f.piece)} picture #${String(f.order).padStart(3, "0")} · stage x ${f.stage_x_um.toFixed(1)}, y ${f.stage_y_um.toFixed(1)}` + (f.z_um == null ? "" : `, focus ${f.z_um.toFixed(1)}`) + " µm"];
  lines.push(`${(f.flat_frac * 100).toFixed(0)} % clean · real-flake probability ${f.p_real.toFixed(2)}` + (f.region ? ` · 10x region ${esc(f.region)} of flake ${esc(f.flake_id)}` : " · no flat 10x region under the cross")
             + (f.quality && f.quality != "good" ? ` · <b>${esc(f.quality)}</b>` : ""));
  if (r && f.thickness_nm != null) lines.push(`the 10x region under the cross read ${f.thickness_nm.toFixed(1)} nm, ${(f.size_um || 0).toFixed(0)} µm across`);
  if ((f.regions100 || []).length) lines.push("flat areas in the 100x picture (100x colour): " + f.regions100.map(a =>
    (a.thickness_nm == null ? "? nm" : a.thickness_nm.toFixed(1) + " nm") + ", " + a.size_um.toFixed(0) + (a.cut ? "+" : "") + " µm"
    + (a.at_cross ? " (at the cross)" : ` (${Math.hypot(a.dx_um, a.dy_um).toFixed(0)} µm off)`)).join(" · "));
  if (!f.region && f.aimed && f.aimed.thickness_nm != null) lines.push(`the scan aimed at a flake read as ${f.aimed.thickness_nm.toFixed(0)} nm, ${(f.aimed.size_um || 0).toFixed(0)} µm (not necessarily what is under the cross)`);
  (f.notes || []).forEach(n => lines.push("<b>" + esc(n) + "</b>"));
  if ((f.also || []).length) lines.push("also photographed as " + f.also.map(esc).join(", "));
  if (why) lines.push("<b>" + esc(why) + "</b>");
  const mk = marks[f.key];
  return `<div class="c ${cls}"><div class="t">${t} <small>${rng}</small>${size}</div><div class="p">${pics}</div><div class="s">${lines.join("<br>")}</div>
<div class="m"><span class="s">By eye:</span><button class="r ${mk == 'real' ? 'on' : ''}" data-k="${f.key}" data-v="real">Real flake</button><button class="g ${mk == 'glue' ? 'on' : ''}" data-k="${f.key}" data-v="glue">Glue</button></div></div>`;
}
function render() {
  const c = crit(), groups = {match: [], possible: [], other: [], doubtful: [], glue: []};
  DATA.forEach(f => { const [k, why, r] = classify(f, c); groups[k].push([f, why, r]); });
  const key = {t: e => (e[2] || e[0]).thickness_nm ?? 1e9, s: e => -((e[2] || e[0]).size_um || 0), c: e => -e[0].flat_frac, n: e => 0}[$("sort").value];
  Object.values(groups).forEach(g => g.sort((a, b) => key(a) - key(b)));
  $("tabs").innerHTML = Object.keys(groups).map(k => `<button data-t="${k}" class="${k == tab ? 'on' : ''}">${NAMES[k]} (${groups[k].length})</button>`).join("");
  $("grid").innerHTML = groups[tab].map(([f, why, r]) => card(f, tab, why, r)).join("");
  $("empty").hidden = groups[tab].length > 0;
}
function reset() { $("tmin").value = CRIT0.thickness_min_nm; $("tmax").value = CRIT0.thickness_max_nm; $("smin").value = CRIT0.size_min_um; $("smax").value = CRIT0.size_max_um;
  $("clean").value = Math.round(CRIT0.clean_min * 100); $("cut").value = CRIT0.real_cut; render(); }
document.addEventListener("input", e => { if (e.target.closest("#bar")) render(); });
$("tabs").addEventListener("click", e => { const b = e.target.closest("button"); if (b) { tab = b.dataset.t; render(); window.scrollTo(0, 0); } });
$("grid").addEventListener("click", e => { const b = e.target.closest(".m button"); if (!b) return;
  if (marks[b.dataset.k] === b.dataset.v) delete marks[b.dataset.k]; else marks[b.dataset.k] = b.dataset.v;
  try { localStorage.setItem("marks_" + RUN, JSON.stringify(marks)); } catch (err) {}
  render(); });
$("reset").addEventListener("click", reset);
$("save").addEventListener("click", () => {      // the labels format of tools/fit_real100.py: one entry per 100x folder
  const by = {}; DATA.forEach(f => { const m = marks[f.key]; if (!m) return; (by[f.piece] = by[f.piece] || {folder: f.piece + "_100x", real: [], glue: []})[m].push(f.order); });
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(Object.values(by), null, 1)], {type: "application/json"}));
  a.download = "marks_" + RUN + ".json"; a.click(); });
reset();
</script></body></html>
"""


def write_page(out, info, flakes):
    out = Path(out)
    data = [{k: f.get(k) for k in KEEP if f.get(k) not in (None, "", [])} for f in flakes]
    from .criteria import Criteria
    crit = Criteria(**info["criteria"])
    html = (HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
            .replace("__CRIT_JSON__", json.dumps(info["criteria"])).replace("__DOUBT__", repr(DOUBT)).replace("__BIGGER__", repr(BIGGER))
            .replace("__RUN__", str(info["run"]).replace('"', "")).replace("__WHEN__", info["sorted"]).replace("__CRIT__", crit.describe()))
    (out / "index.html").write_text(html, encoding="utf-8")
    return out / "index.html"
