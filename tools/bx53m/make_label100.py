"""A page to label a run's 100x pictures by eye (good target / too thick / glue ...), for tuning the automatic 100x verdict.

    py -3.9 tools/bx53m/make_label100.py RESULTS_PIECE_FOLDER [...] -o label_100x.html

RESULTS_PIECE_FOLDER: <results>/<run>/<piece> (the folder with <piece>_tiles_viewer).  Open the page in a browser: one
picture at a time, a key or a click per label; the labels are kept in the browser and "Download labels" saves them as
labels_100x.json (picture, label), which is what to send back.
"""
import argparse, glob, json, os, sys

LABELS = [("1", "good", "Good target"), ("2", "thick", "Too thick"), ("3", "thin", "Too thin / too faint"), ("4", "glue", "Glue / dirty"),
          ("5", "small", "Too small / broken"), ("6", "unsure", "Unsure")]
PAGE = """<!doctype html><meta charset="utf-8"><title>Label 100x pictures</title>
<style>body{background:#1b1b1f;color:#eee;font:15px Segoe UI,sans-serif;margin:12px}img{max-width:100%;max-height:78vh;display:block;margin:8px 0}
button{font:15px Segoe UI;padding:8px 14px;margin:3px;border:1px solid #666;background:#2c2c33;color:#eee;border-radius:6px;cursor:pointer}
button.on{background:#3d7a4a;border-color:#7c7}#bar{color:#aaa}</style>
<div id="bar"></div><img id="im"><div id="btn"></div>
<p><button onclick="go(-1)">&larr; back</button><button onclick="go(1)">next &rarr;</button><button onclick="nextUn()">next unlabelled</button>
<button onclick="dl()">Download labels</button> &nbsp; keys: 1-6 label, arrows move</p>
<script>
const ITEMS=__ITEMS__, LABELS=__LABELS__, KEY="label100x:"+ITEMS.length+":"+ITEMS[0].id;
let L=JSON.parse(localStorage.getItem(KEY)||"{}"), i=0;
function show(){const it=ITEMS[i];document.getElementById("im").src=it.src;
 document.getElementById("bar").textContent=(i+1)+" / "+ITEMS.length+"   "+it.id+"   (10x reading: "+it.t10+")   labelled: "+Object.keys(L).length;
 document.getElementById("btn").innerHTML=LABELS.map(l=>'<button class="'+(L[it.id]==l[1]?"on":"")+'" onclick="setL(\''+l[1]+'\')">'+l[0]+"  "+l[2]+"</button>").join("");}
function setL(v){L[ITEMS[i].id]=v;localStorage.setItem(KEY,JSON.stringify(L));if(i<ITEMS.length-1)i++;show();}
function go(d){i=Math.max(0,Math.min(ITEMS.length-1,i+d));show();}
function nextUn(){for(let k=0;k<ITEMS.length;k++){const j=(i+1+k)%ITEMS.length;if(!L[ITEMS[j].id]){i=j;break;}}show();}
function dl(){const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(L,null,1)],{type:"application/json"}));a.download="labels_100x.json";a.click();}
document.addEventListener("keydown",e=>{const l=LABELS.find(l=>l[0]==e.key);if(l)setL(l[1]);else if(e.key=="ArrowRight")go(1);else if(e.key=="ArrowLeft")go(-1);});
show();
</script>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pieces", nargs="+")
    ap.add_argument("-o", "--out", default="label_100x.html")
    a = ap.parse_args()
    items = []
    for p in a.pieces:
        for v in glob.glob(os.path.join(p, "*_tiles_viewer", "100x")):
            for f in sorted(glob.glob(os.path.join(v, "*_regions.json"))):
                stem = os.path.basename(f)[:-len("_regions.json")]
                jpg = os.path.join(v, stem + ".jpg")
                if os.path.isfile(jpg):
                    parts = stem.split("_")
                    items.append(dict(id=stem, src="file:///" + os.path.abspath(jpg).replace("\\", "/"), t10=next((q for q in parts if q.endswith("nm")), "?")))
    if not items:
        sys.exit("no 100x pictures found")
    open(a.out, "w", encoding="utf-8").write(PAGE.replace("__ITEMS__", json.dumps(items)).replace("__LABELS__", json.dumps(LABELS)))
    print(len(items), "pictures ->", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
