"""The glovebox flake scanner's window: which flakes are wanted, Start scan, and the sorted results.

    python -m flakeid.gscan [--settings scan_settings.json]

The window itself never touches the microscope.  "Start scan" starts the scan job as its own process (engine.py: the
original program's scan, its live view and log window); when that ends the 100x pictures are sorted (collect.py) and
the results page opens.  "Sort again" does only the sorting, on any finished scan, with the values in the window now.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import criteria as C

ROOT = Path(__file__).resolve().parents[2]
GB_PER_SCAN = 4.5           # disk a scan needs: 99 tiles 0.6 GB, each 100x picture 14 MB (raw frame + colour copy)
S_PER_PICTURE = 7.0         # one 100x picture: move, autofocus, 4 frames (raw_3: 30 pictures in 3.4 min)


class App:
    def __init__(self, settings_file):
        from ..scope.settings import load
        from ..scope.ui import apply_theme, dpi_aware, set_icon
        self.settings_file = str(settings_file)
        self.s = load(self.settings_file if Path(self.settings_file).is_file() else None,
                      **({} if Path(self.settings_file).is_file() else {"hardware": "bx53m"}))
        self.local = Path(self.s.local_root or "C:/FlakeScans")
        self.crit_file = self.local / "scanner_criteria.json"
        self.q = queue.Queue()
        self.proc = None
        self.busy = False
        dpi_aware()
        self.root = tk.Tk()
        self._icons = set_icon(self.root)
        apply_theme(self.root, self.s.ui_theme)
        self.root.title("Glovebox flake scanner")
        self.build(C.load(self.crit_file))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.pump)

    # ---- the window
    def build(self, c):
        f = ttk.Frame(self.root, padding=14)
        f.pack(fill="both", expand=True)
        self.v = {}

        def entry(parent, key, value, width=8):
            self.v[key] = tk.StringVar(value=f"{value:g}" if isinstance(value, float) else str(value))
            e = ttk.Entry(parent, textvariable=self.v[key], width=width)
            return e

        def row(parent, r, label, *widgets, hint=""):
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", pady=4, padx=(0, 12))
            box = ttk.Frame(parent)
            box.grid(row=r, column=1, sticky="w")
            for w in widgets:
                w = w(box) if callable(w) else w
                w.pack(side="left", padx=(0, 6))
            if hint:
                ttk.Label(parent, text=hint, foreground="#9a96b8", wraplength=430, justify="left").grid(row=r, column=2, sticky="w", padx=(12, 0))

        ttk.Label(f, text="Flakes wanted", font=("Segoe UI", 13, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
        w = ttk.Frame(f)
        w.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 10))
        self.v["material"] = tk.StringVar(value=c.material)
        row(w, 0, "Material, oxide",
            lambda b: ttk.Combobox(b, textvariable=self.v["material"], values=("hbn", "graphite"), width=9, state="readonly"),
            lambda b: ttk.Label(b, text="on"), lambda b: entry(b, "oxide_nm", float(c.oxide_nm), 6), lambda b: ttk.Label(b, text="nm SiO2 (nominal)"),
            hint="The colour model here is fitted for hBN on ~285 nm oxide; the real oxide is read from the substrate colour.")
        row(w, 1, "Thickness", lambda b: entry(b, "thickness_min_nm", float(c.thickness_min_nm)), lambda b: ttk.Label(b, text="to"),
            lambda b: entry(b, "thickness_max_nm", float(c.thickness_max_nm)), lambda b: ttk.Label(b, text="nm"),
            hint="From the colour; not checked against AFM yet. Flakes whose thickness is unknown or ambiguous are kept as \"possible\", never dropped.")
        row(w, 2, "Size of the flat area", lambda b: entry(b, "size_min_um", float(c.size_min_um)), lambda b: ttk.Label(b, text="to"),
            lambda b: entry(b, "size_max_um", float(c.size_max_um)), lambda b: ttk.Label(b, text="\u00b5m across (0 = no upper limit)"),
            hint="The largest circle that fits in the flake's flat, one-colour area.")
        row(w, 3, "Cleanliness: at least", lambda b: entry(b, "clean_pct", float(round(c.clean_min * 100))), lambda b: ttk.Label(b, text="% clean"),
            hint="Share of the flake that is flat and clean in the 100x picture. 0 = any; speckled or stepped flakes are 20\u201340 %, clean ones 75 % and up.")
        row(w, 4, "Real flake, not glue: at least", lambda b: entry(b, "real_cut", float(c.real_cut)), lambda b: ttk.Label(b, text="probability (0.5 usual)"),
            hint="The 100x sorter. Lower keeps more doubtful pictures; pictures between 0.15 and this are listed as \"not sure\", below as glue.")

        ttk.Separator(f).grid(row=2, column=0, columnspan=3, sticky="ew", pady=6)
        ttk.Label(f, text="Scan", font=("Segoe UI", 13, "bold")).grid(row=3, column=0, columnspan=3, sticky="w")
        g = ttk.Frame(f)
        g.grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 10))
        self.v["name"] = tk.StringVar(value="")
        row(g, 0, "Name", lambda b: ttk.Entry(b, textvariable=self.v["name"], width=24),
            hint="Letters, digits, - and _. Empty = today's date. The chip becomes <name>_1.")
        self.v["thin_search"] = tk.BooleanVar(value=bool(c.thin_search))
        row(g, 1, "Thin-flake search", lambda b: ttk.Checkbutton(b, variable=self.v["thin_search"], text="on, up to"),
            lambda b: entry(b, "thin_max_targets", int(c.thin_max_targets), 5), lambda b: ttk.Label(b, text="extra 100x pictures per chip"),
            hint="Also looks for ~1\u20135 nm hBN, too faint for the normal search. Not yet proven on this camera: many of its finds may be "
                 f"glue, which the 100x sorter removes. About {S_PER_PICTURE:.0f} s per picture.")
        row(g, 2, "At most", lambda b: entry(b, "pictures_per_cm2", float(c.pictures_per_cm2)), lambda b: ttk.Label(b, text="100x pictures per cm\u00b2 of chip"),
            hint="Every flake found at 10x is photographed up to this number (best-scoring first). The test chip had ~420 per cm\u00b2.")
        self.v["skip_thick_at_10x"] = tk.BooleanVar(value=bool(c.skip_thick_at_10x))
        row(g, 3, "Faster", lambda b: ttk.Checkbutton(b, variable=self.v["skip_thick_at_10x"],
                                                       text="do not photograph flakes clearly thicker than the window"),
            hint="Off: everything is photographed and sorted afterwards (safest). The thin end is never cut at 10x.")
        self.v["copy_to"] = tk.StringVar(value=c.copy_to)
        row(g, 4, "Copy the results to", lambda b: ttk.Entry(b, textvariable=self.v["copy_to"], width=44),
            lambda b: ttk.Button(b, text="Browse...", command=self.browse), hint="Optional. They always stay in " + str(self.local / "results") + ".")

        bar = ttk.Frame(f)
        bar.grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 4))
        self.start_b = ttk.Button(bar, text="Start scan", style="Accent.TButton", command=self.start, width=18)
        self.start_b.pack(side="left", padx=(0, 10))
        self.check_b = ttk.Button(bar, text="Check the hardware (nothing moves)", command=self.check)
        self.check_b.pack(side="left", padx=(0, 10))
        ttk.Separator(f).grid(row=6, column=0, columnspan=3, sticky="ew", pady=6)
        old = ttk.Frame(f)
        old.grid(row=7, column=0, columnspan=3, sticky="w")
        ttk.Label(old, text="Earlier scan:").pack(side="left", padx=(0, 6))
        self.v["run"] = tk.StringVar()
        self.run_box = ttk.Combobox(old, textvariable=self.v["run"], width=28, state="readonly")
        self.run_box.pack(side="left", padx=(0, 6))
        self.sort_b = ttk.Button(old, text="Sort again with the values above", command=self.sort_again)
        self.sort_b.pack(side="left", padx=(0, 6))
        self.open_b = ttk.Button(old, text="Open its results", command=self.open_results)
        self.open_b.pack(side="left")
        free = self.free_gb()
        self.status = tk.StringVar(value="Ready.  Close Olympus Stream before a scan."
                                   + (f"  WARNING: only {free:.1f} GB free on the disk; a scan needs about {GB_PER_SCAN:g} GB."
                                      if free is not None and free < 2 * GB_PER_SCAN else ""))
        ttk.Label(f, textvariable=self.status, font=("Segoe UI", 10, "bold"), wraplength=980, justify="left").grid(row=8, column=0, columnspan=3, sticky="w", pady=(10, 4))
        self.log_w = tk.Text(f, height=12, width=120, wrap="word", state="disabled", font=("Consolas", 9), bg="#101014", fg="#d8d8e0")
        self.log_w.grid(row=9, column=0, columnspan=3, sticky="nsew")
        f.rowconfigure(9, weight=1)
        f.columnconfigure(2, weight=1)
        self.refresh_runs()

    def browse(self):
        d = filedialog.askdirectory(title="Copy the results to", initialdir=self.v["copy_to"].get() or str(Path.home()))
        if d:
            self.v["copy_to"].set(d)

    def refresh_runs(self, select=None):
        from .collect import runs
        try:
            self.runs = {name: p for name, p, _ in runs(self.local)}
        except OSError:
            self.runs = {}
        self.run_box["values"] = list(self.runs)
        if select in self.runs:
            self.v["run"].set(select)
        elif self.runs and self.v["run"].get() not in self.runs:
            self.v["run"].set(next(iter(self.runs)))

    # ---- values
    def criteria(self):
        """The window's values as Criteria (saved for the next start), or None with a message when one is not a number."""
        def num(key, what):
            try:
                return float(self.v[key].get().strip().replace(",", ".") or 0)
            except ValueError:
                raise ValueError(f"{what}: \"{self.v[key].get()}\" is not a number")
        try:
            c = C.Criteria(material=self.v["material"].get(), oxide_nm=num("oxide_nm", "Oxide") or 285.0,
                           thickness_min_nm=num("thickness_min_nm", "Thickness"), thickness_max_nm=num("thickness_max_nm", "Thickness"),
                           size_min_um=num("size_min_um", "Size"), size_max_um=num("size_max_um", "Size"),
                           clean_min=num("clean_pct", "Cleanliness") / 100.0, real_cut=num("real_cut", "Real-flake probability") or 0.5,
                           thin_search=bool(self.v["thin_search"].get()), thin_max_targets=int(num("thin_max_targets", "Thin-flake pictures")),
                           pictures_per_cm2=num("pictures_per_cm2", "Pictures per cm2") or 600.0,
                           skip_thick_at_10x=bool(self.v["skip_thick_at_10x"].get()), copy_to=self.v["copy_to"].get().strip()).clean()
        except ValueError as e:
            messagebox.showerror("Glovebox flake scanner", str(e), parent=self.root)
            return None
        try:
            C.save(self.crit_file, c)
        except OSError:
            pass
        return c

    # ---- jobs (each its own process; its output into the log)
    def log(self, text):
        self.q.put(("log", text))

    def pump(self):
        try:
            while True:
                kind, x = self.q.get_nowait()
                if kind == "log":
                    self.log_w.configure(state="normal")
                    self.log_w.insert("end", x.rstrip("\n") + "\n")
                    self.log_w.see("end")
                    self.log_w.configure(state="disabled")
                elif kind == "status":
                    self.status.set(x)
                elif kind == "call":
                    x()
        except queue.Empty:
            pass
        self.root.after(100, self.pump)

    def set_busy(self, on, status=None):
        self.busy = on
        for b in (self.start_b, self.check_b, self.sort_b):
            b.configure(state="disabled" if on else "normal")
        if status:
            self.status.set(status)

    def spawn(self, args, then, quiet=()):
        """python -m <args> in the program folder; its output into the log; then(exit code) in the window's thread."""
        env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(ROOT), OPENCV_LOG_LEVEL="ERROR")
        env.setdefault("FLAKEID_ORIG_PATH", os.environ.get("PATH", ""))
        exe = sys.executable
        if exe.lower().endswith("pythonw.exe"):                 # (started without a console: the jobs too)
            exe = exe
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen([exe, "-u", "-m"] + list(args), cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", creationflags=flags)

        def follow(p=self.proc):
            for line in p.stdout:
                if not any(k in line for k in quiet):
                    self.log(line)
            code = p.wait()
            self.q.put(("call", lambda: then(code)))
        threading.Thread(target=follow, daemon=True).start()

    def start(self):
        c = self.criteria()
        if c is None or self.busy:
            return
        free = self.free_gb()
        if free is not None and free < GB_PER_SCAN:
            if not messagebox.askokcancel(
                    "Disk nearly full",
                    f"Only {free:.1f} GB are free on {self.local.anchor or self.local}.  A scan keeps its raw pictures (about "
                    f"{GB_PER_SCAN:g} GB for a 0.5 cm² chip with 250 100x pictures), and a full disk stops it part-way.\n\n"
                    f"Free some space first (old <chip>_tiles and <chip>_100x folders in {self.local} of scans that will not be "
                    "sorted again), or continue anyway.", icon="warning", parent=self.root):
                return
        if not messagebox.askokcancel(
                "Start scan",
                "Before the scan starts:\n\n"
                "  \u2022  Olympus Stream is closed (it holds the stage and the camera).\n"
                "  \u2022  The chip is on the stage and nothing can be hit: the stage and the focus move by themselves, and "
                "the nosepiece turns by itself (10x for the scan, 100x for the pictures).\n"
                "  \u2022  The manual focus wheel is where it was when the focus limit was calibrated (at its stop).\n"
                "  \u2022  The light path is as it was for the flat fields (same aperture and filters).\n\n"
                f"Wanted: {c.describe()}.\n\n"
                "Next a live view opens: put the cross on each corner of the area to scan, \"Add point\", then \"Finish polygon\" and \"Done\".",
                parent=self.root):
            return
        name = "".join(ch for ch in self.v["name"].get().strip() if ch.isalnum() or ch in "-_")
        jobs = self.local / "scanner_jobs"
        jobs.mkdir(parents=True, exist_ok=True)
        self.job = jobs / (time.strftime("%Y%m%d_%H%M%S") + ".json")
        self.job.write_text(json.dumps(dict(criteria=C.asdict(c), settings=C.scan_settings(c, name or time.strftime("%Y%m%d"))), indent=1),
                            encoding="utf-8")
        self.job_crit = c
        self.log(f"==== scan: {c.describe()}")
        self.set_busy(True, "Scan running: its own windows are open (live view, then the scan log with Stop and Pause).")
        self.spawn(["flakeid.gscan.engine", "--settings", self.settings_file, "--job", str(self.job)], self.scan_ended)

    def free_gb(self):
        import shutil
        try:
            return shutil.disk_usage(str(self.local if self.local.is_dir() else self.local.anchor)).free / 1e9
        except OSError:
            return None

    def scan_ended(self, code):
        res = {}
        try:
            res = json.loads(Path(str(self.job)[:-5] + ".result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        run_file = res.get("run_file")
        if not run_file or not Path(run_file).is_file():
            self.set_busy(False, "The scan did not start (cancelled, or the microscope could not be opened): see the log below.")
            return
        run = Path(run_file).stem
        self.refresh_runs(select=run)
        if not any((Path(run_file).parent / f"{p['name']}_100x").is_dir() for p in json.loads(Path(run_file).read_text()).get("pieces", [])):
            self.set_busy(False, f"{run}: the scan ended before any 100x picture was taken; nothing to sort.")
            return
        how = "finished" if res.get("ok") and not res.get("stopped") else "stopped early: sorting the pictures taken so far"
        self.sort(run, self.job_crit, f"{run} {how}.  Sorting the 100x pictures ...", results=res.get("results") or None)

    def sort(self, run, c, status, results=None):
        self.set_busy(True, status)
        args = ["flakeid.gscan.collect", str(self.runs.get(run, run)), "--criteria", str(self.crit_file), "--settings", self.settings_file]
        if results:
            args += ["--results", results]

        def done(code):
            out = self.sorted_dir(run, results)
            if code == 0 and (out / "index.html").is_file():
                n = {}
                try:
                    n = json.loads((out / "flakes.json").read_text(encoding="utf-8")).get("counts", {})
                except (OSError, ValueError):
                    pass
                self.set_busy(False, f"{run}: {n.get('match', 0)} wanted, {n.get('possible', 0)} possibly wanted, {n.get('other', 0)} other real flakes, "
                                     f"{n.get('doubtful', 0)} not sure, {n.get('glue', 0)} glue.  Results: {out}")
                if c.copy_to:
                    self.copy_sorted(out, Path(c.copy_to) / run / "sorted")
                webbrowser.open((out / "index.html").as_uri())
            else:
                self.set_busy(False, f"{run}: the sorting failed (see the log below).")
        self.spawn(args, done)

    def sorted_dir(self, run, results=None):
        return (Path(results) if results else self.local / "results" / run) / "sorted"

    def copy_sorted(self, src, dst):
        def work():
            import shutil
            try:
                shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
                self.log(f"sorted results copied to {dst}")
            except Exception as e:
                self.log(f"sorted results NOT copied to {dst} ({type(e).__name__}: {e})")
        threading.Thread(target=work, daemon=True).start()

    def sort_again(self):
        c = self.criteria()
        run = self.v["run"].get()
        if c is None or self.busy or not run:
            return
        self.log(f"==== sort again {run}: {c.describe()}")
        self.sort(run, c, f"Sorting {run} again ...")

    def open_results(self):
        run = self.v["run"].get()
        p = self.sorted_dir(run) / "index.html"
        if run and p.is_file():
            webbrowser.open(p.as_uri())
        else:
            messagebox.showinfo("Glovebox flake scanner", f"{run or 'No scan'} has not been sorted yet: \"Sort again with the values above\" makes its page.",
                                parent=self.root)

    def check(self):
        if self.busy:
            return
        self.log("==== hardware check (nothing moves)")
        self.set_busy(True, "Checking the hardware (nothing moves) ...")
        self.spawn(["flakeid.scope.hw.check", "--settings", self.settings_file],
                   lambda code: self.set_busy(False, "Hardware check passed." if code == 0 else "Hardware check FAILED: see the log below (is Olympus Stream closed?)."))

    def close(self):
        if self.busy and self.proc is not None and self.proc.poll() is None:
            if not messagebox.askokcancel("Glovebox flake scanner", "A job is still running.  Closing this window does not stop it (use its own "
                                          "Stop button); its pictures will not be sorted automatically.  Close anyway?", parent=self.root):
                return
        self.root.destroy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", default=str(ROOT / "scan_settings.json"))
    a = ap.parse_args(argv)
    App(a.settings).root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
