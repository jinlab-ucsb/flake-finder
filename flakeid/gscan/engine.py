"""One scan job for the glovebox scanner, in its own process: the original scan program's Scanner (stage, focus, camera,
autofocus, the focus safety limits: flakeid/scope/scan.py, unchanged) with the settings of a job file.

    python -m flakeid.gscan.engine --settings scan_settings.json --job JOB.json [--dry]

JOB.json: {"settings": {...}} (criteria.scan_settings: what to override in the settings file for this scan).  The job's
windows are the original program's: the live view where the scan area is drawn, then the log window with Stop / Pause.
When the job ends well the window closes by itself; after a stop or an error it stays, to be read, until closed.
Written beside the job file when the process ends: <job>.result.json {"ok", "stopped", "run", "run_file", "results"}.

--dry: no microscope; prints the settings the job would run with (a check of the job file).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    from ..scope import __main__ as M
    from ..scope.settings import dump, load
    job = json.loads(Path(a.job).read_text(encoding="utf-8"))
    ov = dict(job.get("settings") or {}, settings_file=a.settings)
    s = load(a.settings if Path(a.settings).is_file() else None, **ov)
    if s.hardware == "mm":
        raise SystemExit("the glovebox scanner needs a settings file with \"hardware\": \"bx53m\" (tools\\bx53m\\scan_settings.json)")
    # (the 100x objective's ratio and offset for the scan and for the analysis process: flakeid/hwconst.py)
    os.environ["FLAKEID_RATIO_100X"] = repr(float(s.ratio_100x))
    os.environ["FLAKEID_OFFSET_100X"] = f"{float(s.offset_100x_um[0])!r},{float(s.offset_100x_um[1])!r}"
    if a.dry:
        d = dump(s)
        print(json.dumps({k: d[k] for k in sorted(set(ov) | {"hardware", "profile_raw", "flat10", "flat100", "analysis_workers",
                                                           "offset_100x_um", "focus_offset_um", "max_dz_um"}) if k in d}, indent=1))
        return 0
    M.enable_crash_log()
    other = M.running_instance(s.local_root)
    if other:
        raise SystemExit(f"a scan program is already running (process {other.get('pid')}, since {other.get('started', '?')}): "
                         "close it first")
    M.claim_instance(s.local_root, "glovebox scanner job")
    from ..scope.microscope import Microscope
    from ..scope.scan import Scanner
    from ..scope.ui import TkUI
    cfg = f"{s.hardware} (flakeid.scope.hw)"
    core = M.make_direct_core(s)
    ui = TkUI(title="Glovebox flake scanner: scan", theme=s.ui_theme)
    ui.microscope = Microscope(core, cfg)
    sc = Scanner(core, s, ui)
    state = dict(ok=False, done=False)
    run0 = sc.run

    def run():
        try:
            run0()
            state["ok"] = True
        finally:
            state["done"] = True
    sc.run = run

    def result():
        man = getattr(sc, "manifest", None)
        d = dict(ok=bool(state["ok"]), stopped=bool(getattr(sc, "stop_all", False) or ui.stop_requested),
                 run=getattr(sc, "run_label", None), run_file=str(man) if man else None,
                 results=str(getattr(sc, "analysis_out", "") or ""))
        try:
            Path(str(a.job)[:-5] + ".result.json" if str(a.job).endswith(".json") else str(a.job) + ".result.json").write_text(
                json.dumps(d, indent=1), encoding="utf-8")
        except OSError:
            pass
        return d

    def watch():                                # the job over, the results copied: a good job's window closes itself
        while not state["done"] or ui.job_running:
            time.sleep(0.5)
        t = getattr(sc, "upload_thread", None)
        if t is not None:
            t.join()
        result()
        if state["ok"] and not sc.stop_all:
            time.sleep(2.0)
            ui._post(ui.root.destroy)
    threading.Thread(target=watch, daemon=True).start()
    ui.run(sc)                                  # (returns when the window is closed)
    d = result()
    return 0 if d["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
