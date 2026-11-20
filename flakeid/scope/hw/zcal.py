"""Focus calibration of the glovebox microscope: where each objective is in focus, and from that the limit the motorised
focus may never pass, so that the stage cannot be driven into an objective.

    py -3.9 -m flakeid.scope.hw.zcal [--settings scan_settings.json]

The microscope has two focus controls that do not know of each other: the manual one and the motorised one (the Tango's
Z axis, the only one the program sees).  The numbers here are only valid with THE MANUAL FOCUS AT ITS LIMIT, where it
must stay for every scan; calibrate again whenever it was moved, the controller's Z zero changed, or the samples sit
higher (another holder, much thicker chips).

What to do: with a chip of the usual kind on the stage, for each objective, lowest magnification first: focus with
the Z buttons (watch the stage from the side for the large steps), then "Save focus for this objective".  Say which way
Z + moves the stage.  "Write the limit" then stores, in the stage's software Z limits (hw/tango.py, checked on every
focus move of every program that uses it):

    limit = the focus nearest the objectives that any objective needed, less than MARGIN_UM beyond each one's focus

MARGIN_UM is small against the working distance of the high magnification objectives, so the limit holds whatever
objective is in.  A sample thicker than the one calibrated with comes closer to the objective by the difference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# how far beyond its own focus (towards the objective, in the controller's um) each objective may go
# (chips differ in height by tens of um and the autofocus searches around the focus: the margin must hold both, and
# stay well below the objective's working distance.  The limit applies per objective: BX53MCore.focusLimitsUm)
MARGIN_UM = {"5x": 300.0, "10x": 300.0, "20x": 200.0, "50x": 100.0, "100x": 50.0}   # (MPLN100x: working distance 0.21 mm)
MARGIN_DEFAULT = 50.0
STEPS = (1.0, 5.0, 20.0, 100.0, 500.0)
MAX_STEP_HIGH_MAG = 20.0            # 50x and above: no larger single step
CAL_FILE = os.path.join(os.path.expanduser("~"), ".flakeid", "bx53m_zcal.json")


def objective_limits(focus, toward_plus):
    """{objective: (z_min, z_max)}: each objective's own limit, its focus + its margin towards the objective."""
    out = {}
    for o, z in focus.items():
        m = MARGIN_UM.get(o, MARGIN_DEFAULT)
        out[o] = (None, round(z + m, 1)) if toward_plus else (round(z - m, 1), None)
    return out


def compute_limit(focus, toward_plus, strict=False):
    """One limit from {objective: focus z}: (z_min, z_max), one of them None.  strict: the tightest of the objectives'
    limits (safe whichever objective is in: used when the objective is not known); else the loosest (the ceiling the
    stage's own software limit holds for every program; the per-objective limits are applied inside it)."""
    if not focus:
        raise ValueError("no focus saved")
    L = objective_limits(focus, toward_plus).values()
    if toward_plus:
        hi = [h for _, h in L]
        return None, (min(hi) if strict else max(hi))
    lo = [l for l, _ in L]
    return (max(lo) if strict else min(lo)), None


def load_cal(path=None):
    """The saved calibration ({focus, toward_plus}), or None."""
    try:
        with open(path or CAL_FILE, encoding="utf-8") as f:
            d = json.load(f)
        return dict(focus={k: float(v) for k, v in d["focus"].items()}, toward_plus=d["z_plus_is"].startswith("towards"))
    except (OSError, ValueError, KeyError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", help="settings file (JSON)")
    a = ap.parse_args(argv)
    import tkinter as tk
    from tkinter import messagebox, ttk
    from PIL import Image, ImageTk
    from ..settings import load
    from . import dp22, tango
    from .cbfm import CBFMNosepiece
    s = load(a.settings, hardware="bx53m")
    names = {v: k for k, v in (s.objective_positions or {}).items()}

    stage = tango.TangoStage(port=s.tango_port, baud=s.tango_baud, dll_path=s.tango_dll or tango.DLL_PATH)
    old_limits = (stage.z_min, stage.z_max)
    stage.z_min = stage.z_max = None            # (the old limits belong to another position of the manual focus)
    cam = turret = None
    try:
        cam = dp22.DP22(stream_dir=s.stream_dir or dp22.STREAM_DIR)
        if s.wb_gains:
            cam.set_white_balance(*s.wb_gains)
        cam.set_auto_exposure(True)
        cam.start_live()
        turret = CBFMNosepiece(s.nosepiece_port)
        if s.illumination:
            turret.set_illumination(s.illumination, s.illumination_level)
    except Exception:
        for d in (turret, cam, stage):
            try:
                d and d.close()
            except Exception:
                pass
        raise

    root = tk.Tk()
    root.title("Focus calibration (manual focus at its limit)")
    view = tk.Label(root, bg="black")
    view.grid(row=0, column=0, padx=6, pady=6)
    side = ttk.Frame(root)
    side.grid(row=0, column=1, sticky="n", padx=6, pady=6)
    info = tk.StringVar()
    ttk.Label(side, textvariable=info, font=("Consolas", 12), justify="left").pack(anchor="w", pady=4)
    step = tk.DoubleVar(value=5.0)
    fr = ttk.LabelFrame(side, text="Z step (um, as the controller counts)")
    fr.pack(fill="x", pady=4)
    for v in STEPS:
        ttk.Radiobutton(fr, text=f"{v:g}", value=v, variable=step).pack(side="left", padx=3)
    status = tk.StringVar(value=f"Old Z limits {old_limits} are not applied in this window.")
    focus, state = {}, {"obj": None, "n": -1}

    def objective():
        return names.get(state["obj"], f"position {state['obj']}")

    def jog(sign):
        d = step.get()
        o = objective()
        if d > MAX_STEP_HIGH_MAG and o not in ("5x", "10x", "20x"):
            status.set(f"{o}: steps of at most {MAX_STEP_HIGH_MAG:g} um")
            return
        try:
            stage.move_rel(z=sign * d, wait=True, force=True)
        except Exception as e:
            status.set(f"move refused: {e}")

    fr = ttk.Frame(side)
    fr.pack(pady=4)
    ttk.Button(fr, text="Z +", width=10, command=lambda: jog(+1)).pack(side="left", padx=4)
    ttk.Button(fr, text="Z −", width=10, command=lambda: jog(-1)).pack(side="left", padx=4)
    ttk.Button(fr, text="STOP", width=8, command=lambda: stage.stop()).pack(side="left", padx=4)
    direction = tk.StringVar(value="")
    fr = ttk.LabelFrame(side, text="Z + moves the stage ...")
    fr.pack(fill="x", pady=4)
    ttk.Radiobutton(fr, text="towards the objective (up)", value="plus", variable=direction).pack(anchor="w")
    ttk.Radiobutton(fr, text="away from the objective (down)", value="minus", variable=direction).pack(anchor="w")
    fr = ttk.LabelFrame(side, text="Objective (hand switch, or turn it from here: the stage is NOT lowered)")
    fr.pack(fill="x", pady=4)

    def turn(n):
        try:
            turret.set_position(n)
        except Exception as e:
            status.set(f"nosepiece: {e}")
    for n in sorted(names):
        ttk.Button(fr, text=names[n], width=6, command=lambda n=n: turn(n)).pack(side="left", padx=2)
    saved = tk.StringVar(value="no focus saved yet")

    def save():
        o, z = objective(), stage.position()[2]
        focus[o] = round(z, 2)
        saved.set("\n".join(f"{k:>5s}: Z {v:9.2f}" for k, v in focus.items()))
        status.set(f"saved: {o} in focus at Z {z:.2f}")
    ttk.Button(side, text="Save focus for this objective", command=save).pack(fill="x", pady=4)
    ttk.Label(side, textvariable=saved, font=("Consolas", 11), justify="left").pack(anchor="w")

    def write():
        if direction.get() not in ("plus", "minus"):
            status.set("say first which way Z + moves the stage")
            return
        try:
            lo, hi = compute_limit(focus, direction.get() == "plus")
        except ValueError as e:
            status.set(str(e))
            return
        per = objective_limits(focus, direction.get() == "plus")
        lim = "; ".join(f"{o}: Z {'max' if v[1] is not None else 'min'} {v[1] if v[1] is not None else v[0]}" for o, v in per.items())
        if not messagebox.askyesno("Write the limit", f"Focus:\n{saved.get()}\n\nThe motorised focus will not go beyond {lim} "
                                                      "(valid only with the manual focus where it is now).\n\nWrite it?"):
            return
        stage.set_z_limits(z_min=lo, z_max=hi)
        os.makedirs(os.path.dirname(CAL_FILE), exist_ok=True)
        with open(CAL_FILE, "w", encoding="utf-8") as f:
            json.dump(dict(date=time.strftime("%Y-%m-%d %H:%M"), focus=focus, z_plus_is="towards the objective" if hi is not None
                           else "away from the objective", z_min=lo, z_max=hi, margin_um=MARGIN_UM,
                           note="valid with the manual focus at its limit"), f, indent=1)
        status.set(f"written: {lim}  ({tango.Z_LIMIT_FILE}; record {CAL_FILE})")
    ttk.Button(side, text="Write the limit", command=write).pack(fill="x", pady=8)
    ttk.Label(side, textvariable=status, wraplength=360, justify="left").pack(anchor="w", pady=4)

    def tick():
        try:
            f, n = cam.latest_frame()
            if f is not None and n != state["n"]:
                state["n"] = n
                g = f[::4, ::4, 1].astype(np.float32)
                sharp = float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) / max(float(g.mean()), 1.0)
                state["photo"] = ImageTk.PhotoImage(Image.fromarray(f[::2, ::2]))
                view.configure(image=state["photo"])
                x, y, z = stage.position()
                info.set(f"objective {objective()}\nZ {z:10.2f} um\nsharpness {1000 * sharp:6.1f}")
        except Exception as e:
            status.set(f"{type(e).__name__}: {e}")
        root.after(60, tick)

    def tick_obj():
        try:
            state["obj"] = turret.position()
        except Exception:
            pass
        root.after(500, tick_obj)

    def close():
        for d in (turret, cam, stage):
            try:
                d.close()
            except Exception:
                pass
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", close)
    tick_obj()
    tick()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
