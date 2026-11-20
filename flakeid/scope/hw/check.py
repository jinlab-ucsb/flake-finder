"""Check the glovebox microscope's devices through the scan program's core, WITHOUT moving anything.

    py -3.9 -m flakeid.scope.hw.check [--settings scan_settings.json] [--out FOLDER]

Close Olympus Stream first (it holds the stage's port and the camera).  It opens the stage, the camera and the
nosepiece as the scan program does and reports: stage position and travel, the software Z limits, the camera's frame
size, frame rate and levels at the exposure in use, the exposure read back after setting it, a frame as the scan
program takes it, and the nosepiece position.  A picture (check_frame.png) is saved to --out.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", help="settings file (JSON)")
    ap.add_argument("--out", default=".", help="folder for check_frame.png")
    ap.add_argument("--exposure", type=float, help="also set this exposure (ms) and read it back")
    a = ap.parse_args(argv)
    from ..settings import load
    from .bx53m import CAM, XY, Z, make_core
    s = load(a.settings, hardware="bx53m")
    ok = True
    core = make_core(s)
    print(f"connecting: stage {s.tango_port} at {s.tango_baud} baud, camera DP22 ...", flush=True)
    t0 = time.time()
    try:
        core.loadSystemConfiguration()
    except Exception as e:
        print(f"FAILED to connect: {type(e).__name__}: {e}\n(is Olympus Stream still open?)")
        return 1
    try:
        print(f"connected in {time.time() - t0:.1f} s; devices {core.getLoadedDevices()}")
        st = core.stage
        print(f"stage: {st.version()}")
        print(f"  position x {core.getXPosition(XY):.1f}  y {core.getYPosition(XY):.1f}  z {core.getPosition(Z):.2f} um")
        for ax in "xyz":
            lo, hi = st.limits(ax)
            print(f"  {ax} travel (controller): {lo:.1f} .. {hi:.1f} um")
        print(f"  software Z limits: min {st.z_min}  max {st.z_max}" +
              ("   <-- none set: only max_dz_um / max_z_move_um restrain the focus" if st.z_min is None and st.z_max is None else ""))
        print(f"  moving: {core.deviceBusy(XY)}")
        W, H = core.getImageWidth(), core.getImageHeight()
        print(f"camera: {core.getProperty(CAM, 'Camera')}, {W} x {H}, exposure {core.getExposure():.3f} ms "
              f"({core.getProperty(CAM, 'Exposure mode')})")
        if a.exposure:
            core.setExposure(a.exposure)
            print(f"  exposure set to {a.exposure:g} ms, reads back {core.getExposure():.4f} ms (camera: {core.cam.exposure_ms():.4f})")
        core.startContinuousSequenceAcquisition(0)
        core.clearCircularBuffer()
        t0, n0 = time.time(), core.getRemainingImageCount()
        time.sleep(2.0)
        n = core.getRemainingImageCount() - n0
        print(f"  stream: {n} frames in 2 s ({n / 2.0:.1f} per second)")
        t0 = time.time()
        core.snapImage()
        img = np.asarray(core.getImage())
        print(f"  snap: {img.shape} {img.dtype} in {time.time() - t0:.2f} s; mean R, G, B = "
              + ", ".join(f"{v:.1f}" for v in img.reshape(-1, img.shape[-1]).mean(0))
              + f"; saturated {100 * (img >= 254).any(axis=2).mean():.1f} % of the pixels")
        core.stopSequenceAcquisition()
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.fromarray(img).save(out / "check_frame.png")
        print(f"  saved {out / 'check_frame.png'}")
    except Exception as e:
        ok = False
        import traceback
        traceback.print_exc()
        print(f"FAILED: {type(e).__name__}: {e}")
    finally:
        core.unloadAllDevices()
        print("stage and camera released")
    if s.nosepiece_port:
        try:
            from .cbfm import CBFMNosepiece
            with CBFMNosepiece(s.nosepiece_port) as n:
                pos = n.position()
                name = next((k for k, v in (s.objective_positions or {}).items() if v == pos), "not in objective_positions")
                print(f"nosepiece on {s.nosepiece_port}: {n.unit}; position {pos} ({name})")
        except Exception as e:
            ok = False
            print(f"FAILED nosepiece: {type(e).__name__}: {e}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
