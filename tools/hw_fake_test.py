"""Test the glovebox microscope's core (flakeid.scope.hw.bx53m.BX53MCore) without hardware.

    python tools/hw_fake_test.py

Stand-ins for the Tango stage and the DP22 camera behave as the real ones do where it matters: the stage takes time to
move and silently drops a move sent while it is moving; the camera delivers frames from its own thread, and a frame
shows a sharp pattern only near the focus.  The test runs the core's calls and then the scan program's own hardware
layer on it (Scanner: devices, camera settings, exposure check, stage moves with read-back, focus moves, frames,
autofocus, a tile written and read back by the analysis).
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flakeid.scope.hw.bx53m import CAM, XY, Z, BX53MCore, HardwareError  # noqa: E402

FOCUS = 350.0


class FakeStage:
    """Axes move at a finite speed; a move sent while any axis moves is dropped (as the Tango does)."""
    SPEED = {"x": 20000.0, "y": 20000.0, "z": 2000.0}      # um / s
    LIMITS = {"x": (0.0, 37753.8), "y": (0.0, 25250.7), "z": (-200000.0, 200000.0)}

    def __init__(self):
        self.p = {"x": 13000.0, "y": 14000.0, "z": 340.0}
        self.move = None                                    # (axis, start, target, t0)
        self.z_min = self.z_max = None
        self.dropped = 0
        self.calls = threading.Lock()                       # (two library calls at once: a failure of the core's lock)
        self.closed = False

    def _call(self):
        if not self.calls.acquire(blocking=False):
            raise AssertionError("two stage library calls at the same time")
        self.calls.release()

    def _update(self):
        if self.move:
            ax, a, b, t0 = self.move
            d = self.SPEED[ax] * (time.time() - t0)
            if d >= abs(b - a):
                self.p[ax], self.move = b, None
            else:
                self.p[ax] = a + np.sign(b - a) * d

    def version(self): return "fake Tango"
    def position(self): self._call(); self._update(); return self.p["x"], self.p["y"], self.p["z"]
    def limits(self, axis): return self.LIMITS[axis]
    def is_moving(self): self._call(); self._update(); return self.move is not None

    def move_abs(self, x=None, y=None, z=None, wait=True, force=False):
        self._call()
        self._update()
        for ax, v in (("x", x), ("y", y), ("z", z)):
            if v is None:
                continue
            lo, hi = self.LIMITS[ax]
            if ax == "z":
                lo = lo if self.z_min is None else max(lo, self.z_min)
                hi = hi if self.z_max is None else min(hi, self.z_max)
            if not lo <= v <= hi:
                raise ValueError(f"{ax} target {v:.1f} um is outside limits {lo:.1f}..{hi:.1f}")
            if self.move is not None:
                self.dropped += 1                           # silently dropped
                continue
            self.move = (ax, self.p[ax], float(v), time.time())

    def stop(self): self._update(); self.move = None
    def close(self): self.closed = True


class FakeCamera:
    """Frames at 25 per second from a thread: a checkerboard blurred by the defocus, brightness by the exposure."""
    width, height, name = 480, 360, "FAKE:1"

    def __init__(self, stage):
        self.stage, self.live, self._n, self._frame = stage, False, 0, None
        self.ms, self.mode = 300.0, 1
        self._lock = threading.Lock()
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        self.pattern = (((xx // 12) + (yy // 12)) % 2).astype(np.float32)
        self._th = None

    def _run(self):
        import cv2
        while self.live:
            time.sleep(0.04)
            z = self.stage.p["z"]
            img = self.pattern
            sig = 0.6 * abs(z - FOCUS)
            if sig > 0.05:
                img = cv2.GaussianBlur(img, (0, 0), min(sig, 30.0))
            level = min(255.0, 14.0 * self.ms)
            f = np.clip(level * (0.5 + 0.4 * img), 0, 255).astype(np.uint8)
            rgb = np.stack([f // 2, f, (f * 0.9).astype(np.uint8)], 2)
            with self._lock:
                self._frame, self._n = rgb, self._n + 1

    def start_live(self):
        if not self.live:
            self.live = True
            self._th = threading.Thread(target=self._run, daemon=True)
            self._th.start()

    def stop_live(self): self.live = False

    def latest_frame(self):
        with self._lock:
            return self._frame, self._n

    def wait_frame(self, timeout=5.0, fresh=True):
        self.start_live()
        n0, t_end = self._n, time.time() + timeout
        while self._n <= n0:
            if time.time() > t_end:
                raise RuntimeError("no frame")
            time.sleep(0.005)
        return self._frame

    def set_exposure_ms(self, ms): self.ms, self.mode = float(ms), 0
    def exposure_ms(self): return self.ms * (1 + 1e-4)      # (read back as a slightly different fraction)
    def close(self): self.live = False


class QuietUI:
    """What the Scanner needs of a UI for its hardware layer."""
    stop_requested = pause_requested = False
    headless_analysis_off = True
    preview = None
    def log(self, msg): pass
    def message(self, title, msg): print("  [message]", title, msg)


def check(name, cond, extra=""):
    print(f"  {'OK    ' if cond else 'FAILED'} {name}{('  ' + extra) if extra else ''}")
    if not cond:
        check.failed += 1


check.failed = 0


def main():
    stage = FakeStage()
    core = BX53MCore(stage_factory=lambda: stage, camera_factory=lambda: FakeCamera(stage))
    print("core on stand-in devices")
    core.loadSystemConfiguration()
    check("devices loaded", core.getLoadedDevices() == [XY, Z, CAM])

    # ---- moves: X and Y both arrive although the stage drops overlapping moves
    t0 = time.time()
    core.setXYPosition(XY, 13500.0, 14200.0)
    core.waitForDevice(XY)
    x, y = core.getXPosition(XY), core.getYPosition(XY)
    check("XY move arrives on both axes", abs(x - 13500) < 0.5 and abs(y - 14200) < 0.5, f"{x:.1f}, {y:.1f} in {time.time() - t0:.2f} s")
    core.setXYPosition(XY, 13000.0, 14000.0)                # a second move right away, then a Z move during it
    core.setPosition(Z, 345.0)
    core.waitForDevice(XY); core.waitForDevice(Z)
    p = stage.position()
    check("moves sent one after the other, none dropped", stage.dropped == 0 and np.allclose(p, (13000, 14000, 345), atol=0.5), str(p))
    check("deviceBusy false when stopped", not core.deviceBusy(XY))

    # ---- polling from another thread while a move runs (the windows read the position while the job moves)
    err = []

    def poll():
        try:
            for _ in range(200):
                core.getXPosition(XY); core.deviceBusy(XY)
        except BaseException as e:
            err.append(e)
    th = threading.Thread(target=poll)
    th.start()
    for k in range(5):
        core.setXYPosition(XY, 13000.0 + 100 * (k + 1), 14000.0)
        core.waitForDevice(XY)
    th.join()
    check("stage library calls never overlap between threads", not err, str(err[:1]))

    # ---- a controller whose X / Y units are half the real distance (xy_scale 2, the glovebox stage)
    core.xy_scale = 2.0
    x_real = core.getXPosition(XY)
    core.setXYPosition(XY, x_real + 80.0, core.getYPosition(XY))
    core.waitForDevice(XY)
    check("xy_scale: 80 um asked = 40 controller units", abs(stage.position()[0] - (x_real + 80.0) / 2.0) < 0.5
          and abs(core.getXPosition(XY) - x_real - 80.0) < 1.0 and core.stageLimitsUm(0.0)[1] == 2 * 37753.8)
    core.xy_scale = 1.0

    # ---- Z guards
    try:
        core.setPosition(Z, 345.0 + 2000.0)
        check("large single focus move refused", False)
    except HardwareError:
        check("large single focus move refused", True)
    stage.z_max = 400.0
    try:
        core.setPosition(Z, 450.0)
        check("software Z limit refuses a move beyond it", False)
    except (ValueError, HardwareError):
        check("software Z limit refuses a move beyond it", True)
    # per-objective limits from a focus calibration: the objective in place decides; not known: the tightest
    class Nose:
        pos = 2
        def position(self): return self.pos
    stage.z_max = None
    core.zcal = dict(focus={"10x": 345.0, "100x": 330.0}, toward_plus=True)
    core.objective_names = {2: "10x", 5: "100x"}
    check("focus limit, objective not known: the tightest", core.focusLimitsUm() == (None, 380.0), str(core.focusLimitsUm()))
    core.nosepiece = Nose()
    check("focus limit at 10x", core.focusLimitsUm() == (None, 645.0), str(core.focusLimitsUm()))
    core.nosepiece.pos, core._obj = 5, (0.0, None)
    check("focus limit at 100x", core.focusLimitsUm() == (None, 380.0), str(core.focusLimitsUm()))
    try:
        core.setPosition(Z, 440.0)
        check("move beyond the 100x limit refused", False)
    except HardwareError:
        check("move beyond the 100x limit refused", True)
    core.zcal, core.nosepiece = None, None
    stage.z_max = 400.0
    check("focus did not move on the refused moves", abs(core.getPosition(Z) - 345.0) < 0.05)
    lim = core.stageLimitsUm(20.0)
    check("stage travel reported", lim == [20.0, 37733.8, 20.0, 25230.7], str(lim))

    # ---- camera
    core.setExposure(5.0)
    check("exposure reads back as set", core.getExposure() == 5.0)
    core.startContinuousSequenceAcquisition(0)
    core.clearCircularBuffer()
    check("buffer empty after clear", core.getRemainingImageCount() == 0)
    time.sleep(0.2)
    check("frames counted after clear", core.getRemainingImageCount() >= 3)
    core.snapImage()
    img = core.getImage()
    check("frame is (H, W, 3) uint8", img.shape == (360, 480, 3) and img.dtype == np.uint8)

    # ---- the scan program's hardware layer on this core
    from flakeid.scope.scan import Scanner
    from flakeid.scope.settings import load
    tmp = Path(tempfile.mkdtemp(prefix="hwfake_"))
    s = load(None, hardware="bx53m", local_root=str(tmp), verbose=False, nosepiece_port="", exposure_ms=5.0,
             camera_raw=False, save_colour=False, dark=0.0, target_level=0.25)      # (the stand-in camera gives colour pictures)
    sc = Scanner(core, s, QuietUI())
    sc.setup_devices()
    check("Scanner devices", (sc.xy, sc.zdev, sc.cam, sc.W, sc.H) == (XY, Z, CAM, 480, 360) and sc.z_scale == 1.0)
    check("Scanner knows the stage travel", sc.stage_box() == lim, sc.box_text())
    sc.apply_camera(s.exposure_ms)
    sc.DARK = s.dark * (sc.full_scale() + 1) / 4096.0
    check("full scale 255, no dark level", sc.full_scale() == 255.0 and sc.DARK == 0.0)
    sc.check_exposure(True)
    lv = sc.frame_level()
    check("auto exposure reaches the target level", abs(lv[0] / s.target_level - 1) < 0.2, f"level {lv[0]:.3f} at {core.getExposure():.2f} ms")
    sc.stage_self_test()
    sc.move_to(sc.X0 + 300.0, sc.Y0 - 200.0)
    check("Scanner.move_to with read-back", abs(core.getXPosition(XY) - sc.X0 - 300) < 0.5)
    try:
        sc.move_to(40000.0, sc.Y0)
        check("move beyond the stage travel refused", False)
    except Exception as e:
        check("move beyond the stage travel refused", type(e).__name__ in ("StageLimit", "Stop"), type(e).__name__)
    f = sc.frame()
    check("Scanner.frame", f.shape == (360, 480, 3))
    stage.z_max = None
    sc.view_scale = 1.0
    z, ratio, ok = sc.autofocus(sc.Z0, s.af10)
    check("autofocus finds the focus", ok and abs(z - FOCUS) < 1.5, f"Z {z:.2f} (true {FOCUS}), ratio {ratio:.2f}")

    # ---- a tile as the scan writes it, read by the analysis
    from flakeid import mm
    d = tmp / "t_tiles" / "Pos-000_000"
    d.mkdir(parents=True)
    sc.save_tile(d, 0, "Pos-000_000", 0, 0)
    t = mm.read_tile(d / "metadata.txt")
    fr = mm.read_frame(t)
    check("tile read back by the analysis (RGB32)", t.data_format == "RGB32" and fr.shape == (360, 480, 3) and mm.full_scale(t) == 255.0)
    check("tile colours in the right order", abs(float(fr[..., 1].mean()) - float(sc.last_pic["frame"][..., 1].mean())) < 0.5
          and fr[..., 0].mean() < fr[..., 2].mean() < fr[..., 1].mean())

    core.unloadAllDevices()
    check("devices released", stage.closed and core.getLoadedDevices() == [])
    print("ALL OK" if not check.failed else f"{check.failed} FAILED")
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main())
