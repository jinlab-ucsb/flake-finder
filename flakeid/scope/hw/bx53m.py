"""The glovebox microscope (Olympus BX53M) as a Micro-Manager-like "core" for the scan program.

The scan program (flakeid.scope.scan.Scanner, the windows in flakeid.scope.ui) talks to the microscope through one
object with the method names of Micro-Manager's CMMCore (pymmcore).  This PC has no Micro-Manager: the stage, the focus
and the camera are driven through their vendors' libraries (flakeid.scope.hw.tango, flakeid.scope.hw.dp22).  BX53MCore
wraps the two and offers the CMMCore methods the scan program uses, as flakeid.scope.mock.MockCore does for the
simulator.

    stage + focus   Marzhauser Tango (X, Y, Z in um; device labels "TangoXY", "TangoZ")
    camera          Olympus DP22 (1920 x 1440, 8-bit colour from its live stream; device label "DP22")

The nosepiece (BX3M-CBFM control box) is not part of the core: flakeid.scope.hw.cbfm, opened by the Scanner.

What differs from Micro-Manager, and how it is handled here:

* The Tango silently drops a move sent while an axis is still moving, and nothing says whether its library may be
  called from two threads at once.  Every library call is made under one lock, moves are sent without the library's
  own wait, and X, Y and Z moves are sent one after the other, each when the stage has stopped.  setXYPosition returns
  once the Y move is on its way (as Micro-Manager's does); waitForDevice waits until the stage has stopped.
* The controller has no Z limits configured (+-200 mm): the software Z limits of TangoStage (set_z_limits, kept in
  ~/.flakeid/tango_z_limits.json) are checked on every focus move, and a single focus move larger than max_z_move_um is
  refused whatever the caller asks.
* The DP22 has no snap: its live stream runs from connect to release.  The "sequence acquisition" calls only say
  which frames count: clearCircularBuffer forgets the frames so far, getRemainingImageCount counts the new ones,
  getLastImage gives the newest.  snapImage waits for the second new frame (the first may have been exposing before
  the call).
* Frames are (H, W, 3) uint8 RGB (the camera's processed picture: white-balanced, probably with a tone curve), not
  linear raw data.  The scan program handles such frames ("RGB32" tiles); the thickness calibration does not apply.
"""
from __future__ import annotations

import threading
import time

XY, Z, CAM = "TangoXY", "TangoZ", "DP22"
RAW_BLACK, RAW_BLACK_CAMERA = 128, 41            # black level handed out in raw mode / the DP22's own (lamp off, 2026-11-15)


class HardwareError(RuntimeError):
    pass


class BX53MCore:
    def __init__(self, port="COM3", baud=57600, tango_dll="", stream_dir="", max_z_move_um=500.0, z_limit_file="",
                 stage_factory=None, camera_factory=None, xy_scale=1.0, wb_gains=None):
        """stage_factory / camera_factory: callables returning an open TangoStage / DP22 (tests give stand-ins)."""
        self.port, self.baud, self.tango_dll, self.stream_dir = port, int(baud), tango_dll, stream_dir
        self.z_limit_file = z_limit_file
        self.max_z_move_um = float(max_z_move_um)
        # real um per unit of the controller's X / Y read-out (setting stage_xy_scale).  The glovebox Tango has a spindle
        # pitch of 1 mm configured and reports a travel of 37.75 x 25.25 "mm", but 40 of its "um" moved the picture by
        # about 82 um (Olympus Stream's pixel calibration, 2026-11-13): the stage is a 75 x 50 mm one with 2 mm spindles.
        # Everything outside this class is in real um; the controller's configuration is not changed (Stream uses it)
        self.xy_scale = float(xy_scale)
        # the camera's white balance gains (R, G, B), set every time the camera is opened (it forgets them); None: as
        # the camera is
        self.wb_gains = tuple(float(v) for v in wb_gains) if wb_gains else None
        # raw mode (hw/dp22.py): the frames the scan works with are the sensor's linear 12-bit mosaic, (H, W) uint16,
        # red at the top-left pixel, as the XC10's RAW16; the white balance gains are then 1 / 1 / 1 (they multiply
        # the raw values) and wb_gains are those of the colour pictures taken beside them (colourImage).  The camera's
        # black level (~41 counts) is raised to RAW_BLACK, the XC10's, which the scan program and the analysis assume
        # where no dark frame says otherwise
        self.raw = False
        self.colour_target = 0.486               # colour pictures: exposure that brings their median green to this
        self._colour_exp = {}                    # ... found once per objective
        # focus safety: the focus calibration (hw/zcal.py: each objective's focus; None = none), the nosepiece reader
        # that says which objective is in (set by the Scanner while it has it open) and the positions' names
        self.zcal = None
        self.nosepiece = None
        self.objective_names = {}
        self._obj = (0.0, None)
        # the focus axis moved slowly while connected (velocity rev/s, acceleration m/s2; 0 = the controller's own): at
        # the controller's 5 rev/s and 0.1 m/s2 it lost position on repeated moves (it has no encoder: found 2026-11-15)
        self.z_vel, self.z_accel = 0.0, 0.0
        self._z0 = None
        self.xy_accel = 0.0                      # X / Y acceleration set while connected (m/s2; 0 = the controller's)
        self._accel0 = None
        self._stage_factory, self._camera_factory = stage_factory, camera_factory
        self.stage = None
        self.cam = None
        self.lock = threading.RLock()            # every stage library call
        self.streaming = False                   # what isSequenceRunning says (the camera itself is always live)
        self._base = 0                           # frame counter at the last clearCircularBuffer
        self._snap = None
        self._exposure = None                    # the exposure last set (ms)
        self._timeout_ms = 60000
        self._target = {}                        # axis -> where it was last sent (um)
        self._t_sent = 0.0
        self._pending = False                    # a move was sent and has not been seen to end yet
        self._t_stop = 0.0                       # when the last move was seen to have ended
        # the DP22's live picture lags the stage: measured 2026-11-13, a frame 0.10 s after a move still showed the old
        # place, one 0.17-0.2 s after it the new one.  Frames taken for the scan wait this long after a move has ended
        self.camera_latency_s = 0.25

    # ---- loading / unloading (flakeid.scope.microscope.Microscope: devices open only while a job uses them)
    def _open_stage(self):
        if self._stage_factory is not None:
            return self._stage_factory()
        from . import tango
        return tango.TangoStage(port=self.port, baud=self.baud, dll_path=self.tango_dll or tango.DLL_PATH,
                                z_limit_file=self.z_limit_file or tango.Z_LIMIT_FILE)

    def _open_camera(self):
        if self._camera_factory is not None:
            cam = self._camera_factory()
        else:
            from . import dp22
            cam = dp22.DP22(stream_dir=self.stream_dir or dp22.STREAM_DIR)
        try:
            if self.raw:
                cam.set_white_balance(1.0, 1.0, 1.0)
                cam.set_mode(True)
            elif self.wb_gains:
                cam.set_white_balance(*self.wb_gains)
                got = cam.white_balance_gains()
                if max(abs(a - b) for a, b in zip(got, self.wb_gains)) > 0.01:
                    raise HardwareError(f"the camera's white balance gains read {got} after setting {self.wb_gains}")
        except BaseException:
            cam.close()
            raise
        return cam

    def colourImage(self, key=None):
        """The camera's own colour picture of what is in view, (H, W, 3) uint8, as Olympus Stream shows it: white
        balance wb_gains, the camera's tone curve, an exposure that brings the median green to colour_target (found the
        first time for each key, e.g. the objective, and used again).  The raw mode and its exposure are put back.
        About 0.5 s."""
        cam = self._cam()
        if not self.raw:
            self._after_move()
            self._wait_frames(2)
            return self.getLastImage()
        import numpy as np
        self._after_move()
        e_raw = self._exposure if self._exposure is not None else cam.exposure_ms()
        if key is None:
            key = self._obj[1]
        try:
            cam.set_mode(False)
            cam.set_white_balance(*(self.wb_gains or (1.0, 1.0, 1.0)))
            e = self._colour_exp.get(key)
            cam.set_exposure_ms(e or e_raw)
            f = None
            for k in range(1 if e else 8):
                time.sleep(0.1 + 2.5 * cam.exposure_ms() / 1000.0)      # (the frames under way have the old settings)
                self._wait_frames(2)
                f = cam.latest_frame()[0]
                if e:
                    break
                lv = float(np.median(f[..., 1])) / 255.0
                e1 = cam.exposure_ms()
                if abs(lv / self.colour_target - 1.0) < 0.05:
                    self._colour_exp[key] = e1
                    break
                cam.set_exposure_ms(min(1000.0, max(0.2, e1 * min(max(self.colour_target / max(lv, 0.01), 0.3), 3.0) ** 0.9)))
            else:
                self._colour_exp[key] = cam.exposure_ms()
            return np.array(f)
        finally:
            cam.set_white_balance(1.0, 1.0, 1.0)
            cam.set_mode(True)
            cam.set_exposure_ms(e_raw)
            self._wait_frames(2)
            self._base = self._count()

    def loadSystemConfiguration(self, cfg=None):
        self.unloadAllDevices()
        try:
            with self.lock:
                self.stage = self._open_stage()
                if self.xy_accel > 0:
                    self._accel0 = self.stage.accel()[0]
                    self.stage.set_xy_accel(self.xy_accel)
                if self.z_vel > 0 or self.z_accel > 0:
                    self._z0 = (self.stage.velocity()[2], self.stage.accel()[2])
                    self.stage.set_z_motion(self.z_vel or None, self.z_accel or None)
            self.cam = self._open_camera()
            self.cam.start_live()
            self.cam.wait_frame(timeout=10.0)
        except BaseException:
            self.unloadAllDevices()
            raise
        self.streaming, self._base, self._snap, self._exposure, self._target = False, 0, None, None, {}
        if not getattr(self, "_atexit", False):       # (a program that ends without releasing: the controller's own
            import atexit                           # speeds put back, the camera and the ports closed)
            atexit.register(self.unloadAllDevices)
            self._atexit = True

    def unloadAllDevices(self):
        cam, self.cam = self.cam, None
        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass
        with self.lock:
            st, self.stage = self.stage, None
            if st is not None:
                try:
                    if self._accel0:
                        st.set_xy_accel(self._accel0)        # (as it was: Olympus Stream and the joystick use it)
                        self._accel0 = None
                    if self._z0:
                        st.set_z_motion(*self._z0)
                        self._z0 = None
                except Exception:
                    pass
                try:
                    st.close()
                except Exception:
                    pass
        self.streaming = False

    def getLoadedDevices(self):
        return ([XY, Z] if self.stage is not None else []) + ([CAM] if self.cam is not None else [])

    # the camera closed and opened again (Scanner.reconnect_camera)
    def unloadDevice(self, d):
        if d == CAM:
            cam, self.cam = self.cam, None
            self.streaming = False
            if cam is not None:
                cam.close()

    def loadDevice(self, d, lib, name):
        if d != CAM:
            raise HardwareError(f"cannot load {d}")

    def initializeDevice(self, d):
        if d == CAM and self.cam is None:
            self.cam = self._open_camera()
            self.cam.start_live()
            self.cam.wait_frame(timeout=10.0)
            self._base, self._snap = 0, None

    def getDeviceLibrary(self, d): return "flakeid.scope.hw"
    def getDeviceName(self, d): return d
    def getParentLabel(self, d): return ""
    def setParentLabel(self, d, parent): pass
    def getVersionInfo(self): return "BX53MCore (flakeid.scope.hw: Tango stage, DP22 camera)"
    def getAPIVersionInfo(self): return "no Micro-Manager: vendor libraries through ctypes"
    def enableDebugLog(self, on): pass
    def setPrimaryLogFile(self, path, truncate=False): pass
    def setDeviceAdapterSearchPaths(self, paths): pass
    def getTimeoutMs(self): return self._timeout_ms
    def setTimeoutMs(self, ms): self._timeout_ms = int(ms)

    def getXYStageDevice(self): return XY
    def getFocusDevice(self): return Z
    def getCameraDevice(self): return CAM if self.cam is not None else ""

    def setCameraDevice(self, c):
        if c != CAM:
            raise HardwareError(f"no camera {c!r} on this microscope (only {CAM})")

    def _st(self):
        if self.stage is None:
            raise HardwareError("the stage is not connected")
        return self.stage

    def _cam(self):
        if self.cam is None:
            raise HardwareError("the camera is not connected")
        return self.cam

    # ---- stage and focus
    def _pos(self):
        with self.lock:
            return self._st().position()

    def getXPosition(self, dev=None): return self._pos()[0] * self.xy_scale
    def getYPosition(self, dev=None): return self._pos()[1] * self.xy_scale
    def getPosition(self, dev=None): return self._pos()[2]

    def _moving(self):
        with self.lock:
            return bool(self._st().is_moving())

    def _settled(self):
        """The stage has stopped: not moving, and either where it was last sent or the move had time to start (a move
        the controller did not take, or one that ended at a limit: the caller's read-back sees it)."""
        if self._moving():
            return False
        if time.time() - self._t_sent <= 0.5:
            p = dict(zip("xyz", self._pos()))
            if not all(abs(p[a] - v) <= (0.05 if a == "z" else 0.5) for a, v in self._target.items()):
                return False
        if self._pending:
            self._pending, self._t_stop = False, time.time()
        return True

    def _after_move(self):
        """Wait until the camera's picture shows where the stage is now (camera_latency_s after the last move ended)."""
        if self.stage is not None and self._pending:
            self._wait_stopped()
        dt = self._t_stop + self.camera_latency_s - time.time()
        if dt > 0:
            time.sleep(dt)

    def _wait_stopped(self, what="the stage"):
        t_end = time.time() + self._timeout_ms / 1000.0
        while not self._settled():
            if time.time() > t_end:
                raise HardwareError(f"Wait for device {what} timed out after {self._timeout_ms} ms")
            time.sleep(0.005)

    def _send(self, **axes):
        """One axis move sent (not waited for), when the stage has stopped (a move sent earlier is dropped otherwise)."""
        self._wait_stopped()
        with self.lock:
            self._st().move_abs(wait=False, force=True, **axes)
            self._target = {k: float(v) for k, v in axes.items()}
            self._t_sent = time.time()
            self._pending = True

    def setXYPosition(self, *a):
        x, y = float(a[-2]) / self.xy_scale, float(a[-1]) / self.xy_scale      # (controller units)
        cx, cy, _ = self._pos()
        if abs(x - cx) > 1e-4:
            self._send(x=x)
        if abs(y - cy) > 1e-4:
            self._send(y=y)

    def setPosition(self, *a):
        z = float(a[-1])
        cz = self._pos()[2]
        if abs(z - cz) > self.max_z_move_um:
            raise HardwareError(f"focus move of {z - cz:+.1f} um refused: more than {self.max_z_move_um:g} um at once "
                                "(max_z_move_um)")
        lo, hi = self.focusLimitsUm()
        if (lo is not None and z < lo) or (hi is not None and z > hi):
            raise HardwareError(f"focus move to {z:.1f} um refused: beyond the safety limit of the objective in place "
                                f"({lo} .. {hi}; flakeid.scope.hw.zcal)")
        if abs(z - cz) > 1e-5:
            self._send(z=z)                      # (TangoStage checks its software Z limits and raises outside them)

    def waitForDevice(self, dev):
        if dev in (XY, Z):
            self._wait_stopped(dev)

    def waitForSystem(self):
        self._wait_stopped()

    def deviceBusy(self, dev):
        return dev in (XY, Z) and not self._settled()

    def focusLimitsUm(self):
        """(z min, z max) the focus drive may be sent to, None = no limit: the stage's software Z limits (hw/zcal.py)."""
        st = self._st()
        lo, hi = getattr(st, "z_min", None), getattr(st, "z_max", None)
        if self.zcal:
            from .zcal import compute_limit, objective_limits
            now = time.time()
            if self.nosepiece is not None and now - self._obj[0] > 1.0:
                try:
                    self._obj = (now, self.objective_names.get(self.nosepiece.position()))
                except Exception:
                    self._obj = (now, None)
            name = self._obj[1] if self.nosepiece is not None else None
            per = objective_limits(**self.zcal)
            # (the objective in place: its own limit; not known: the tightest of them all)
            l2, h2 = per[name] if name in per else compute_limit(strict=True, **self.zcal)
            lo = l2 if lo is None else (lo if l2 is None else max(lo, l2))
            hi = h2 if hi is None else (hi if h2 is None else min(hi, h2))
        return lo, hi

    def stageLimitsUm(self, margin=0.0):
        """[x min, x max, y min, y max] the controller allows (um), margin inside them (Scanner.stage_box)."""
        with self.lock:
            (x0, x1), (y0, y1) = self._st().limits("x"), self._st().limits("y")
        k = self.xy_scale
        return [x0 * k + margin, x1 * k - margin, y0 * k + margin, y1 * k - margin]

    def stop(self, dev=None):
        with self.lock:
            self._st().stop()

    # ---- properties
    def _props(self, dev):
        if dev == CAM and self.cam is not None:
            try:
                wb = ", ".join(f"{v:.4g}" for v in self.cam.white_balance_gains())
            except Exception:
                wb = "?"
            return {"Data format": "RAW16" if self.raw else "RGB32", "Image data bit depth": "12" if self.raw else "8",
                    "Camera": getattr(self.cam, "name", CAM),
                    "White balance gains": wb,
                    "Exposure mode": "auto" if self._exposure is None else "manual"}
        return {}

    def getDevicePropertyNames(self, dev): return list(self._props(dev))
    def hasProperty(self, dev, name): return name in self._props(dev)
    def isPropertyReadOnly(self, dev, name): return True
    def isPropertyPreInit(self, dev, name): return False
    def getAllowedPropertyValues(self, dev, name): return []

    def getProperty(self, dev, name):
        try:
            return self._props(dev)[name]
        except KeyError:
            raise HardwareError(f"{dev} has no property {name!r}")

    def setProperty(self, dev, name, value):
        if str(self.getProperty(dev, name)) != str(value):
            raise HardwareError(f"{dev} property {name!r} cannot be set (it is {self.getProperty(dev, name)!r})")

    # ---- camera
    def getImageWidth(self): return int(self._cam().width)
    def getImageHeight(self): return int(self._cam().height)
    def getBytesPerPixel(self): return 2 if self.raw else 4
    def getImageBitDepth(self): return 12 if self.raw else 8
    def getNumberOfComponents(self): return 1 if self.raw else 4

    def _count(self):
        return self._cam().latest_frame()[1]

    def _wait_frames(self, n, timeout=5.0):
        """Wait for n frames that arrive from now on."""
        cam = self._cam()
        if not cam.live:
            cam.start_live()
        c0, t_end = self._count(), time.time() + timeout
        while self._count() < c0 + n:
            if time.time() > t_end:
                raise HardwareError(f'Error in device "{CAM}": no frame arrived within {timeout:g} s')
            time.sleep(0.002)

    def setExposure(self, *a):
        ms = float(a[-1])
        self._cam().set_exposure_ms(ms)
        self._exposure = ms
        self._wait_frames(2)                     # (the frames under way were exposed with the old time)

    def getExposure(self, *a):
        try:
            ms = float(self._cam().exposure_ms())
        except Exception:
            ms = float("nan")
        e = self._exposure
        # (the camera keeps the time as a fraction: what was asked for when it reads back the same within 2 %)
        if e is not None and (ms != ms or abs(ms - e) <= 0.02 * max(e, 1e-9)):
            return e
        return ms

    def isSequenceRunning(self, *a): return self.streaming

    def startContinuousSequenceAcquisition(self, interval_ms=0):
        cam = self._cam()
        if not cam.live:
            cam.start_live()
        self.streaming, self._base = True, self._count()

    def stopSequenceAcquisition(self, *a):
        self.streaming = False

    def clearCircularBuffer(self):
        self._after_move()                       # (the frames counted from here on show the stage where it is now)
        self._base = self._count()

    def getRemainingImageCount(self):
        return max(0, self._count() - self._base) if self.streaming and self.cam is not None else 0

    def getLastImage(self):
        f = self._cam().latest_frame()[0]
        if f is None:
            raise HardwareError(f'Error in device "{CAM}": no frame yet')
        if f.ndim == 2:                          # (raw: the black level raised to the XC10's, see __init__)
            import numpy as np
            return np.minimum(f + np.uint16(RAW_BLACK - RAW_BLACK_CAMERA), 4095).astype(np.uint16)
        return f

    def snapImage(self):
        self._after_move()
        self._wait_frames(2)
        self._snap = self.getLastImage()

    def getImage(self, *a):
        if self._snap is None:
            raise HardwareError("no picture taken yet (snapImage)")
        return self._snap


def make_core(s):
    """The core for the settings s (flakeid.scope.settings.Settings), nothing opened yet (loadSystemConfiguration)."""
    from .zcal import load_cal
    core = _make(s)
    core.zcal = load_cal()
    core.objective_names = {v: k for k, v in (s.objective_positions or {}).items()}
    core.xy_accel = float(s.stage_xy_accel or 0.0)
    core.z_vel, core.z_accel = float(s.stage_z_vel or 0.0), float(s.stage_z_accel or 0.0)
    core.raw = bool(s.camera_raw)
    core.colour_target = float(s.colour_target)
    return core


def _make(s):
    return BX53MCore(port=s.tango_port, baud=s.tango_baud, tango_dll=s.tango_dll, stream_dir=s.stream_dir,
                     max_z_move_um=s.max_z_move_um, xy_scale=s.stage_xy_scale, wb_gains=s.wb_gains)
