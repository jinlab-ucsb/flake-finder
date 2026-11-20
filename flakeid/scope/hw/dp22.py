"""Python control of the Olympus DP22 camera.

Calls the Olympus camera library CamDriver64.dll that ships with Stream Basic.
The library is undocumented: the function signatures below were worked out by
studying how Stream's own CamDP2x.dll calls it, so treat anything not listed
here as unknown. Stream Basic must be closed, since only one program can own
the camera.

    from flakeid.scope.hw.dp22 import DP22
    with DP22() as cam:
        cam.start_live()
        frame = cam.wait_frame()        # numpy array, height x width x 3, RGB
        cam.save("picture.png")
"""

import ctypes as C
import os
import threading
import time

import numpy as np

STREAM_DIR = r"C:\Program Files\Stream Basic"
CAMERA_DIR = os.path.join(STREAM_DIR, "Device", "camera")

# Callback event codes seen so far.
EVENT_INIT_DONE = 0
EVENT_LIVE_FRAME = 2

# Exposure modes for CAM_EXP_SetModeEx (found by experiment, see CLAUDE.md).
EXPOSURE_AUTO = 1
EXPOSURE_MANUAL = 0

# Raw mode (found 2026-11-13/15, docs/bx53m.md): CAM_LV_SetCCDImageTypeEx(h, 1, 1) makes the live buffer hold the
# sensor's own data, one 16-bit value per pixel (12 bits used), RAW_W x RAW_H pixels in rows of RAW_W: a Bayer mosaic
# with red at the top-left pixel.  The colour picture (type 0) is that frame from pixel (10, 9) on; the raw frame handed
# out is cut from (RAW_X0, RAW_Y0) = (10, 8), so that it has the colour picture's size, shows the same field to within
# one pixel and keeps red at its top-left pixel.  CAM_IP_SetContrastModeEx(h, 1, mode): the tone curve, applied to the
# raw data too; mode 1 (the camera's own) is compressive, mode 3 linear (black level ~41 counts).
RAW_W, RAW_H, RAW_X0, RAW_Y0 = 1940, 1456, 10, 8
CONTRAST_DEFAULT, CONTRAST_LINEAR = 1, 3

_CALLBACK = C.CFUNCTYPE(None, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p)


class _Shutter(C.Structure):
    """Exposure time in seconds as numerator / denominator."""
    _fields_ = [("num", C.c_uint), ("den", C.c_uint)]


class DP22Error(RuntimeError):
    pass


def _load(stream_dir=STREAM_DIR):
    camera_dir = os.path.join(stream_dir, "Device", "camera")
    os.add_dll_directory(camera_dir)
    os.add_dll_directory(stream_dir)
    if camera_dir not in os.environ["PATH"]:
        os.environ["PATH"] = camera_dir + ";" + stream_dir + ";" + os.environ["PATH"]
    d = C.CDLL(os.path.join(camera_dir, "CamDriver64.dll"))
    h, i, u = C.c_void_p, C.c_int, C.c_uint
    pu, pi, pd, dbl = C.POINTER(u), C.POINTER(i), C.POINTER(C.c_double), C.c_double
    d.CAM_EnumCameraEx.argtypes = [i, C.POINTER(C.c_char_p), i, i]
    d.CAM_CreateCameraEx.restype = h
    d.CAM_CreateCameraEx.argtypes = [C.c_char_p, i]
    d.CAM_DeleteCameraEx.argtypes = [h, i]
    d.CAM_RegisterCallbackEx.argtypes = [h, _CALLBACK, h]
    d.CAM_InitCameraEx.argtypes = [h, i, i]
    d.CAM_GetLastErrorEx.argtypes = [h, i, pu]
    d.CAM_LV_GetImageSizeEx.argtypes = [h, i, pu, pu]
    d.CAM_LV_SetImageSizeEx.argtypes = [h, i, i, i]
    d.CAM_LV_StartMovieEx.argtypes = [h, i, h]
    d.CAM_LV_ContinueMovieEx.argtypes = [h, i, h]
    d.CAM_LV_StopMovieEx.argtypes = [h, i]
    d.CAM_LV_SetCCDImageTypeEx.argtypes = [h, i, i]
    d.CAM_IP_SetContrastModeEx.argtypes = [h, i, i]
    d.CAM_EXP_SetModeEx.argtypes = [h, i, i]
    d.CAM_EXP_GetModeEx.argtypes = [h, i, pi]
    d.CAM_EXP_SetManualShutterSpeedEx.argtypes = [h, i, C.POINTER(_Shutter)]
    d.CAM_EXP_GetManualShutterSpeedEx.argtypes = [h, i, C.POINTER(_Shutter)]
    d.CAM_EXP_GetAeShutterSpeedEx.argtypes = [h, i, C.POINTER(_Shutter)]
    d.CAM_EXP_SetIsoEx.argtypes = [h, i, i]
    d.CAM_EXP_GetIsoEx.argtypes = [h, i, pi]
    d.CAM_WB_SetModeEx.argtypes = [h, i, i]
    d.CAM_WB_GetModeEx.argtypes = [h, i, pi]
    d.CAM_WB_SetRGBGainEx.argtypes = [h, i, dbl, dbl, dbl]
    d.CAM_WB_GetRGBGainEx.argtypes = [h, i, pd, pd, pd]
    return d


class DP22:
    def __init__(self, init_timeout=10.0, stream_dir=STREAM_DIR):
        self._dll = _load(stream_dir)
        self._h = None
        self._live = False
        self.raw = False  # raw mode: frames are (height, width) uint16 Bayer mosaics instead of RGB pictures
        self._lock = threading.Lock()
        self._frame = None
        self._frame_count = 0
        self._new_frame = threading.Event()
        self._init_done = threading.Event()
        self._callback = _CALLBACK(self._on_event)  # keep a reference alive

        self._dll.CAM_InitializeSDK()
        names = [C.create_string_buffer(33) for _ in range(10)]
        arr = (C.c_char_p * 10)(*[C.cast(b, C.c_char_p) for b in names])
        self._dll.CAM_EnumCameraEx(1, arr, 10, 33)
        found = [b.value for b in names if b.value]
        if not found:
            self._dll.CAM_TerminateSDK()
            raise DP22Error("No camera found (is Stream Basic still open? is the camera on?)")
        self.name = found[0].decode()

        self._h = self._dll.CAM_CreateCameraEx(found[0], 1)
        if not self._h:
            self._dll.CAM_TerminateSDK()
            raise DP22Error(f"Could not open camera {self.name}")
        try:
            self._ok(self._dll.CAM_RegisterCallbackEx(self._h, self._callback, None), "register callback")
            self._ok(self._dll.CAM_InitCameraEx(self._h, 0, 1), "initialise camera")
            if not self._init_done.wait(init_timeout):
                raise DP22Error("Camera did not finish initialising")
            w, h = C.c_uint(0), C.c_uint(0)
            self._ok(self._dll.CAM_LV_GetImageSizeEx(self._h, 1, C.byref(w), C.byref(h)), "get image size")
            self.width, self.height = w.value, h.value
            self._buf = (C.c_ubyte * (self.width * self.height * 4))()
        except Exception:
            self.close()
            raise

    # -- internals ---------------------------------------------------------

    def _ok(self, result, what):
        if not result:
            err = C.c_uint(0)
            self._dll.CAM_GetLastErrorEx(self._h, 1, C.byref(err))
            raise DP22Error(f"Camera call failed: {what} (error {err.value})")

    def _on_event(self, cam, event, a3, a4, a5, ctx):
        # Runs on a thread owned by the camera library.
        if event == EVENT_INIT_DONE:
            self._init_done.set()
        elif event == EVENT_LIVE_FRAME:
            if self.raw:
                m = np.frombuffer(self._buf, dtype=np.uint16, count=RAW_W * RAW_H).reshape(RAW_H, RAW_W)
                rgb = m[RAW_Y0:RAW_Y0 + self.height, RAW_X0:RAW_X0 + self.width].copy()
            else:
                raw = np.frombuffer(self._buf, dtype=np.uint8).reshape(self.height, self.width, 4)
                rgb = raw[:, :, 2::-1].copy()  # camera delivers B, G, R, unused
            with self._lock:
                self._frame = rgb
                self._frame_count += 1
            self._new_frame.set()
            if self._live:
                self._dll.CAM_LV_ContinueMovieEx(self._h, 0, self._buf)

    # -- live view and pictures ---------------------------------------------

    def start_live(self):
        if not self._live:
            self._live = True
            self._new_frame.clear()
            try:
                self._ok(self._dll.CAM_LV_StartMovieEx(self._h, 0, self._buf), "start live view")
            except Exception:
                self._live = False
                raise

    def stop_live(self):
        if self._live:
            self._live = False
            self._dll.CAM_LV_StopMovieEx(self._h, 1)

    def set_mode(self, raw, contrast=None):
        """Raw mosaic frames (raw=True) or the colour picture, and the tone curve (contrast mode; None: linear for raw,
        the camera's own for the picture).  The live stream is stopped for it and started again if it was running."""
        if contrast is None:
            contrast = CONTRAST_LINEAR if raw else CONTRAST_DEFAULT
        was = self._live
        self.stop_live()
        self._ok(self._dll.CAM_LV_SetCCDImageTypeEx(self._h, 1, 1 if raw else 0), "set image type")
        self._ok(self._dll.CAM_IP_SetContrastModeEx(self._h, 1, int(contrast)), "set contrast mode")
        self.raw = bool(raw)
        with self._lock:
            self._frame = None
        if was:
            self.start_live()

    @property
    def live(self):
        return self._live

    def latest_frame(self):
        """Return (frame, counter). frame is None until the first image arrives."""
        with self._lock:
            return self._frame, self._frame_count

    def wait_frame(self, timeout=5.0, fresh=True):
        """Return the next live frame as an RGB numpy array."""
        if not self._live:
            self.start_live()
        if fresh:
            self._new_frame.clear()
        if not self._new_frame.wait(timeout):
            raise DP22Error("No frame arrived from the camera")
        return self.latest_frame()[0]

    def save(self, path, fresh=True):
        """Save a full-resolution picture (format from the file extension)."""
        from PIL import Image
        frame = self.wait_frame(fresh=fresh)
        Image.fromarray(frame).save(path)
        return path

    # -- exposure and colour -------------------------------------------------

    def set_auto_exposure(self, on=True):
        self._ok(self._dll.CAM_EXP_SetModeEx(self._h, 1, EXPOSURE_AUTO if on else EXPOSURE_MANUAL), "set exposure mode")

    def exposure_mode(self):
        v = C.c_int(-1)
        self._ok(self._dll.CAM_EXP_GetModeEx(self._h, 1, C.byref(v)), "get exposure mode")
        return v.value

    def set_exposure_ms(self, ms):
        """Switch to manual exposure with the given time in milliseconds."""
        self._ok(self._dll.CAM_EXP_SetModeEx(self._h, 1, EXPOSURE_MANUAL), "set exposure mode")
        s = _Shutter(max(1, int(round(ms * 1000))), 1_000_000)
        self._ok(self._dll.CAM_EXP_SetManualShutterSpeedEx(self._h, 1, C.byref(s)), "set exposure time")

    def exposure_ms(self):
        s = _Shutter(0, 1)
        fn = self._dll.CAM_EXP_GetManualShutterSpeedEx if self.exposure_mode() == EXPOSURE_MANUAL \
            else self._dll.CAM_EXP_GetAeShutterSpeedEx
        self._ok(fn(self._h, 1, C.byref(s)), "get exposure time")
        return 1000.0 * s.num / s.den if s.den else float("nan")

    def set_white_balance_gains(self, r, g, b):
        self._ok(self._dll.CAM_WB_SetRGBGainEx(self._h, 1, float(r), float(g), float(b)), "set white balance")

    def set_white_balance(self, r, g, b, mode=2):
        """Fixed white balance gains.  The camera refuses gains in its mode 0 (error 191; the mode it is in when opened);
        in mode 2 it takes them (found 2026-11-13).  Neither survives closing the camera: set after every open."""
        self._ok(self._dll.CAM_WB_SetModeEx(self._h, 1, int(mode)), "set white balance mode")
        time.sleep(0.3)
        self.set_white_balance_gains(r, g, b)

    def white_balance_mode(self):
        v = C.c_int(-1)
        self._ok(self._dll.CAM_WB_GetModeEx(self._h, 1, C.byref(v)), "get white balance mode")
        return v.value

    def white_balance_gains(self):
        r, g, b = C.c_double(), C.c_double(), C.c_double()
        self._ok(self._dll.CAM_WB_GetRGBGainEx(self._h, 1, C.byref(r), C.byref(g), C.byref(b)), "get white balance")
        return r.value, g.value, b.value

    # -- teardown ------------------------------------------------------------

    def close(self):
        if self._h:
            self.stop_live()
            time.sleep(0.2)  # let an in-flight frame callback finish
            self._dll.CAM_DeleteCameraEx(self._h, 1)
            self._h = None
            self._dll.CAM_TerminateSDK()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    with DP22() as cam:
        print(f"camera {cam.name}, {cam.width} x {cam.height}")
        cam.start_live()
        frame = cam.wait_frame()
        print("frame", frame.shape, "mean R,G,B =", frame.reshape(-1, 3).mean(0).round(1))
        print("saved", cam.save("dp22_test.png"))
