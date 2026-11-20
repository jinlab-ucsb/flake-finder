"""Python control of the Marzhauser Tango stage controller (X, Y and Z focus).

Uses the vendor Tango_DLL.dll that ships with Olympus Stream Basic. Stream must
be closed (or have the stage removed from its device list), because only one
program can hold the COM port.

All positions are in micrometres.

    from flakeid.scope.hw.tango import TangoStage
    with TangoStage() as stage:
        print(stage.position())
        stage.move_rel(x=100)          # 100 um in +X, waits until done
        stage.move_abs(x=18000, y=13000)
"""

import ctypes as C
import json
import os

DLL_PATH = r"C:\Program Files\Stream Basic\Device\misc\Tango_DLL.dll"
COM_PORT = "COM3"
BAUD = 57600

AXES = {"x": 1, "y": 2, "z": 3}

# Largest single Z (focus) move allowed without force=True. The controller has
# no real Z limits configured (it reports +/-200 mm), so nothing else stops the
# stage from driving a sample into an objective.
MAX_Z_STEP_UM = 100.0

# Software Z limits chosen by the user, kept between runs. They are in the
# controller's own Z coordinates, so they are only valid while the controller's
# Z zero and the manual focus wheel stay where they were when the limit was set.
# The file is outside the program folder so that an update does not remove it.
Z_LIMIT_FILE = os.path.join(os.path.expanduser("~"), ".flakeid", "tango_z_limits.json")


class TangoError(RuntimeError):
    pass


class TangoStage:
    def __init__(self, port=COM_PORT, baud=BAUD, dll_path=DLL_PATH, z_limit_file=Z_LIMIT_FILE):
        self._dll = C.WinDLL(dll_path)
        self._z_limit_file = z_limit_file
        self._limits = {}  # controller travel limits per axis, read once
        self._declare()
        self._id = C.c_int(0)
        self._connected = False
        self.z_min = None  # software Z limits in um, None = not set
        self.z_max = None
        self._load_z_limits()
        self._check(self._dll.LSX_CreateLSID(C.byref(self._id)), "CreateLSID")
        try:
            self._check(
                self._dll.LSX_ConnectSimple(self._id, 1, port.encode(), baud, 0),
                f"connect to {port} (is Stream Basic still open?)",
            )
            self._connected = True
            # Make sure the controller reports and accepts micrometres.
            self._check(self._dll.LSX_SetDimensions(self._id, 1, 1, 1, 1), "SetDimensions")
        except Exception:
            self.close()
            raise

    # -- setup -------------------------------------------------------------

    def _declare(self):
        d, i, dbl = self._dll, C.c_int, C.c_double
        pd, pi = C.POINTER(C.c_double), C.POINTER(C.c_int)
        d.LSX_CreateLSID.argtypes = [pi]
        d.LSX_ConnectSimple.argtypes = [i, i, C.c_char_p, i, i]
        d.LSX_Disconnect.argtypes = [i]
        d.LSX_FreeLSID.argtypes = [i]
        d.LSX_SetDimensions.argtypes = [i, i, i, i, i]
        d.LSX_GetPos.argtypes = [i, pd, pd, pd, pd]
        d.LSX_GetVel.argtypes = [i, pd, pd, pd, pd]
        d.LSX_SetVel.argtypes = [i, dbl, dbl, dbl, dbl]
        d.LSX_GetAccel.argtypes = [i, pd, pd, pd, pd]
        d.LSX_SetAccel.argtypes = [i, dbl, dbl, dbl, dbl]
        d.LSX_GetLimit.argtypes = [i, i, pd, pd]
        d.LSX_MoveAbsSingleAxis.argtypes = [i, i, dbl, i]
        d.LSX_MoveRelSingleAxis.argtypes = [i, i, dbl, i]
        d.LSX_StopAxes.argtypes = [i]
        d.LSX_GetStatusAxis.argtypes = [i, C.c_char_p, i]
        d.LSX_GetTangoVersion.argtypes = [i, C.c_char_p, i]

    def _check(self, code, what):
        if code != 0:
            raise TangoError(f"Tango error {code} during {what}")

    # -- queries -----------------------------------------------------------

    def version(self):
        buf = C.create_string_buffer(256)
        self._check(self._dll.LSX_GetTangoVersion(self._id, buf, 256), "GetTangoVersion")
        return buf.value.decode(errors="replace")

    def position(self):
        """Return (x, y, z) in micrometres."""
        x, y, z, a = (C.c_double() for _ in range(4))
        self._check(
            self._dll.LSX_GetPos(self._id, C.byref(x), C.byref(y), C.byref(z), C.byref(a)),
            "GetPos",
        )
        return x.value, y.value, z.value

    def limits(self, axis):
        """Return (min, max) travel limits of 'x', 'y' or 'z' in micrometres."""
        if axis not in self._limits:
            lo, hi = C.c_double(), C.c_double()
            self._check(
                self._dll.LSX_GetLimit(self._id, AXES[axis], C.byref(lo), C.byref(hi)), "GetLimit"
            )
            self._limits[axis] = (lo.value, hi.value)
        return self._limits[axis]

    def velocity(self):
        """Return (x, y, z) speeds in motor revolutions per second."""
        x, y, z, a = (C.c_double() for _ in range(4))
        self._check(
            self._dll.LSX_GetVel(self._id, C.byref(x), C.byref(y), C.byref(z), C.byref(a)),
            "GetVel",
        )
        return x.value, y.value, z.value

    def accel(self):
        """Return the (x, y, z, a) accelerations in m/s2."""
        v = [C.c_double() for _ in range(4)]
        self._check(self._dll.LSX_GetAccel(self._id, *[C.byref(q) for q in v]), "GetAccel")
        return tuple(q.value for q in v)

    def set_xy_accel(self, a):
        """Set the X and Y acceleration (m/s2); Z and the fourth axis stay as they are."""
        cur = self.accel()
        self._check(self._dll.LSX_SetAccel(self._id, float(a), float(a), cur[2], cur[3]), "SetAccel")

    def set_z_motion(self, vel=None, accel=None):
        """Set the Z (focus) axis's velocity (motor revolutions per second) and / or acceleration (m/s2); X, Y stay."""
        if vel is not None:
            v = self.velocity()
            self._check(self._dll.LSX_SetVel(self._id, v[0], v[1], float(vel), 0.0), "SetVel")
        if accel is not None:
            a = self.accel()
            self._check(self._dll.LSX_SetAccel(self._id, a[0], a[1], float(accel), a[3]), "SetAccel")

    def is_moving(self):
        buf = C.create_string_buffer(64)
        self._check(self._dll.LSX_GetStatusAxis(self._id, buf, 64), "GetStatusAxis")
        return b"M" in buf.value[:3]

    # -- motion ------------------------------------------------------------

    def move_abs(self, x=None, y=None, z=None, wait=True, force=False):
        """Move to absolute coordinates (um). Axes left as None do not move."""
        cur = dict(zip("xyz", self.position()))
        targets = {k: v for k, v in (("x", x), ("y", y), ("z", z)) if v is not None}
        self._validate(targets, cur, force)
        for axis, value in targets.items():
            self._check(
                self._dll.LSX_MoveAbsSingleAxis(self._id, AXES[axis], float(value), int(wait)),
                f"MoveAbs {axis}",
            )

    def move_rel(self, x=None, y=None, z=None, wait=True, force=False):
        """Move by a relative distance (um). Axes left as None do not move."""
        cur = dict(zip("xyz", self.position()))
        steps = {k: v for k, v in (("x", x), ("y", y), ("z", z)) if v}
        self._validate({k: cur[k] + v for k, v in steps.items()}, cur, force)
        for axis, value in steps.items():
            self._check(
                self._dll.LSX_MoveRelSingleAxis(self._id, AXES[axis], float(value), int(wait)),
                f"MoveRel {axis}",
            )

    # -- software Z limits ---------------------------------------------------

    def _load_z_limits(self):
        try:
            with open(self._z_limit_file) as f:
                d = json.load(f)
            self.z_min, self.z_max = d.get("z_min"), d.get("z_max")
        except (OSError, ValueError):
            pass

    def set_z_limits(self, z_min="keep", z_max="keep"):
        """Set or clear (None) the software Z limits and save them."""
        if z_min != "keep":
            self.z_min = z_min
        if z_max != "keep":
            self.z_max = z_max
        os.makedirs(os.path.dirname(self._z_limit_file), exist_ok=True)
        with open(self._z_limit_file, "w") as f:
            json.dump({"z_min": self.z_min, "z_max": self.z_max}, f)

    def _validate(self, targets, current, force):
        for axis, target in targets.items():
            lo, hi = self.limits(axis)
            if axis == "z":
                if self.z_min is not None:
                    lo = max(lo, self.z_min)
                if self.z_max is not None:
                    hi = min(hi, self.z_max)
            if not lo <= target <= hi:
                raise ValueError(f"{axis} target {target:.1f} um is outside limits {lo:.1f}..{hi:.1f}")
            if axis == "z" and not force and abs(target - current["z"]) > MAX_Z_STEP_UM:
                raise ValueError(
                    f"Z move of {target - current['z']:+.1f} um exceeds the {MAX_Z_STEP_UM} um "
                    "safety step; pass force=True if this is intended"
                )

    def stop(self):
        """Stop all axes immediately."""
        self._check(self._dll.LSX_StopAxes(self._id), "StopAxes")

    # -- teardown ----------------------------------------------------------

    def close(self):
        if self._connected:
            self._dll.LSX_Disconnect(self._id)
            self._connected = False
        if self._id.value:
            self._dll.LSX_FreeLSID(self._id)
            self._id = C.c_int(0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    with TangoStage() as stage:
        print(stage.version())
        print("position (um): x=%.1f y=%.1f z=%.1f" % stage.position())
        for ax in "xyz":
            print(f"{ax} limits (um): %.1f .. %.1f" % stage.limits(ax))
