"""Motorised objective turret (nosepiece) of the Olympus BX53M through the BX3M-CBFM control box.

The control box is a serial device (COM4 on the glovebox PC): 19200 baud, 8 data bits, even parity, 2 stop bits.
Commands are text lines prefixed with "1" and ended with CR LF; refused or unknown commands answer "1x".

Moving the turret needs a remote login with "1L 2" (what Olympus Stream sends); "1L 1" also logs in but "1OB <n>" is
then refused.  "1L 2" itself is refused unless the box is logged out, and a box left logged in makes Stream report the
nosepiece as unavailable.  So this class stays logged out and logs in only for the duration of a move; if the box does
not answer the position query while logged out, it logs in and stays logged in until close().

Illumination (found 2026-11-13 by asking the box; the names are in Stream's drvB3M.dll).  Queries: "1EIL?" the
reflected-light (brightfield) lamp's level 0..255, "1EILSW?" its switch, "1MILS?" the MIX slider's LED ring (darkfield
from the side; a hexadecimal mask of its 4 segments, F = all on, 0 = off), "1MIL?" the ring's level.  Setting them needs
the remote login.  While the box is logged out it is in its hand-switch state: on this microscope the ring on and the
brightfield lamp dark, although "1EILSW?" still answers 1; logging out puts it back there.  So brightfield from the
program means: log in, "1MILS 0", "1EILSW 1" (sent again even if it reads 1), "1EIL <level>", and stay logged in
for as long as the light is needed (set_illumination does this; close() logs out).

It has the interface the scan program expects of a nosepiece reader (flakeid.scope.nosepiece.Nosepiece: unit, port,
position(), close()) plus set_position(n).

    from flakeid.scope.hw.cbfm import CBFMNosepiece
    with CBFMNosepiece("COM4") as turret:
        print(turret.position())     # 1..6, or None
        turret.set_position(2)
"""
from __future__ import annotations

import threading

POSITIONS = 6
QUERY_TIMEOUT_S = 1.0
MOVE_TIMEOUT_S = 8.0


class NosepieceError(RuntimeError):
    pass


class CBFMNosepiece:
    def __init__(self, port="COM4", baud=19200):
        import serial
        self.ser = serial.Serial(port, baud, bytesize=8, parity="E", stopbits=2, timeout=QUERY_TIMEOUT_S, write_timeout=1.0)
        self.lock = threading.RLock()
        self.port = port
        self.logged_in = False
        self.hold_login = False            # the box only answers when logged in: stay logged in until close()
        try:
            self._ask("L 0")               # clear a login left behind by a crashed session
            self.unit = self._ask("U?").split(" ", 1)[-1]
            if self.position() is None:    # no answer while logged out: try logged in
                self._login()
                self.hold_login = True
        except Exception:
            self.ser.close()
            raise

    def _ask(self, command, timeout=QUERY_TIMEOUT_S):
        with self.lock:
            self.ser.timeout = timeout
            self.ser.reset_input_buffer()
            self.ser.write(b"1" + command.encode("ascii") + b"\r\n")
            reply = self.ser.read_until(b"\r\n").decode("ascii", "replace").strip()
            if not reply:
                raise NosepieceError(f"no reply from the control box on {self.port} to {command!r} (is Olympus Stream still open?)")
            return reply[1:] if reply.startswith("1") else reply

    def _login(self):
        if not self.logged_in:
            if "+" not in self._ask("L 2"):
                raise NosepieceError("the control box refused the remote login (1L 2)")
            self.logged_in = True

    def _logout(self):
        if self.logged_in:
            self.logged_in = False
            self._ask("L 0")

    def position(self):
        """Turret position 1..6, or None (between positions, refused, or no answer)."""
        try:
            return int(self._ask("OB?").split()[1])
        except (IndexError, ValueError, NosepieceError):
            return None

    def set_position(self, n):
        """Rotate the turret to position n (1..6); returns when it has arrived.  The stage is NOT lowered first."""
        n = int(n)
        if not 1 <= n <= POSITIONS:
            raise ValueError(f"objective position must be 1..{POSITIONS}")
        with self.lock:
            self._login()
            try:
                reply = self._ask(f"OB {n}", timeout=MOVE_TIMEOUT_S)
            finally:
                if not self.hold_login:
                    self._logout()
            if "+" not in reply:
                raise NosepieceError(f"the control box refused OB {n}: {reply!r}")

    # ---- illumination
    def illumination(self):
        """What the box says: {"EIL": level, "EILSW": 0 / 1, "MILS": mask, "MIL": level} (a value None: not answered)."""
        out = {}
        for q in ("EIL", "EILSW", "MILS", "MIL"):
            try:
                out[q] = self._ask(q + "?").split()[1]
            except (IndexError, NosepieceError):
                out[q] = None
        return out

    def _set(self, command):
        reply = self._ask(command, timeout=MOVE_TIMEOUT_S)
        if "+" not in reply:
            raise NosepieceError(f"the control box refused {command!r}: {reply!r}")

    def set_illumination(self, mode, level=None):
        """mode "brightfield": the reflected-light lamp on (level 0..255; None: as it is), the MIX ring off;
        "darkfield": the MIX ring on (all segments; level: its level), the lamp off; "mix": both on.  The box stays
        logged in from here on (logging out puts the hand-switch state back): close() when the light is not needed."""
        mode = str(mode).lower()
        if mode not in ("brightfield", "darkfield", "mix"):
            raise ValueError(f"illumination {mode!r}: brightfield, darkfield or mix")
        with self.lock:
            self._login()
            self.hold_login = True
            self._set("MILS 0" if mode == "brightfield" else "MILS F")
            if mode == "darkfield":
                self._set("EILSW 0")
                if level is not None:
                    self._set(f"MIL {int(level)}")
            else:
                self._set("EILSW 0")              # (off and on: after a logout the lamp is dark although it reads on)
                self._set("EILSW 1")
                if level is not None:
                    self._set(f"EIL {max(0, min(255, int(level)))}")

    def close(self):
        if not self.ser.is_open:
            return
        try:
            self.logged_in = True          # (log out whatever the state: a box left logged in blocks Stream)
            self._logout()
        except Exception:
            pass
        finally:
            self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    import sys
    with CBFMNosepiece(sys.argv[1] if len(sys.argv) > 1 else "COM4") as n:
        print("units:", n.unit)
        print("objective position:", n.position(), "(queried while logged", "in)" if n.hold_login else "out)")
