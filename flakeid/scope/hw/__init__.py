"""Microscopes driven without Micro-Manager, through their vendors' libraries.

bx53m.py   the glovebox Olympus BX53M as a Micro-Manager-like core for the scan program (setting hardware = "bx53m")
tango.py   Marzhauser Tango stage controller (X, Y, Z focus) through Tango_DLL.dll
dp22.py    Olympus DP22 camera through CamDriver64.dll (both libraries ship with Olympus Stream)
cbfm.py    motorised nosepiece through the BX3M-CBFM control box (serial)
check.py   python -m flakeid.scope.hw.check: every device through the core, without moving anything
"""
