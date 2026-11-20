# The glovebox microscope (Olympus BX53M)

How the scan program runs on the glovebox PC, which has no Micro-Manager. The scan program itself is described in
[scan_guide.md](scan_guide.md); this page covers what is different here and what is still to do.

## Hardware

| what | device | how it is driven |
|---|---|---|
| stage and focus | Märzhäuser Tango PCIe (COM3, 57600 baud); travel 75.5 × 50.5 mm; the controller's X / Y units are half the real distance (`stage_xy_scale` 2, see below); Z in µm; no encoders | `Tango_DLL.dll` (ships with Olympus Stream), [flakeid/scope/hw/tango.py](../flakeid/scope/hw/tango.py) |
| camera | Olympus DP22, 1920 × 1440, 8-bit colour, about 26 frames per second | `CamDriver64.dll` (ships with Olympus Stream; undocumented), [flakeid/scope/hw/dp22.py](../flakeid/scope/hw/dp22.py) |
| nosepiece | motorised, BX3M-CBFM control box (COM4); positions 1–5: 5x, 10x, 20x, 50x, 100x; 6 empty (`objective_positions`) | serial, [flakeid/scope/hw/cbfm.py](../flakeid/scope/hw/cbfm.py) |
| PC | Windows 10, Python 3.9 (`py -3.9`) | `pip install -r requirements.txt pyserial sv-ttk` |

The scan program talks to a microscope through one "core" object with Micro-Manager's method names.
[flakeid/scope/hw/bx53m.py](../flakeid/scope/hw/bx53m.py) is that object for this microscope, built on the stage and
camera drivers above. The setting `hardware: "bx53m"` (or `--hw bx53m`) selects it; it also changes the defaults
that only fit the BX51 (`HARDWARE_DEFAULTS` in [settings.py](../flakeid/scope/settings.py)).

## Running

Close Olympus Stream first: it holds the stage's port and the camera. From the program folder:

| command | what |
|---|---|
| `tools\bx53m\run_scanner.bat` | **the glovebox flake scanner**: thickness / size / cleanliness boxes, scan, sorted results ([glovebox_scanner.md](glovebox_scanner.md)) |
| `tools\bx53m\run_flakeid.bat` | the original scan program (start screen) |
| `tools\bx53m\run_flakeid.bat check` | opens every device as the scan program does and reports; **nothing moves** |
| `tools\bx53m\run_flakeid.bat calibrate --write` | measures the pixel → stage matrix `M`; **moves the stage** by 40 µm in X and Y and back |
| `tools\bx53m\run_flakeid.bat faketest` | the hardware layer on stand-in devices (no hardware needed) |

The settings are `scan_settings.json` in the program folder (not in git: it holds this microscope's calibration).

## Before the first scan

1. **Focus calibration** (`run_flakeid.bat zcal`), with the manual focus at its limit, where it must stay: the manual
   and the motorised focus do not know of each other, and the Tango has no Z limits of its own (±200 mm). Focus each
   objective with the window's Z buttons, save it, say which way Z + moves the stage, *Write the limit*. From then on
   the motorised focus goes no further towards the objective than the focus of the objective in place plus a margin
   (5x, 10x: 300 µm; 20x: 200 µm; 50x, 100x: 100 µm; `MARGIN_UM` in [zcal.py](../flakeid/scope/hw/zcal.py)), read
   from the nosepiece on every move; when the objective is not known, the tightest of them. The margin is the room
   for chips of different heights and for the autofocus search; it must stay well below the objectives' working
   distances. Calibrate again when the manual focus was moved or the samples sit much higher. (Calibrated
   2026-11-13: Z + is towards the objective; focus 5x 347, 10x 327, 20x 313, 50x 312, 100x 313.) A job also keeps
   every focus move within `max_dz_um` (150 µm) of where it started.
2. **Measure `M`** with the 10x objective on something with detail: `run_flakeid.bat calibrate --write`. The default
   `M` has Olympus Stream's pixel size (0.5857 µm per pixel at 10x) but a guessed orientation; with a wrong `M` the
   tiles do not join and the chip edges are not found.
3. **First scan: 10x scan only**, with *Analyse the flakes at 10x* unticked (the stitched picture and its viewer).

## What is different from the BX51 setup

- **White balance and exposure.** The camera's white balance gains are set to 2.07 / 1.0 / 1.0 (`wb_gains`) every
  time the camera is opened: it forgets them, and takes them only in its white balance mode 2. The exposure follows
  what Olympus Stream's auto exposure gives, 25 ms at 10x on empty 285 nm substrate: `exposure_ms` 25, and auto
  exposure brings the picture's median green to 124 of 255 (`target_level` 0.486), the level measured there. Blue is
  then just below clipping (247), so flakes brighter than the substrate saturate in blue; the saturation limits of
  the exposure check are widened for it (`exposure_sat_ok`, `exposure_sat_max`).
- **The nosepiece is motorised and has no hand switch**: the program turns it whenever a job needs another objective (`nosepiece_auto`). The stage is not lowered first; the objectives are parfocal within about 35 µm (focus calibration).
- **Stage units.** The controller has a spindle pitch of 1 mm configured and reports a travel of 37.75 × 25.25 mm,
  but the stage moves twice as far as it reports (40 of its µm moved the picture by about 82 µm against Stream's pixel
  calibration, 2026-11-13). `stage_xy_scale` (2.0) converts: every position in the program, its files and the viewer
  is in real µm, i.e. twice the number Olympus Stream or the controller shows. The controller's own configuration is
  not changed.
- **Camera lag.** The DP22's live picture shows a stage move 0.15–0.2 s after the controller reports it done; frames
  for the scan wait 0.25 s after a move (`camera_latency_s` in `bx53m.py`).
- **The light is set by the program** (`illumination`: `"brightfield"`, level `illumination_level` 0–255). With
  Olympus Stream closed the control box is in its hand-switch state: the MIX slider's LED ring on (a darkfield picture)
  and the brightfield lamp dark. When a job or the live view opens the microscope the program logs in to the box,
  switches the ring off and the lamp on, and stays logged in; at the end it logs out and the box goes back to the
  hand-switch state. `"darkfield"` and `"mix"` are the other choices, `""` leaves the light alone. The commands are in
  [cbfm.py](../flakeid/scope/hw/cbfm.py).
- **Gentle stage moves.** While the program is connected the X / Y acceleration is 0.1 m/s² (`stage_xy_accel`; the
  controller's 0.6 is put back when it disconnects). A chip lying loose on the stage slid 25 µm in 24 moves at 0.6
  and stayed within 1 µm at 0.2 and below; chips need no tape.
- **Stage moves.** The Tango drops a move sent while it is still moving, so X, Y and Z moves are sent one after the
  other (X, then Y). Tiles are therefore a little slower than they could be.
- **No IR camera**: `domain_imaging` is `"no"`.
- **Start-screen "Update..."** is disabled (`update_repo` empty): it would replace this code with the BX51 version.

## The DP22's raw mode (found 2026-11-13, not used by the program yet)

The camera library can deliver the sensor's own data instead of the finished picture (tested through `CamDriver64.dll`,
signatures as the other calls: `f(handle, 1, value)`):

- `CAM_LV_SetCCDImageTypeEx(h, 1, 1)` (0 = the colour picture): the live buffer then holds one 16-bit value per pixel
  (1920 × 1440, little-endian, 12 bits used, 0..4095), a colour mosaic in 2 × 2 cells, still 26 frames per second.
  Type 2 is refused (error 72). 16 bits per channel of the colour picture (`CAM_LV_SetBytesAChannelEx`) is not
  supported (error 10).
- `CAM_IP_SetContrastModeEx(h, 1, mode)`: the tone curve, also applied to the raw data. Mode 1 is what the camera is
  in when opened (compressive: counts rise 1.3-1.8x per doubling of the exposure); **mode 3 is linear** (1.94-1.98x
  per doubling, a straight line within ~1 % with a black level of about 30-40 counts); 2 and 4 are other curves; 0 is
  refused (error 190).
- The white balance gains multiply the raw channels (red cell 748 -> 1396 counts for 1.0 -> 2.07).
- On 285 nm SiO2 at 10x, 12 ms, gains 1 / 1 / 1, mode 3: cells (0,0) 347, (0,1) 1109, (1,0) 841, (1,1) 1851 counts: no
  channel near clipping. Which cell is which colour is not settled: (0,0) follows the red gain, (1,1) is the
  brightest (blue on this substrate), and the two others differ by 1.3x, which two green sites should not.
- In the colour picture the green channel is far below red and blue on this substrate and jumps up once blue clips:
  the camera's colour matrix subtracts blue from green. The raw mode has no such matrix.

## Thickness (colour calibration)

The scan records the camera's raw mosaic (linear, 12 bit), so the thin-film colour model of the original program applies.
What was set up for this camera (2026-11-16):

- **Flat fields**: `flats/flat_10x_dp22_raw.npz`, `flats/flat_100x_dp22_raw_285nm.npz`
  (`py -3.9 -m flakeid.scope.hw.flatfield`, on an empty chip).
- **Colour profile** `flakeid/profiles/bx53m_dp22_raw_led.json` (setting `profile_raw`): the original unsupervised fit
  (`flakeid.calibrate`) with a white LED lamp instead of the halogen, NA 0.25, fitted to the hBN colours of one chip with
  `tools/bx53m/fit_colour_profile.py`. On that chip every flat region fits (316 terraces, residual median 0.05, none
  flagged poor_colour_match), and a second fit with other channel ranges gives the same thicknesses within 5 nm.
- **Oxide per chip** from the substrate's colour: the profile carries `substrate_anchors` (the fitted chip: 295 nm at
  R/G 0.3055, B/G 2.652), used by `thickness.oxide_from_substrate` instead of the XC10's anchors. The empty chip of the
  same nominal 285 nm reads 285.5 nm. If the light changes (lamp level is fine; a filter or the aperture is not), the
  analysis stops with "the substrate colour fits no oxide".

**Not checked against AFM.** The fit is self-consistent, which fixes the shape of the colour curve, but only flakes of
known thickness photographed on this microscope show that the numbers are right. Known limits: the red channel is dim on
this oxide (6 % of full scale, noisy) and blue clips at +57 % contrast; graphite still uses the XC10's profile.

## Not done yet

- The scan-time substrate colour check (`substrate_colour_check`, off) has no references for this camera.
- A colour model for 100x (thick100 and the 100x keep score are the XC10's); the 100x sorting uses `flakeid.real100`.
- The pause window, device photos, objective offsets and the remeasure job have not been tried on this hardware.
