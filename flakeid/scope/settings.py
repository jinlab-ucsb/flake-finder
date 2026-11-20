"""Scan settings: the settings block of tools/mm_chip_scan.bsh, with the same names (lower case) and defaults.

Override any of them with a JSON file (python -m flakeid.scope --settings my_settings.json): {"exposure_ms": 3.0, ...}.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Settings:
    # which microscope: "mm" = through Micro-Manager (--mm, --cfg: the BX51 with the XC10 and the Oasis stage);
    # "bx53m" = the glovebox BX53M through the vendors' libraries (flakeid/scope/hw: Tango stage, DP22 camera, BX3M-CBFM
    # nosepiece; no Micro-Manager).  A hardware other than "mm" changes other defaults too: HARDWARE_DEFAULTS below
    hardware: str = "mm"
    tango_port: str = "COM3"                # bx53m: the Tango stage controller's serial port ...
    tango_baud: int = 57600
    tango_dll: str = ""                     # ... its library; "" = the copy in Olympus Stream (hw/tango.py DLL_PATH)
    stage_xy_accel: float = 0.0             # bx53m: the Tango's X / Y acceleration while the program is connected, m/s2
                                            # (0 = the controller's own, 0.6 on the glovebox stage); a loose chip slides
                                            # at each start and stop
    stage_z_vel: float = 0.0                # bx53m: the focus axis's velocity (motor rev/s) and acceleration (m/s2) while the
    stage_z_accel: float = 0.0              # program is connected; 0 = the controller's own (5 and 0.1 on the glovebox stage)
    stage_xy_scale: float = 1.0             # bx53m: real um per unit of the Tango's X / Y read-out (2.0 on the glovebox
                                            # stage: its controller has half the real spindle pitch configured)
    profile_raw: str = ""                   # the analysis's camera profile for raw tiles (flakeid/profiles/<name>.json or a
                                            # path); "" = the XC10's (bx51_xc10_mm_raw16 ...)
    camera_raw: bool = False                # bx53m: the scan and the analysis work with the DP22's raw, linear 12-bit
                                            # frames (as the XC10's RAW16); wb_gains then apply to the colour pictures only
    save_colour: bool = False               # ... and beside every saved raw picture the camera's own colour picture, as
                                            # Olympus Stream shows it: colour.jpg in each tile's folder, <name>_colour.tif
                                            # beside each 100x picture
    colour_target: float = 0.486            # ... its exposure: the one that brings its median green to this (25 ms at
                                            # 10x on empty 285 nm substrate at the lamp setting of 2026-11-13)
    wb_gains: tuple = ()                    # bx53m: the DP22's white balance gains (R, G, B), set every time the camera
                                            # is opened (it forgets them); () = as the camera is
    stream_dir: str = ""                  # bx53m: Olympus Stream's folder (the DP22's library); "" = hw/dp22.py STREAM_DIR
    max_z_move_um: float = 500.0            # bx53m: a single focus move larger than this is refused (the Tango has no Z
                                            # limits configured; the software Z limits of hw/tango.py apply as well)
    # camera
    set_camera: bool = True
    exposure_ms: float = 3.0                # 10x; hBN on 285 nm: substrate green ~25 % of full scale
    auto_exposure: bool = False
    target_level: float = 0.25
    exposure_sat_ok: float = 0.01           # auto exposure is content with up to this fraction of saturated places ...
    exposure_sat_max: float = 0.05          # ... and the exposure check fails above this one ("overexposed")
    sat_channels: str = "rgb"               # the channels whose saturation counts in the exposure check ("rg": a camera
                                            # whose blue channel is at its top on bare substrate at the wanted exposure)
    exposure_level_max: float = 0.85        # ... or with the frame level above this
    level_channel: str = "green"            # the frame level (exposure check, auto exposure): the median of the green
                                            # channel, or "max": of each place's brightest channel (a camera whose
                                            # picture of the substrate is far from grey: its green would be set to
                                            # target_level with another channel saturated)
    exposure_range: tuple = (0.05, 500.0)
    camera_props: tuple = (("Data format", "RAW16"), ("Device output data bit depth", "12"),
                           ("Image data bit depth", "12", "optional"), ("AEAG", "Off"), ("Auto white balance", "Off"),
                           ("Gain", "0"), ("Luminosity gamma", "1"), ("Chromaticity gamma", "1"), ("Sharpness", "0"),
                           ("Downsampling", "1x1"), ("Binning", "1"))
    verbose: bool = True
    arrive_tol_um: float = 10.0
    stage_timeout_s: float = 60.0         # Micro-Manager waits this long for a stage / focus move (its default 5 s is too short for a long move at 100x)
    selftest_um: float = 100.0
    mode: str = "multi"                     # multi | chip | targets | flat | capture | retarget
    ask_mode: bool = False                  # start screen: choose scan / flat field / capture (set when started without --mode)
    scan10_only: bool = False               # multi mode, the start screen's "10x scan only": the chips scanned at 10x, no 100x
                                            # imaging (the start screen is back when the pieces are scanned)
    scan10_analysis: bool = True            # ... the flakes analysed at 10x as usual; False: the stitched 10x image and its
                                            # viewer (index.html) only (flakeid.live --image-only)
    remeasure_from: str = ""                # the start screen's "Remeasure flake coordinates": the old run's results folder;
                                            # a 10x scan only (image only), then remeasure.py moves the old run's coordinates
    remeasure_old: str = ""                 # (2026-10-30) the start screen's "Remeasure flake coordinates" now: the old run's
                                            # results folder; two spots centred by hand, then 100x passes until two agree
                                            # (Scanner.run_remeasure; no new scan)
    remeasure_flakes: str = "all"           # ... the flakes: all (imaged at 100x) | kept (the designer's kept folder) | custom
                                            # | random (remeasure_random_n of them: the first and last imaged, the rest at random)
    remeasure_list: str = ""                # ... custom: a flake list (JSON: the designer's "Export flakes (JSON)")
    remeasure_random_n: int = 10            # ... random: this many flakes (2026-11-02: "if they put 10, do first, last and 8 random")
    remeasure_agree_um: float = 5.0         # ... two passes agree: at least remeasure_agree_frac of the flakes within this of
                                            # the pass before (their common shift taken off; 2026-11-02: was every flake within 3 um)
    remeasure_agree_frac: float = 0.5       # ... the share of the flakes (found in either pass) that must agree
    remeasure_max_passes: int = 8           # ... 100x passes at most: then failed, with a report (2026-10-30e: was 6, then asked)
    coord_check: bool = True                # after the 100x imaging: every 100x picture again (its position and focus) until
                                            # two passes agree; the positions go with the results (Scanner.coord_check)
    coord_check_agree_um: float = 5.0       # ... two passes agree: more than coord_check_agree_frac of the flakes within
                                            # this of the pass before (common shift off; 2026-11-01: was every flake
                                            # within 3 um); a flake this far from where its picture was taken: flagged
    coord_check_agree_frac: float = 0.5     # ... the share of the flakes (found in either pass) that must agree
    coord_check_refocus_every: int = 60     # ... the 100x focus drifts: an autofocus on the first flake of every pass and
                                            # then every this many flakes; its change from the focus the picture was
                                            # taken at is added to the following flakes' focus (0: off; 2026-11-01)
    coord_check_refocus_max_um: float = 10.0  # ... an autofocus further than this from the prediction is not used (a
                                            # wrong peak)
    coord_check_max_passes: int = 8         # ... at most; then flagged, a report, the positions not changed
    remeasure_focus: str = "autofocus"      # ... (2026-11-02: always autofocus, the focus drifts; "previous", each flake at its
                                            # old 100x focus without autofocus, was removed and is ignored)
    remeasure_refocus_um: float = 5.0       # ... no autofocus: the old focus; plus the chips' change in height when the two
                                            # spots (focused by hand at 10x) show more than this
    remeasure_min_peak100: float = 0.7      # ... least correlation of a flake's 100x frame with its old 100x picture
                                            # (same flake: 0.96-0.99 on the simulator; a chance match elsewhere 0.5)
    settings_file: str = ""                 # the JSON file these settings came from (a new flat field can be written into it)
    focus_mode: str = "auto"                # auto | manual
    out_dir: str = "C:/scans"
    name: str = "chip"
    run_scan: bool = False                  # chip mode: acquire the tiles after the plan
    show_plan: bool = False                 # show each piece's scan-plan picture in a window (it is written to the results either way)
    edge_points: int = 5
    edge_spread: float = 0.40
    arc_r_range: tuple = (15000.0, 200000.0)
    overlap: float = 0.15
    margin_um: float = 100.0
    max_travel_um: float = 30000.0
    # the stage's travel, [x min, x max, y min, y max] in stage um (null = not known); limits are also learned when the
    # stage stops short at one (kept in <local_root>/stage_limits.json): a chip partly beyond them is scanned up to them
    stage_limits_um: object = None
    stage_limit_learn: bool = True
    stage_limit_margin_um: float = 20.0
    step_frac: float = 0.80
    dark: float = 128.0                     # black level at 12 bit (RAW16), scaled to the bit depth
    M: tuple = ((-1.0905, -0.0067), (0.0066, -1.0917))   # 10x pixel -> stage um (stage that centres pixel p = stage + M (p - c))
    sample: int = 16
    chroma_tol: float = 0.10
    start_min_substrate: float = 0.20
    # the substrate's colour at the start of each piece (after the start autofocus) against the expected one for its
    # oxide (Avram, 2026-10-29: a run with a redder, dimmer light found no flakes): R/G and B/G within
    # substrate_colour_tol, the brightness (at the exposure in use) within substrate_brightness_tol; else a window
    # (lamp, filters, oxide) before the scan.  "auto": on the microscope, not on the simulator
    substrate_colour_check: str = "auto"    # auto | on | off
    substrate_colour_tol: float = 0.12
    substrate_brightness_tol: float = 0.30
    substrate_colour: object = None         # None, or (R/G, B/G) of bare SiO2
    bright_min: float = 0.45
    bright_max: float = 1.60
    bin_um: float = 20.0
    gap_um: float = 80.0
    gap_fraction: float = 0.03
    surround_dark: bool = True
    dark_level: float = 0.6
    edge_any_um: float = 1500.0
    # chips close together (2026-10-20: a lab member's chips were not found): the walk to an edge stops before the next
    # chip (its outline once scanned, else its marked position); before that a narrower dark gap is taken as the edge,
    # and with none (touching chips) the user is asked to put the crosshair on the edge (edge_ask; off: the piece fails)
    narrow_gap_um: float = 20.0
    edge_ask: bool = True
    focus_points: int = 4                   # manual focus
    focus_points_auto: int = 9              # autofocus: 4 or 9
    # small pieces (both sides below small_piece_mm, from the first edge search): fewer edge points per side and
    # autofocus points (a tilted plane is enough over ~1 cm; the 9-point quadratic surface is for inch-size pieces)
    small_piece_mm: float = 15.0
    small_edge_points: int = 3              # points per side, the first one included (3 = 2 more)
    small_focus_points: int = 4
    focus_inset: float = 0.15
    # chips mounted with tape over their edges (2026-10-20): a focus point must be on bare substrate (else it moves towards
    # the chip's centre, focus_moves times); one that disagrees with the plane through the others by more than
    # focus_outlier_um is refocused further in, and left out of the focus surface if it still disagrees
    focus_min_substrate: float = 0.3
    focus_moves: int = 3
    focus_outlier_um: float = 8.0
    max_dz_um: float = 150.0
    # the focus drive read back short of where it was sent by more than this: at a limit of its travel (the Oasis
    # controller's lower / upper limit; BN_15_auto 20261025_run_4 stopped at -181.9 um, 33 of 58 100x pictures out of
    # focus): logged, and one message (Slack / email) per direction and job
    z_reach_tol_um: float = 1.0
    focus_limit_stop: bool = True          # an autofocus whose best focus is at the focus drive's lower limit stops the
                                           # scan with a message (Scanner.focus_limit_check; passing the limit on the way: fine)
    z_stepsize_um: float = 0.0250 / 12.0
    z_scale: float = 12.0
    focus_surface: str = "quadratic"
    max_bow_um: float = 15.0
    max_tilt: float = 10.0                  # um/mm
    max_focus_rms: float = 5.0
    af10: tuple = (30.0, 5.0, 6.0, 1.0, 0.8)
    af100_first: tuple = (15.0, 1.0, 1.0, 0.25, 0.9)
    # 100x after the first flakes: the prediction was within 1.3 um of the focus for 95 % of 148 flakes (microscope PC,
    # 2026-10-03; max 3.5 um, found by the sweep's extension when the peak is at an end): +-2.4 um coarse, +-0.5 fine
    af100: tuple = (2.4, 0.8, 0.5, 0.25, 0.9)
    # ... and that sweep alone (its peak and two neighbours, a parabola through log sharpness) instead of the fine sweep
    # after it, when the peak is clear (not at an end of the sweep, curving down); otherwise the fine sweep as before.
    # 7 frames instead of 12-17 (20261014_run_1: 445 autofocus calls x 4.2 s)
    af100_fit: bool = True
    # the focus drive slipped by 40-60 um during 20261014_run_1 (outside the +-15 um af100_first search): every flake
    # after it failed, and the first flake of each new piece waited focus_dialog_timeout_s for an answer.  When the
    # usual searches fail on af_long_after flakes in a row (or before a focus question would be asked), one long coarse
    # search (+-80 um, 3 um steps, no extension; then the fine sweep) is made, once per run of failures; a focus found far
    # from the prediction is confirmed as a focus jump (image_flake)
    af_long_after: int = 2                  # 0 = never
    af_long: tuple = (80.0, 3.0, 1.5, 0.25, 0.9)
    retake_failed: bool = True              # flakes not focused are imaged again at the end of their piece (when a flake
                                            # was focused after them: the prediction knows more), replacing the first image
    # the retakes search harder (2026-10-27): after the usual searches, af_retake (+-40 um coarse, 2 um steps), moved on
    # up to af_retake_extend times when the peak is at an end (+-120 um); the 100x window's "Manual focus (retakes)"
    # button: the current retake (if not focused) or the next one focused by hand (live view, fine knob)
    af_retake: tuple = (40.0, 2.0, 1.0, 0.25, 0.9)
    af_retake_extend: int = 2
    af_max_dev_um: tuple = (5.0, 3.0)
    af_dev_per_mm: float = 0.5
    af_max_dev_cap: float = 8.0
    af_agree_um: float = 1.5
    # focus drift over a long run (2026-11-17): each 100x start Z is the 10x focus + the 100x - 10x offset of the flakes
    # focused before.  The offset drifts with time (glovebox BX53M, 20261117_1: +1.9 um/h during the scan, -1.2 um/h over
    # the next 4 h, then +4 um/h), so the offsets are brought forward by the drift rate of the last
    # focus_drift_window_min (Theil-Sen, clipped to +-10 um/h) and weighted down by age (half the weight every
    # focus_drift_half_life_min).  A replay of that scan stretched to 4-12 h with such drifts: 95 % of start Z within
    # 0.7-0.8 um (only by distance: 3-9 um, outside the +-2.4 um sweep).  0 = by distance only, as before
    focus_drift_half_life_min: float = 15.0
    focus_drift_window_min: float = 60.0
    # Pause / Resume (the main window's Pause button; 2026-10-27): at Resume the focus where the scan paused is measured
    # again (autofocus) and a change beyond pause_refocus_tol_um (10x, 100x) shifts every later Z by it (the coarse knob
    # is not seen by the drive); above pause_refocus_warn_um a message too.  pause_refocus False: no focus check
    pause_refocus: bool = True
    pause_refocus_tol_um: tuple = (1.5, 0.5)
    pause_refocus_warn_um: float = 10.0
    af_extend: int = 2
    af_metric: str = "mean"                 # sharpness of a colour frame: "mean" = the mean gradient (the XC10's raw frames);
                                            # "top" = the strongest 1 % of the gradients at a quarter of the resolution (a
                                            # noisy 8-bit picture: at 100x on the DP22 the focus stood 7-8x above the
                                            # background with it, 1.15x with the mean)
    # ... on a raw (mosaic) frame at 100x, the colour sites it is measured on: "green" (as before), "red", "blue".  The
    # glovebox 100x objective focuses red ~1 um higher than green/blue; Stream's view (sharp) was at red's best focus,
    # the green sites' best focus was visibly soft (2026-11-17).  The 10x always uses green (depth of field ~8 um)
    af100_channel: str = "green"
    af_min_ratio: float = 1.10
    af_backlash_um: float = 5.0
    spike_factor: float = 3.0
    # wait after every Z move before the next frame (the drive reports "done" when the motor stops; 20 ms let the
    # objective settle).  A 100x autofocus makes ~16 Z moves, so each 10 ms here is ~0.16 s per flake.  Not lowered:
    # the simulator has no settling to test it against; try 10 at the microscope and compare the focus repeatability
    z_settle_ms: float = 20.0
    overlap_z_move: bool = True            # tiles: the focus backlash approach is sent during the XY move
    tile_readback_every: int = 10          # tiles: the stage position read back (and checked against arrive_tol_um) on every
                                           # Nth tile, the last and any that raised; the others are recorded where they were
                                           # sent (two controller round trips saved per tile; 1 = every tile, as before)
    background_save: bool = True           # tiles: written by a thread while the stage moves on
    ui_theme: str = "dark"                 # windows: "dark" / "light" (Sun Valley, sv-ttk) or "native" (Windows' own look)
    af_retries: int = 2
    # targets / 100x
    targets_csv: str = ""
    chip_json: str = ""
    plane_min_spread_um: float = 1500.0
    target_exposure_ms: float = 27.0
    target_auto_exposure: bool = False
    target_box_margin_um: float = 2000.0
    # the 100x objective against the 10x one (flakeid/hwconst.py; the BX51's measured values): 100x pixels per 10x pixel,
    # and the stage move (um) that puts under the 100x objective what was centred at 10x
    ratio_100x: float = 9.54
    offset_100x_um: tuple = (24.0, -18.3)
    focus_offset_um: float = -42.0          # 100x focus Z minus 10x focus Z (real um); NaN = measured at the first flake
    hundred_shifts_um: object = ((10.0, 0.0), (0.0, 10.0))
    hundred_save_frames: bool = False      # also save every frame averaged into a flake's 100x image (<file>_f1.tif ...)
    pattern_frames: int = 0                # targets mode: then this many frames around the first target, for the optics
                                           # pattern (dust, specks: fixed to the camera, found by comparing positions)
    pattern_radius_um: float = 250.0       # ... at up to this distance from it
    hundred_frames: int = 4                 # frames averaged for each flake's main 100x image (pixel noise ~2 % -> ~1 %);
                                            # saved as rounded 16-bit; the shifted frames are single (the analysis weights them)
    # each 100x target checked against its 10x picture (<piece>_targets_crops) after focusing, and the stage moved onto
    # the flake when the shot is off (2026-10-14: 214 of 1574 100x frames had bare substrate at the centre)
    # mosaics of large terraces (flakeid/fieldmosaic.py): when a target's chosen terrace (its outline from the analysis:
    # terrace_poly_um in the target list) reaches beyond the main 100x picture, the fields it reaches into are imaged
    # too (<main>_m02.tif ..., <piece>_100x_mosaic.csv) and stitched by the export (<main>_mosaic.jpg in the viewer, key W).
    # Target lists without the column (before 2026-10-19): the single picture as before
    hundred_mosaic: bool = True             # (off: the single centred picture only, 100x and IR)
    hundred_mosaic_max_tiles: int = 8       # at most this many pictures per flake, the main one included (the nearest
                                            # to it kept when the terrace needs more); 30 before 2026-10-19 (Jordan: 5-10)
    hundred_mosaic_overlap: float = 0.15    # overlap between neighbouring fields (fraction of the field)
    hundred_mosaic_budget_min: float = 20.0  # extra 100x imaging time per run for mosaics; when it is used up the
                                            # remaining flakes get their single picture (logged)
    hundred_mosaic_frames: int = 0          # frames averaged per mosaic field (0 = hundred_frames); no shifted frames
    # each field's focus: predicted from the main picture's focus + the chip's tilt (the fields are within a few hundred
    # um of the focused main picture); a tuple (half range, step, roi) adds one short sweep per field
    # (Scanner.autofocus_fit), e.g. (1.2, 0.4, 0.9) (the default until 2026-10-19: ~7 frames a field)
    hundred_mosaic_af: tuple = ()
    hundred_mosaic_af_min_rise: float = 1.05  # ... its sharpest point this much above both ends of the sweep, else the
                                            # predicted Z is used (focus "predicted" in the mosaic index)
    hundred_mosaic_max_dev_um: float = 2.0  # a sweep result further than this from the prediction is not used
    domain_mosaic_budget_min: float = 10.0  # the same for the IR (stacking-domain) pictures, their own budget per run
                                            # (0 = no IR mosaics); same max tiles and overlap
    domain_mosaic_af: tuple = ()            # IR fields' focus: the main IR picture's (its depth of field is several um);
                                            # a tuple (half range, step, roi) adds a sweep per field, e.g. (4.0, 0.8, 0.9)
    # the IR camera's pixel -> stage matrix (um per px, as its frames are saved: after domain_flip), for its mosaic fields;
    # the Domain camera test prints it ("IR camera pixel -> stage") and saves it here.  null = the 100x camera's
    # directions at 0.358 IR px per 100x px (measured 2026-10-15; the ~6 deg between the cameras is left to the export's
    # overlap matching)
    domain_px_um_matrix: object = None
    xy_approach_um: float = 100.0           # 100x and domain targets are reached from -x / -y: first this far short of
                                            # them, then a short +x / +y move (same backlash every time; 0 = direct)
    xy_approach_clear_um: float = 20.0      # ... an axis on which the move to the target is already at least this far in +
                                            # needs no pre-move (it arrives from - anyway, its backlash taken up: the median
                                            # miss was 2 um after +y moves against 8 um after -y moves); 0 = always pre-move
    hundred_shift_approach_um: float = 20.0  # the shifted 100x frames: a shift move back in -x / -y goes this far short
                                            # first, so every shifted frame is also reached from -x / -y
    hundred_recentre: bool = True
    recentre_before_focus: bool = True      # re-centre at the predicted Z, then autofocus on the centred flake (2026-10-14:
                                            # ~14 % of 100x frames had bare substrate at the centre: no focus signal, a wide
                                            # retry); off, or when the focus is not known yet: after the autofocus as before
    recentre_min_peak: float = 0.5          # least correlation (100x frame against the 10x picture) to trust a match ...
    recentre_max_second: float = 0.85       # ... and the best match elsewhere at most this fraction of it
    recentre_min_um: float = 4.0            # smaller offsets are left alone
    recentre_max_um: float = 150.0          # larger ones are not trusted (a different place that looks alike)
    recentre_search: bool = True            # no confident match: look around (a ring of 6 frames) for the flake
    recentre_carry: int = 5                 # the next targets start corrected by the median of the last N corrections
                                            # (0 = off); slow drift of the stage or of the 10x mapping is absorbed
    lost_after: int = 3                     # this many 100x flakes in a row not recognised (re-centring: no confident match):
    #                                         the operator is alerted and centres and focuses the flake by hand (fine or coarse
    #                                         knob); the stage's shift and the focus are corrected from it (0: never)
    af_manual_fallback: str = "first"       # first | always | never
    focus_dialog_timeout_s: float = 120.0
    # target_mode "avram_new1" only: at the start of each piece's 100x imaging, int(its 10x focus points / this) of its
    # targets (spread over the piece, a short path) are focused by hand (0 = none).  They calibrate 10x -> 100x
    # (scan.offset_model: a plane of the 100x - 10x offset, or its median); every flake then starts at the 10x focus
    # map + that calibration (corrected by the flakes focused since), then autofocus.  Skip: that point; Stop, or no
    # answer within focus_dialog_timeout_s: no more points
    hundred_manual_focus_divisor: int = 3
    focus_dialog_repeat: bool = False       # after a focus question went unanswered, ask again on that piece (false: the
                                            # rest of the piece's failures are imaged at the predicted Z and retaken at the end)
    operator_sound: bool = True             # a sound (Windows) when the operator is needed: objective change, IR setup, a
                                            # focus question
    # messages to you (flakeid/scope/notify.py: python -m flakeid.scope.notify --test): Slack and / or email when the
    # operator is needed and when a job ends; the SMTP password from FLAKEID_SMTP_PASSWORD or notify_smtp_password_file
    ask_chip_notes: bool = True             # multi / chip scans: a window beside the scan for each chip's exfoliation details
                                            # (crystal batch, tape, ...), saved as <chip>_notes.json and in the run file
    notify_slack_webhook: str = ""          # a Slack incoming-webhook URL
    notify_email_to: str = ""               # address(es), comma-separated
    notify_email_from: str = ""
    notify_smtp_host: str = ""
    notify_smtp_port: int = 587             # 587 STARTTLS, 465 SSL, 25 plain
    notify_smtp_user: str = ""
    notify_smtp_password_file: str = ""
    notify_slack_token_file: str = ""       # Slack bot (app "FlakeID Scanner"): a file with its xoxb- token (or FLAKEID_SLACK_TOKEN)
    # program updates from GitHub (the start screen's "Update...", run_flakeid.bat update; flakeid/scope/update.py)
    update_repo: str = "afylab/Flake-Finder"
    update_branch: str = "main"
    update_token_file: str = "C:/FlakeID/github_token.txt"   # only for a private update_repo: a token that can read it (or FLAKEID_GITHUB_TOKEN)
    # the logs copied to a private GitHub repository by themselves (flakeid/scope/logupload.py): at the start, after
    # every job and after a background export; nothing without the token
    log_upload: bool = True
    log_upload_repo: str = "afylab/flakeid-logs"
    log_upload_token_file: str = "C:/FlakeID/github_logs_token.txt"   # Contents read and write on that repository only (or FLAKEID_LOGS_TOKEN)
    notify_to: str = ""                     # who runs the scan: Slack email, name or member ID (the start screen asks; remembered)
    notify_slack_channel: str = ""          # optional: a channel ("#microscope") that gets each run's thread too
    notify_on: tuple = ("operator", "scanned", "finished", "stopped", "failed")   # (+ "progress": each chip's plan)
    notify_threads: bool = False            # Slack: each run's messages in one thread (replies do not notify: off, every
                                            # message a new one, 2026-10-19)
    notify_min_job_min: float = 5.0         # end-of-job messages only for jobs at least this long (not a quick flat field)
    # multi
    piece_centers: object = None            # None: asked (live viewer); or [[x, y(, z)], ...]
    piece_materials: object = None          # [["graphite", 90], ["hbn", 285], ...] with piece_centers
    # what is on the stage (asked in the Chips window after Start scan, remembered in <local_root>/chips.json):
    #  "large"        larger pieces: the edge walk (find_outline), as before 2026-10-21
    #  "loose_1cm"    1 cm chips (diced by hand), each on its own: a nominal square checked by one probe per side
    #  "mounted_1cm"  the same, rigidly mounted on a glass slide (often taped): up to 2 blocks per scan, each with 3
    #                 reference corners marked at the start (one stage -> AFM mapping for all its chips)
    #  "polygon"      scan areas drawn by hand: no edge search; at the start each area's corners are marked one by one
    #                 (a polygon, as many points as wanted; several polygons), and every field that overlaps it is scanned
    chip_type: str = "large"
    chip_nominal_mm: float = 10.0           # 1 cm chips: the nominal side
    chip_size_tol: float = 0.30             # ... cut by hand: any side within +-30 % of it
    chip_centre_err_mm: float = 1.0         # ... how far the marked centre may be off the true one
    chip_focus_inset_mm: float = 1.5        # ... focus points this far in from the chip's edges (tape over the edges)
    chip_edge_any_um: float = 150.0         # ... this long without substrate across the whole field is the edge (tape, glue:
                                            # not dark, so the usual edge_any_um (1.5 mm) let the probe cross ~1 mm of tape)
    piece_mounts: object = None             # with piece_centers: the block of each piece, e.g. [1, 1, 2, 2] (mounted_1cm)
    mount_corners: object = None            # with piece_centers: {"1": [[x, y], [x, y], [x, y]], "2": [...]}
    piece_polygons: object = None           # chip_type "polygon": None = marked at the start (live viewer); or each area's
                                            # points [[[x, y(, z)], ...], ...] (piece_materials as with piece_centers)
    polygon_focus_mode: str = ""            # ... how its 10x focus points are focused: "auto" | "manual" (the polygon window's
                                            # choice); "" = focus_mode.  The 100x imaging keeps focus_mode
    polygon_focus_per_cm2: float = 4.0      # ... focus points per cm2 of a polygon (at least 4), spread uniformly over it
    polygon_focus_smoothing: float = 1.0    # ... its focus surface: a smooth one through all the points (thin-plate spline);
                                            # larger = smoother (towards a plane), 0 = through every point exactly
    run_dir: str = ""
    run_name: str = ""
    targets_wait_s: float = 3600.0
    objective_prompt: bool = True
    material: str = "hbn"
    oxide_nm: float = 285.0
    # stacking domains (graphite)
    domain_imaging: str = "ask"             # ask | yes | no
    domain_min_layers: int = 3
    domain_max_nm: float = 0.0
    domain_camera: str = "Raptor Owl Camera 640"
    # the Raptor Owl 640 through Micro-Manager's RaptorEPIX driver (tests 2026-10-09, tools/win7/ir_camera_test.py):
    #  - "Frame Average" > 1 is the driver snapping and averaging in software, and a timeout in it leaves the camera stuck
    #    "live": kept at 1 (the scan enforces it), the frames are averaged by the scan instead (domain_frames)
    #  - the exposure setting does not change the picture (not in the camera's own software either): its auto exposure
    #    stays on, domain_exposure_ms is sent but has no effect on this camera
    #  - High Gain Off: darker, but more signal-to-noise per frame than On (53 against 35 at the same exposure)
    domain_camera_props: tuple = (("Exposure: Auto", "On", "optional"), ("High Gain", "Off", "optional"), ("Frame Average", "1", "optional"))
    domain_exposure_ms: float = 28.0
    domain_frames: int = 16                 # frames averaged for each domain image (from the camera's stream; ~1.3 s at 25 Hz)
    domain_wake: bool = True                # the domain camera live for a moment, then stopped, before its settings (it
                                            # gave empty frames until it had been live once)
    domain_flip: str = "xy"                 # the Raptor's picture is mirrored in X and Y against the colour camera: its
                                            # frames are flipped back ("x", "y", "xy", "" = as the camera gives them)
    flat_domain: str = ""                   # IR flat field (.npz from Flat field, objective "IR (domain camera)"): the domain
                                            # images are divided by it in the export ("" = none)
    domain_xy_offset_um: tuple = (0.0, 0.0)
    domain_focus_offset_um: float = float("nan")
    domain_search: tuple = (30.0, 1.0, 1.5, 0.25, 0.9)
    domain_af_first: tuple = (15.0, 1.0, 1.5, 0.25, 0.9)
    domain_af: tuple = (4.0, 0.8, 1.5, 0.25, 0.9)
    domain_max_dev_um: float = 2.0
    domain_af_fit: tuple = (4.0, 0.8, 0.9)  # IR autofocus at a predicted focus: one sweep (half range, step, roi), focus from
                                            # a fit (Scanner.autofocus_fit; empty: the coarse + fine sweeps of domain_af)
    domain_af_min_rise: float = 1.03        # ... its sharpest point at least this much above both ends of the sweep
    domain_restore_camera: bool = True
    # flat
    flat_objective: str = "10x"
    flat_frames: int = 25
    flat_step_um: float = 0.0
    flat_min_substrate: float = 0.6
    flat_focus: str = "fieldstop"           # fieldstop: plane through the field-stop edge focused at 5 points; fixed; auto (each position)
    flat_dark_frames: int = 20
    flat_oxide_nm: float = 285.0            # 100x flat fields are per oxide: in the file name and the flat100 entry
    flats_dir: str = ""                     # where new flat fields are saved; "" = <program folder>/flats
    flat_measured_oxide_nm: float = 0.0     # Filmetrics value of the flat-field piece (0 = none): recorded as the oxide calibration
    flat_gui: bool = False
    # re-image the 100x targets of a piece scanned before (start screen): find the chip again from its flakes
    retarget_piece: str = ""                # piece name (its <piece>_targets.csv and _chip.json are in local_root)
    retarget_reloaded: bool = False         # taken off and put back: search widely (shift and rotation)
    retarget_only_failed: bool = False      # only the targets without a focused image last time (<piece>_retake.csv)
    retarget_manual_points: int = 3         # the user centres this many flakes at 100x (spread over the piece) and the
                                            # positions are converted from those; 0 = found automatically (edges, then
                                            # 10x spots against the first scan's mosaic; less reliable)
    retarget_update_only: bool = False      # no imaging: the re-images done so far into the viewer, designer run and the
                                            # results folder (start screen)
    retarget_spots: int = 7                 # 10x reference spots for the registration (spots that disagree are left out)
    retarget_find_edges: bool = True        # first find the piece's edges again (as the scan does): its shift and rotation
    retarget_min_peak: float = 0.2          # 10x spot against the first scan's mosaic: least correlation to use it
    retarget_check100: bool = True          # check (and correct) the 100x position on the first targets against the mosaic
    retarget_min_peak100: float = 0.3       # ... least correlation for that
    retarget_recentre: bool = True          # re-centre each target at 100x against the mosaic before imaging it
    edge_tol_um: float = 60.0
    lead: float = 1.5
    # analysis on this PC (multi mode): python -m flakeid.live follows the run folder and writes the target lists
    local_root: str = "C:/FlakeScans"       # multi mode: tiles, 100x images and the run's files go here (local disk)
    ask_results_folder: bool = True         # multi mode: ask where the results are copied at the end (copy_results_to)
    local_analysis: bool = True
    analysis_restarts: int = 2              # the analysis crashed (exit 1): started again this many times (it resumes)
    results_dir: str = ""                   # analysis results; "" = <local_root>/results
    analysis_workers: int = 2               # leave the processor room for the acquisition
    flat10: str = ""                        # 10x flat field .npz for the analysis
    flat100: tuple = ()                     # ("285=C:/FlakeID/flats/flat_100x_285nm.npz", "90=...")
    keep_tiles: bool = False
    copy_results_to: str = ""               # where the results (viewers, records) are copied at the end; "" = asked
    upload_each_chip: bool = True           # each piece copied there as soon as its 100x export ends (the designer adds
                                            # it while the next pieces are imaged); the rest at the end
    # a faint region surrounded by similar haze (glue) is kept as a flake if its own outline is this compact (solidity,
    # 0..1; 0 = off: left out as residue).  flakeid.pipeline.RESIDUE_RING_SOLIDITY
    residue_ring_solidity: float = 0.0
    faint_flakes: bool = False              # the analysis keeps faint flat films with a sharp edge that the faint-residue
                                            # rules (low_contrast, residue_like) would leave out (flakeid.pipeline
                                            # FAINT_FLAKES; the glovebox scanner turns it on)
    strict_uniformity_um: float = 0.0       # the analysis leaves out regions without a uniform disc of this diameter (um)
                                            # or with an irregular shape (tape glue; 0 = off: the irregular ones are
                                            # kept and scored, as the BX51's labels wanted)
    # 100x targets, the analysis's cuts (None / () = its defaults: score 0.75, best terrace 30 um wide, hBN 0-100 nm)
    target_min_score: object = None         # flakeid.live --min-score
    target_min_d_um: object = None          # flakeid.live --min-d
    hbn_range_nm: tuple = ()                # the hBN thickness window of the 10x analysis (min, max)
    real100: str = ""                       # after the 100x export every 100x picture judged real flake or glue from its raw
                                            # frame (flakeid/real100.py): "judge" = only the list (real100.csv); "sort" =
                                            # the glue pictures moved to <piece>_100x/rejected/ as well; "" = not judged
    # ... how picky (flakeid.real100.judge): the least probability of a real flake (lower keeps doubtful ones); the least
    # share of the flake that is flat and clean, 0..1 (speckled or dirty flakes: 0.2-0.4, clean ones 0.75 up; 0 = any);
    # the least area of the flake, um2 (0 = any).  The kept flakes' colour pictures go to <results>/<piece>/real_flakes/
    real100_cut: float = 0.5
    flake_min_flat: float = 0.0
    flake_min_area_um2: float = 0.0
    min_targets: int = 10                   # 100x: at least this many flakes per piece, the best-scoring ones even if they
                                            # score below the cut (e.g. graphite on 285 nm oxide); 0 = only those above it
    # 100x targets per piece in proportion to its area (found by the edge walk): targets_per_cm2 x area, at least
    # targets_min_piece, at most targets_max_piece; asked in the Messages popup after Start scan (remembered in
    # <local_root>/targets.json).  0 = the analysis's fixed --max-targets (80 per piece, before 2026-10-21).  25/cm2:
    # Jordan's ~4.5 cm2 pieces had 145-306 qualifying flakes and got 80 (2026-10-20).  100 since 2026-10-27 (Avram: 100
    # targets per cm2, every one imaged at 100x)
    targets_per_cm2: float = 100.0
    targets_min_piece: int = 10
    targets_max_piece: int = 0              # 0 = no upper limit (Avram, 2026-10-27: was 400; a 74 x 10 mm polygon at
                                            # 100 per cm2 wants ~740)
    # hBN 100x targets (the analysis on this PC, python -m flakeid.live --target-order / --target-rank; 2026-10-22):
    #   target_order "size": imaged largest best terrace first (groups of 10, a short stage path within each), so a run
    #     stopped or a piece skipped early still has the biggest flakes; "path": the shortest stage path (before)
    #   target_rank "score": the best-scoring flakes are targets (as before); "size": ranked by logit(score) +
    #     2 ln(best terrace d) instead: the trial, a run of one chip (C:\FlakeID\run_flakeid.bat scope --set target_rank=size;
    #     not in scan_settings.json, or every later run would be a trial too).
    #     Each hBN piece also gets <piece>_targets_alt.csv: the flakes the other ranking would add / drop (comparison)
    #   graphite and other materials are not affected
    # which flakes are 100x targets (the Chips window asks; flakeid.live --target-mode): "standard" = the learned score and
    # the quality flags (everything below); "avram_new1" = every flake whose clean area (its largest one-colour region x
    # the fraction of it matching that colour) is at least target_min_clean_um2, largest first
    target_mode: str = "standard"
    # the Chips window's "Scan profile" (2026-11-11): a named set of settings for the 100x part of a job (SCAN_PROFILES
    # below, plus any in scan_profiles); "current" = this file's settings as they are.  Remembered in <local_root>/chips.json
    scan_profile: str = "current"
    scan_profiles: object = None            # more profiles: {"name": {"label": ..., "hint": ..., "settings": {...}}}
    target_min_clean_um2: float = 400.0
    target_clean_thickness_nm: tuple = ()   # avram_new1: the clean areas' thickness (min, max) nm; () = the material's
                                            # window (hBN 0-100, graphite 0.3-25)
    target_order: str = "size"
    target_rank: str = "score"
    # hBN, target_order "size": of each piece's targets only this many with the largest best terrace are imaged (the
    # same targets as before, the smallest left out: 50 % of the 100x time, ~79 % of the keeps and all the flakes used
    # in designs in a replay of 21 pieces; flakeid.live HBN_MAX_TARGETS); 0 = all.  48 since 2026-10-25: the inclusive
    # thickness (thick_context etc. no longer exclude) put ~7 likely-thick flakes into a piece's 40 (multitest chip_1);
    # 8 more slots (~2-3 min) keep the thin ones in, the thick ones are told apart at 100x
    # 0 since 2026-10-27 (Avram: every target is imaged at 100x; the Chips window's targets per cm2 sets how many)
    hbn_max_targets: int = 0
    # hBN: of each piece's targets, hbn_imaged_per_cm2 x its area (cm2) with the largest best terrace are imaged (at least
    # targets_min_piece; a 74 x 10 mm polygon at 30: 222); 0 = every target (the default since 2026-10-27, when the
    # Chips window stopped asking for it).  Pieces of unknown area: hbn_max_targets
    hbn_imaged_per_cm2: float = 0.0
    # hBN: the thin-hBN search (the Chips window's "Thin hBN search", opt-in; flakeid.live --thin-hbn): the 10x picture is
    # also searched for thin hBN (~1.5-5 nm, below the normal detection) in its thin-hBN channel (~ blue - red), and up to
    # this many of those flakes per piece are added as extra 100x targets (~7 s each: 30 ~ 3.5 min); 0 = off
    thin_hbn_targets: int = 0
    # graphite: "model" = targets ranked and imaged by the kept-at-100x model (flakeid.target_model, fitted on past runs'
    # designer decisions), within the ribbon / 1-4 layer / thicker groups; "tiers" = layer tier then score, shortest path
    graphite_rank: str = "model"
    # graphite: at most this many imaged per piece (the first in graphite_rank order, ribbon places and the thick fraction
    # kept; replay of 10 pieces: 40 = 56 % of the 100x shots, 111 of 156 keeps); 0 = all
    graphite_max_targets: int = 0
    hundred_s_per_target: float = 7.6       # 100x time per target, for the popup's estimate (bn_20261020_run_1: 40.6 min / 320)
    designer_on_pc: bool = False            # ignored (kept so older settings files still load): the vdW Stack Designer
                                            # (afylab/flakeid-designer) builds each run's designer files from the results
    export_in_background: bool = True       # the 100x export (and the copy to the share) finishes in the background: the
                                            # start screen is back as soon as the microscope work ends
    analysis_wait_h: float = 2.0            # after the run: wait this long for the analysis (100x export) to finish
    # live view Autofocus button (10x): a wide coarse sweep (another holder can put the sample tens of um away), then a
    # short sweep around its best point; at 100x the af100_first search
    live_af_range_um: float = 150.0
    live_af_step_um: float = 8.0
    live_af_refine: tuple = (12.0, 3.0)
    # frames from the camera's continuous stream (the newest frame after each move, as the live view) instead of a snap
    # per frame (~0.2 s each with the XC10); false = snaps
    stream_frames: bool = True
    # ... and which frame of the stream: false = the buffer cleared after the move and the second new frame used (the
    # first may have been exposing during the move; ~1.5 frame periods, ~105 ms at 14.2 fps).  true = the first frame
    # whose exposure started after the move ended, from its time stamp (Micro-Manager's ElapsedTime-ms) taken as
    # exposure + frame_readout_ms before it reached the buffer; the old way whenever the camera gives no time stamps.
    # Off until measured at the microscope: with a readout of a whole frame period it is no faster than the old way
    frame_timestamps: bool = False
    frame_readout_ms: float = 70.0          # time from the end of a frame's exposure to its arrival in the buffer
    # coded nosepiece (Olympus U-CBS box): which objective is in.  objective_positions maps "10x", "100x" (and
    # "domain") to nosepiece positions 1..7; unknown ones are not checked (the position is still recorded)
    # device photos (start screen "Device photos", flakeid/scope/devicephotos.py): frames averaged per picture, and at 100x
    # / 50x the same at shifted stage positions (um at 100x; doubled at 50x) to average dust in the light path away
    device_name: str = ""                   # (with a stack folder: the folder's device name, see device_stack_dir)
    # the stack (device) folder the photos go to, as a fabrication step NN_<step>/ (docs/scan_guide.md,
    # "Device photos"); "" = the old way: a folder <device>_<date> copied to device_photos_to.  The start screen asks for all
    # three (the folder is remembered in <local_root>/device_photos.json)
    device_stack_dir: str = ""
    device_step: str = ""                   # the step's short name (contacts, SETs, tgetch, ...): part of the file names
    device_step_description: str = ""       # what was done in this step (step.json "description")
    device_frames: int = 16                 # 100x / 50x main picture
    device_shift_frames: int = 8            # each shifted picture
    device_frames_low: int = 4              # 10x / 5x (one position)
    device_shifts_um: object = ((10.0, 0.0), (0.0, 10.0), (-10.0, 0.0), (0.0, -10.0))
    # objective offsets (start screen "Objective offsets", flakeid/scope/objoffsets.py): per objective, where the same
    # feature is relative to the reference objective (stage um, focus um) and its pixel matrix (stage um per pixel)
    # {"50x": {"dx": .., "dy": .., "dz": .., "M": [[..], [..]], "ref": "100x", "date": ..}, ...}
    objective_offsets: object = None
    offsets_reference: str = "100x"         # the objective the others are measured from
    device_photos_to: str = ""              # without a stack folder: where the device folders are copied; "" =
                                            # <copy_results_to>/../device_photos
    nosepiece_port: str = ""                # e.g. "COM1"; "" = no nosepiece reading
    nosepiece_type: str = "ucbs"            # "ucbs": coded nosepiece, read only (nosepiece.py); "cbfm": the BX53M's
                                            # motorised one through the BX3M-CBFM box (hw/cbfm.py)
    # bx53m: the light set by the program through the nosepiece's control box when a job (or the live view) opens the
    # microscope: "brightfield" (the reflected-light lamp at illumination_level 0..255, the MIX ring off), "darkfield"
    # (the ring), "mix" (both), "" = left as it is.  The box stays logged in during the job; at its end it goes back to
    # its hand-switch state (on the glovebox microscope: the ring on, the lamp dark)
    illumination: str = ""
    illumination_level: int = 224
    nosepiece_retract_um: float = 0.0       # ... before it turns to the 50x or 100x objective the focus is lowered by
                                            # this much (the stage away from the objective); the job focuses afterwards
    nosepiece_auto: bool = False            # a motorised nosepiece is turned by the program when it needs another
                                            # objective (the stage is NOT lowered first); False: the window waits for
                                            # you to change it, as with a coded nosepiece
    objective_positions: object = None      # e.g. {"10x": 2, "100x": 5}
    # display
    live_scale: float = 0.5                 # live view at this fraction of the camera resolution
    # how the live views (10x and 100x) and the pictures beside them are shown: flipped left-right, flipped up-down.
    # (True, True) = the camera's image turned 180 deg ("O1", as the stage axes and the scan plans); display only (the
    # saved images stay as the camera takes them).  The live view's "Flip" boxes change it (remembered in
    # <local_root>/view_flip.json)
    view_flip: tuple = (True, True)
    # the main window's camera panel: the last picture the scan took (tiles, autofocus, 100x), what it is doing; display
    # only (no pictures of its own), at most ~2 a second.  Its "Show the camera" box changes it (remembered in
    # <local_root>/show_camera.json)
    show_camera: bool = True
    display_level: float = 0.31             # substrate brightness in the live view (as the chip viewers); 0 = the
                                            # camera's picture shown as it is (a camera that delivers a finished picture)
    live_ae_target: float = 0.0             # the live views' Auto exposure: the picture's median green brought to this
                                            # fraction of full scale; 0 = its brightest part to ~70 % (the XC10)


# Scan profiles (the Chips window): the settings each one changes for the job.  The window's own boxes (targets per cm2,
# Which flakes, Thin hBN search) are filled from the profile when it is chosen and can still be changed.
SCAN_PROFILES = {
    "current": dict(
        label="Current settings",
        hint="100 targets per cm\u00b2 by default, every target imaged at 100x, then the coordinate check (every 100x picture "
             "found again until two passes agree)",
        settings={}),
    "jordan_oct20": dict(
        label="Jordan's original (as of 2026-10-20)",
        hint="25 targets per cm\u00b2 (at most 400 a piece); hBN: the 48 of each piece's targets with the largest best "
             "terrace imaged; no coordinate check; Standard targets, no thin hBN search.  As the current settings: 3 flakes "
             "in a row not found at 100x ask the operator, retakes search harder",
        settings=dict(targets_per_cm2=25.0, targets_max_piece=400, hbn_imaged_per_cm2=None, hbn_max_targets=48,
                      coord_check=False, target_mode="standard", thin_hbn_targets=0)),
}


def scan_profiles(s) -> dict:
    """The scan profiles: SCAN_PROFILES and the settings file's scan_profiles (same name: the file's)."""
    out = {k: dict(v) for k, v in SCAN_PROFILES.items()}
    extra = getattr(s, "scan_profiles", None)
    if isinstance(extra, dict):
        names = {f.name for f in fields(Settings)}
        for k, v in extra.items():
            if isinstance(v, dict) and isinstance(v.get("settings", {}), dict):
                st = {a: b for a, b in (v.get("settings") or {}).items() if a in names}
                out[str(k)] = dict(label=str(v.get("label") or k), hint=str(v.get("hint") or ""), settings=st)
    return out


def apply_profile(s, name) -> bool:
    """The profile's settings into s (this job's settings).  False: no such profile (nothing changed)."""
    pr = scan_profiles(s).get(name)
    if pr is None:
        return False
    for k, v in pr["settings"].items():
        if isinstance(getattr(s, k, None), tuple) and isinstance(v, list):
            v = tuple(tuple(x) if isinstance(x, list) else x for x in v)
        setattr(s, k, v)
    s.scan_profile = name
    return True


# Defaults that differ from the Settings above for a hardware other than "mm" (the settings file and the command line
# still override them).
HARDWARE_DEFAULTS = {
    "bx53m": dict(
        camera_props=(),                    # the DP22 has no settable properties here (exposure only)
        # raw mode: linear 12-bit frames with the black level at 128, as the XC10's (hw/bx53m.py), and beside each saved
        # one the camera's own colour picture as Olympus Stream shows it: white balance 2.07 / 1 / 1 and the exposure
        # Stream's auto exposure gives (25 ms at 10x on empty 285 nm substrate, median green 124 of 255: colour_target)
        camera_raw=True, save_colour=True, wb_gains=(2.07, 1.0, 1.0),
        residue_ring_solidity=0.85,             # a clean thin flake beside glue is not residue (raw_3 picture 21)
        profile_raw="bx53m_dp22_raw_led",       # LED colour model fitted to one hBN chip (2026-11-16): not checked against AFM
        # raw frames: the substrate's green at 20 % of full scale (blue is 2.3x green on 285 nm oxide: 47 %, and a flake
        # 1.7x brighter than the substrate still fits); set from the picture at each piece (the lamp is not constant)
        exposure_ms=12.0, auto_exposure=True, target_level=0.20,
        # (strict_uniformity_um, 20 um, was a workaround for the camera's processed picture: it also switched the
        # residue rules off.  On raw frames the original rules work: of 27 glue patches imaged at 100x on 2026-11-15 they
        # reject 24, and keep two of the three real flakes; so it is off, the default)
        # 100x: Olympus Stream's pixel sizes (585.714 and 58.571 nm) give exactly 10; the focus calibration gave the
        # 100x focus 10-11 um below the 10x one; the XY offset: measured with python -m flakeid.scope.hw.offset100
        ratio_100x=10.0, offset_100x_um=(21.1, -38.4), focus_offset_um=-10.0,   # offset measured 2026-11-15
        target_exposure_ms=80.0, target_auto_exposure=True,
        # 100x autofocus on the red sites (2026-11-17): green's best focus was 0.86-0.93 um below red's at two flakes and
        # visibly soft; red's matched Olympus Stream's view
        af100_channel="red",
        # the first 100x runs here (2026-11-15): more targets than the score alone gives (it is not calibrated for this
        # camera: 2 flakes above its cut), the plain picture of each (no shifted frames, mosaics or coordinate check),
        # the raw pictures kept, and no wide focus searches with the 100x objective in (the long search +-80 um and
        # the retakes' +-120 um: a failed autofocus asks for the focus by hand instead)
        # 10x generous, 100x decides (2026-11-15): until the colour model is calibrated for this camera the 10x score
        # and thickness do not choose: every flake that passes the 10x rules is a 100x target (up to the targets per
        # cm2), best terrace 15 um wide or more, any thickness the model reads up to 400 nm
        target_min_score=0.0, target_min_d_um=15.0, hbn_range_nm=(0.0, 400.0), targets_per_cm2=300.0,
        real100="sort",                         # ... and 100x decides: glue pictures go to <piece>_100x/rejected/
        min_targets=0, coord_check=False, hundred_mosaic=False, hundred_shifts_um=(), keep_tiles=True,
        af_long_after=0, af_retake=(15.0, 1.0, 1.0, 0.25, 0.9), af_retake_extend=0,
        nosepiece_retract_um=30.0,
        z_stepsize_um=0.0, z_scale=1.0,     # the Tango's Z is in um
        overlap_z_move=False,               # the Tango drops a move sent while another axis moves
        stage_xy_scale=2.0,
        # a chip lying loose on the stage slid at every move with the controller's 0.6 m/s2 (25 um after 24 moves of
        # 300 um); at 0.2 and below it stayed within 1 um (2026-11-13).  0.1: about 0.05 s more per move
        stage_xy_accel=0.1,
        # the focus axis has no encoder and lost position at the controller's speed: the in-focus Z of one flake went
        # 269 -> 331 -> 429 um over a 100x run (3700 focus moves) and tests.  100 cycles of 5 um down and up, 2026-11-15:
        # fast (5 rev/s, 0.1 m/s2) 0, +3, +3, +4, 0 um; at 0.5 rev/s and 0.01 m/s2: 0 in five rounds
        stage_z_vel=0.5, stage_z_accel=0.01,
        af_metric="top",
        # 10x pixel -> stage um: Olympus Stream's calibration (585.714 nm per pixel: 3.69 um pixels, 0.63x adapter).
        # Measured 2026-11-13 (python -m flakeid.scope.hw.calibrate, 150 um steps): 0.589 um per pixel, both signs
        # negative, rotation below 0.5 degrees
        M=((-0.5857, 0.0), (0.0, -0.5857)),
        illumination="brightfield",
        nosepiece_type="cbfm", nosepiece_port="COM4",
        nosepiece_auto=True,                # this microscope has no hand switch: only a program can turn the nosepiece
        objective_positions={"5x": 1, "10x": 2, "20x": 3, "50x": 4, "100x": 5},   # position 6: empty
        domain_imaging="no",                # no IR camera
        substrate_colour_check="off",       # no colour references for this camera
        log_upload=False,
        update_repo="",                     # "Update..." would replace this fork's code with afylab/Flake-Finder's
    ),
}


def load(path=None, **overrides) -> Settings:
    s = Settings()
    d = {}
    if path:
        d.update(json.loads(Path(path).read_text()))
    d.update({k: v for k, v in overrides.items() if v is not None})
    hw = str(d.get("hardware", s.hardware))
    if hw != "mm":
        if hw not in HARDWARE_DEFAULTS:
            raise ValueError(f"unknown hardware {hw!r}: mm, " + ", ".join(HARDWARE_DEFAULTS))
        d = dict(HARDWARE_DEFAULTS[hw], **d)
    names = {f.name for f in fields(Settings)}
    bad = [k for k in d if k not in names]
    if bad:
        raise ValueError(f"unknown setting(s): {', '.join(bad)}")
    for k, v in d.items():
        if isinstance(getattr(s, k), tuple) and isinstance(v, list):
            v = tuple(tuple(x) if isinstance(x, list) else x for x in v)
        setattr(s, k, v)
    return s


def dump(s: Settings) -> dict:
    return {k: (v if v == v else None) if isinstance(v, float) else v for k, v in asdict(s).items()}
