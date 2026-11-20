# The glovebox flake scanner

A second program for the glovebox microscope (Olympus BX53M). You say which flakes you want, it scans the chip at 10x,
photographs every candidate at 100x, and sorts the pictures into wanted flakes, other real flakes and tape glue.

The original scan program is still there and unchanged in how it is started (`tools\bx53m\run_flakeid.bat`). The new
program uses its stage, focus, camera and autofocus code as it is, including the focus safety limits; see
[bx53m.md](bx53m.md) for the hardware, the calibrations and the flat fields.

## Running it

Close Olympus Stream first (it holds the stage and the camera). Then:

```
tools\bx53m\run_scanner.bat
```

1. **Fill in the flakes wanted** (the values are remembered):

   | box | meaning |
   |---|---|
   | Thickness, nm | from the flake's colour. **Not checked against AFM yet.** |
   | Size of the flat area, µm | the largest circle that fits in the flake's flat, one-colour area. 0 as the upper value = no limit |
   | Cleanliness, % | least share of the flake that is flat and clean in the 100x picture. 0 = any. Speckled or stepped flakes are 20–40 %, clean ones 75 % and up |
   | Real flake, not glue | least probability from the 100x sorter. 0.5 is the usual value; lower keeps more doubtful pictures |

2. **Start scan.** Check the list in the message that appears (chip on the stage, nothing in the way: the stage, the
   focus and the nosepiece move by themselves). A live view opens: put the cross on each corner of the area to scan and
   press *Add point*, then *Finish polygon* and *Done*. From here the scan needs nobody: 10x tiles, analysis, the
   nosepiece turns to 100x, one picture per candidate. The scan's own window has *Stop* and *Pause*.
3. **The results page opens by itself** when the scan ends (also when it was stopped: what was photographed is sorted).

About 2 s per 10x tile and 7 s per 100x picture: the 0.56 cm² test chip is 99 tiles and, with everything photographed,
about 250 pictures, so roughly 35 minutes.

## What "wanted" means: nothing is thrown away at 10x

At 10x a patch of glue can look like a thin flake and the thickness is only as good as the colour model, so the 10x
step only chooses where to look. It leaves a flake out only if its flat area is well below the smallest size wanted
(under 60 % of it), or, with the *Faster* box ticked, if it is clearly thicker than the window. The thin end is never
cut at 10x. The criteria are applied after the 100x pictures:

| group | what is in it |
|---|---|
| **Wanted** | real flake; thickness, size and cleanliness all inside your values |
| **Possibly wanted** | real flake that may be inside them: no flat 10x region under the middle of the picture (thickness and size unknown), or its colour is also that of a thickness inside the window. Look at these |
| **Other real flakes** | real, but outside your values |
| **Not sure** | the sorter gives a real-flake probability between 0.15 and your value |
| **Glue** | below 0.15 |

The **thin-flake search** (on by default) adds a second 10x search for ~1–5 nm hBN, which is too faint for the normal
one, and photographs up to the given number of its finds. It has not been proven on this camera: on the test chip it
reported 109 places and none of them had been photographed, so expect many of them to be glue until a scan shows
otherwise.

## The results

`C:\FlakeScans\results\<scan>\sorted\` (and the folder given under *Copy the results to*):

- `index.html`: one card per flake with the 100x picture, the scan's 10x picture, thickness, size, cleanliness and
  stage position. **The numbers are of what lies under the white cross in the middle of the 100x picture.** The boxes
  at the top re-sort the page on the spot, so you can try other values without the scanner.
- `match\` and `possible\`: the full-size colour pictures of those two groups, named
  `<thickness>nm_<size>um_<chip>_<picture>_x<stage x>_y<stage y>.png`.
- `flakes.csv`: every flake with all its numbers. Stage positions are with the 100x objective in; `x10_um`, `y10_um`
  are the same place with the 10x objective in.

**Sort again**: pick a scan under *Earlier scan*, change the values, press *Sort again with the values above*. This
needs no microscope and takes under a minute; it rewrites the page and the two picture folders. Without the window:
`tools\bx53m\run_scanner.bat sort <scan>` (for example `sort raw_run_3`).

**Marking by eye**: *Real flake* / *Glue* on a card overrides the sorter on that page (kept in the browser). *Save my
marks* downloads them as a labels file; `tools\fit_real100.py` trains the sorter from such files.

## How a picture gets its numbers

- The place under the cross is the picture's stage position less the 10x → 100x offset (`offset_100x_um`). Its 10x
  region is the flat region whose largest inscribed circle contains that place. Thickness and size are that region's,
  never a neighbour's.
- Small regions: the 100x step re-centres each picture on the 10x picture (up to ~10 µm), which can leave the cross just
  outside a small region's circle. Then the region the scan aimed that picture at counts (the card says how far out).
- **Flat areas of the 100x picture itself** (`colour100.regions`): every picture is also cut into its flat, one-colour
  areas with a clear edge, each with a size and a thickness from the 100x colour alone. The 10x analysis can miss a
  faint thin flake next to the thicker piece it targeted (e.g. a 3 nm, 28 µm flake beside an 11.6 nm piece); a picture
  is also *wanted* when one of these areas fits the criteria, and a wanted area much larger than the 10x region heads
  the card. Areas thinner than ~1.5 nm are not found this way (the substrate's colour drifts by as much across a frame).
- The same region photographed twice gives one card (the picture nearest the region's middle).
- Real or glue: `flakeid/real100.py` on the raw 100x frame (edge sharpness, flatness, mottle).
- Where one 10x colour fits two thicknesses (about 30 and 316 nm, or 97 and 236 nm), the 100x colour decides when it
  is clear (`flakeid/gscan/colour100.py`); the card says so. Where the 100x colour alone disagrees with a single 10x
  reading, the card says "check by eye" and both values count for the sorting.

## Known limits

- Thickness is not checked against AFM. On the test chip the 10x and 100x colours agree within 5 nm for 61 of 70
  flakes, which shows consistency, not correctness.
- A flake the 10x rules reject as residue is not photographed (103 regions on the test chip, 15 of them known glue,
  the rest not looked at). The scanner turns on `faint_flakes` (`flakeid/pipeline.py`, `FAINT_FLAKES`): a faint region
  left out only as too faint or residue-like is kept when it is clearly a film: the colour of 1.5 nm or more of hBN,
  flat, a sharp edge and no haze around it. On chip 20261117_1 this caught a 2 nm hBN flake 25 µm across that had been
  missed (found on another microscope), and added 7 regions of 562 left out.
- **Focus drift.** The focus of this microscope drifts by a few um per hour, and not always the same way (20261117_1:
  +1.9 um/h during the scan, -1.2 um/h over the next 4 h, then +4 um/h). Every 100x picture is focused on its own (a
  sweep of +-2.4 um around a predicted Z), so the prediction only has to be close. It follows the drift
  (`focus_drift_half_life_min`, `focus_drift_window_min` in the settings): the latest flakes count most and the drift
  rate of the last hour is carried forward. On a replay of that scan stretched to 4-12 h, 95 % of the predictions were
  within 0.8 um; by distance alone they were 3-9 um off. The scan log gives the rate (`focus drift ... um/h`).
- **100x focus colour.** The 100x objective brings red to focus about 0.9 um higher than green and blue. The
  autofocus measured green until 2026-11-17, which left the 100x pictures visibly softer than Olympus Stream's view;
  it now measures red (`af100_channel`), which matched Stream at two flakes. Pictures taken before then are slightly
  soft.
- One 100x picture per flake, at its best flat region: a thin step on a thick flake is not photographed on its own.
- **Disk space.** The scan keeps its raw 10x tiles and 100x frames in `C:\FlakeScans` so that it can be sorted again:
  0.6 GB of tiles and about 14 MB per 100x picture, so roughly 4.5 GB for the test chip with 250 pictures. The window
  warns when less than that is free. Delete the `<chip>_tiles` and `<chip>_100x` folders of scans you will not sort
  again; their `results\<scan>\sorted` folder keeps the page and the wanted flakes' pictures.
- Graphite can be chosen but uses the other microscope's colour profile; only hBN has been set up here.
- The analysis runs with one worker on this PC (7.8 GB of memory); two or more make it swap.

## Files

| | |
|---|---|
| [flakeid/gscan/app.py](../flakeid/gscan/app.py) | the window |
| [flakeid/gscan/criteria.py](../flakeid/gscan/criteria.py) | the criteria, the scan settings that follow from them, the sorting rule |
| [flakeid/gscan/engine.py](../flakeid/gscan/engine.py) | one scan job through the original program's `Scanner` |
| [flakeid/gscan/collect.py](../flakeid/gscan/collect.py) | joins pictures to 10x regions, judges, sorts |
| [flakeid/gscan/colour100.py](../flakeid/gscan/colour100.py) | the 100x colour check; profile `bx53m_dp22_raw_led_100x` |
| [flakeid/gscan/page.py](../flakeid/gscan/page.py) | the results page |
