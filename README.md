# FlakeIdentifier

Automated search for exfoliated 2D flakes (hBN, graphite / graphene) on SiO2/Si chips, from the microscope to the
heterostructure design. Developed in the Young Lab, UCSB, for building van der Waals devices.

**Who it is for.** The analysis (finding flakes, their thickness and quality in optical images) works on any
microscope's pictures once its camera is calibrated (see [docs/manual.md](docs/manual.md)). The scan program is written
for one setup: an Olympus BX51 microscope with a XIMEA XC10 camera, an Oasis motorised stage and focus drive, and a
coded nosepiece, run through Micro-Manager on a Windows 7 PC with Python 3.8. Another setup needs its own
Micro-Manager configuration, camera calibration and flat fields, and some of the settings in
`flakeid/scope/settings.py` adapted.

A second setup is being added: the glovebox Olympus BX53M (Märzhäuser Tango stage, Olympus DP22 camera, motorised
nosepiece), driven through the vendors' libraries without Micro-Manager (`flakeid/scope/hw/`, `--hw bx53m`). The
hardware layer works; calibration and the 100x part do not yet. See [docs/bx53m.md](docs/bx53m.md).

**Status.** In daily use in the lab and changing quickly; see [CHANGELOG.md](CHANGELOG.md).

1. **Scan** (microscope PC): an Olympus BX51 with an XIMEA XC10 camera and an Oasis stage, driven from Python through
   Micro-Manager's device drivers. It finds each chip's edges, focuses, scans it at 10x, and images the best flakes at
   100x (optionally the stacking domains of graphite with an IR camera).
2. **Analyse** (on the same PC, while it scans): flakes and terraces from the 10x tiles, thickness from calibrated
   optical contrast, a learned score, the 100x target list, and a zoomable chip viewer with the 100x images.
3. **Design** (any computer): the vdW Stack Designer places flakes from the scans into stacks and checks overlaps and
   shorts. It lives in its own repository, [afylab/flakeid-designer](https://github.com/afylab/flakeid-designer), and
   builds its files from the scan results this repository writes.

## Repository layout

| path | what |
|---|---|
| `flakeid/` | the analysis package: identification (`pipeline`, `segment`, `thickness`), live analysis during a scan (`live`), 100x export and screening (`hundred`, `screen100`), chip viewer (`zoomview`, `livecanvas`), calibration profiles (`profiles/`) |
| `flakeid/scope/` | the scan program for the microscope PC (`python -m flakeid.scope`): devices, chip finding, focus, tiles, 100x, window UI |
| `tools/` | calibration, labelling, model fitting, repairs of built viewers (`reattach_100x.py`), Micro-Manager simulator (`mm_mock/`) |
| `tools/win7/` | the microscope PC (Windows 7, Python 3.8): offline install kit, launcher, Micro-Manager and nosepiece tests |
| `ml/` | research code: neural-network screening experiments (not used by the scan program; see its README) |
| `tools/archive/` | one-off analyses and labelling-set makers from development, kept for reference (see its README) |
| `flats/` | flat fields (not in git; see `flats/README.md`) |
| `find_flakes.py` | command line for one stitched overview image (`python find_flakes.py -h`) |
| `docs/scan_guide.md` | **how to scan**: step-by-step use of the scan program, common mistakes, setup and calibration (also as `docs/scan_guide.pdf`) |
| `docs/manual.md` | **how the analysis works**: flakes, thickness, scores, outputs, calibration, tools and the Python API |

Data (scans, images, results, label sets, flat fields) is not in the repository, including the example chip pictures
the simulator and some examples use (`example images/`).

## Quick starts

**Analysis on a workstation** (Python 3.9+):

```sh
pip install -r requirements.txt
python find_flakes.py IMAGE hbn 285 --annotate          # one stitched overview image
python -m flakeid.live RUN_FOLDER -o RESULTS            # a Micro-Manager scan (tiles), live or afterwards
python -m flakeid.hundred RUN/<piece>_100x --viewer RESULTS/<piece>/<piece>_tiles_viewer   # 100x images into a viewer
```

**Microscope PC**: build the offline kit on a Mac or Linux machine with the flat fields in `flats/`
(`sh tools/win7/make_kit.sh`), copy `scope_pc_kit/` to the PC and run `setup_scope_pc.bat` there. It installs Python 3.8
into `C:\FlakeID`, creates a desktop shortcut and pins it to the taskbar. Settings: `C:\FlakeID\scan_settings.json`
(see `flakeid/scope/settings.py`). Useful commands: `C:\FlakeID\run_flakeid.bat check`, `... mmtest`, `... objtest`, `... screenshots`. Using it: [docs/scan_guide.md](docs/scan_guide.md).
Later versions install from GitHub with **Update...** on the start screen (or `run_flakeid.bat update`); only new
packages need a new kit. A private fork as `update_repo` needs a read-only token in `C:\FlakeID\github_token.txt`.

**Designer** and **device design in KLayout** (placing microscope pictures of a stack on the chip layout by its
alignment marks): see [afylab/flakeid-designer](https://github.com/afylab/flakeid-designer).

## Development

- **Microscope PC compatibility.** Everything the microscope PC runs (`flakeid/`, `flakeid/scope/`, `tools/win7/`) must work on Windows 7 with Python 3.8: check the syntax
  with `ast.parse(source, feature_version=(3, 8))` (no `X | Y` types at run time, no f-strings reusing their own quote
  character, no `match`), pass `str` (not `Path`) to `np.memmap`, write text as UTF-8 explicitly, and expect Windows file
  locking (virus scanner, indexer): retry renames, never let a status or log write stop a run.
- **Rebuild the kit** after such changes: `sh tools/win7/make_kit.sh` (needs the flat fields in `flats/`), then
  `python3 tools/win7/check_kit.py scope_pc_kit/wheels tools/win7/requirements-py38.txt tools/win7/requirements-mm-py38.txt`
  must report nothing missing.
- **Designer.** afylab/flakeid-designer keeps its own copy of the analysis code (`flakeid/`) to build its runs: a change
  to the analysis that the designer should see is made there too.
- **Tests.** The Micro-Manager simulator tests the scan program without hardware:
  `python tools/mm_mock/run_mock_py.py` (options: `--graphite`, `--mounted`, `--nosepiece`, `--partial`, `--mosaic`,
  `--stage-timeout`, ...). It needs example chip pictures as `example images/` beside the code, which are not in the
  repository: stitched 10x chip pictures (`<name>.jpg`, picked with `--chips`) and 100x pictures in
  `example images/demo_high_magnification/<flake>/*_100x.jpg`. The
  chip viewer pages are checked in real browsers with Playwright.
- **Data** (scans, pictures, results, label sets, flat fields) stays outside the repository.

## License

MIT (see [LICENSE](LICENSE)). `flakeid/vendor/` contains OpenSeadragon (BSD-3-Clause) and `flakeid/scope/theme_sv/` the
Sun Valley ttk theme (MIT), each with its own license.
