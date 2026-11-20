@echo off
rem FlakeIdentifier on the glovebox microscope PC (Olympus BX53M: Tango stage, DP22 camera, BX3M-CBFM nosepiece;
rem Windows 10, Python 3.9, no Micro-Manager).  Close Olympus Stream first.
rem
rem   run_flakeid.bat                 the scan program (start screen)
rem   run_flakeid.bat check           every device through the program's hardware layer, nothing moves
rem   run_flakeid.bat calibrate ...   measure the pixel -> stage matrix M (MOVES THE STAGE; --write puts it in the settings)
rem   run_flakeid.bat zcal            focus of every objective and the focus drive's safety limit (you move Z with its buttons)
rem   run_flakeid.bat faketest        the hardware layer on stand-in devices (no hardware)
rem   run_flakeid.bat <module> ...    any other flakeid module (live, hundred, mm, ...)
rem
rem The settings are scan_settings.json in the program folder (made from tools\bx53m\scan_settings.json the first time).
setlocal
set ROOT=%~dp0..\..
set PYTHONUTF8=1
rem (this PC's own Python and its packages are used as they are: no isolated install as on the BX51's PC)
set FLAKEID_ORIG_PATH=%PATH%
set PYTHONPATH=%ROOT%
cd /d "%ROOT%"
if not exist "%ROOT%\scan_settings.json" copy /Y "%~dp0scan_settings.json" "%ROOT%\scan_settings.json" >nul
if "%~1"=="" (
  py -3.9 -m flakeid.scope --settings "%ROOT%\scan_settings.json"
) else if /i "%~1"=="check" (
  py -3.9 -m flakeid.scope.hw.check --settings "%ROOT%\scan_settings.json" %2 %3 %4 %5 %6 %7 %8 %9
) else if /i "%~1"=="calibrate" (
  py -3.9 -m flakeid.scope.hw.calibrate --settings "%ROOT%\scan_settings.json" %2 %3 %4 %5 %6 %7 %8 %9
) else if /i "%~1"=="zcal" (
  py -3.9 -m flakeid.scope.hw.zcal --settings "%ROOT%\scan_settings.json"
) else if /i "%~1"=="faketest" (
  py -3.9 tools\hw_fake_test.py
) else (
  py -3.9 -m flakeid.%*
)
endlocal
