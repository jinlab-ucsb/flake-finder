@echo off
rem The glovebox flake scanner (Olympus BX53M): say which flakes are wanted, scan, get them sorted.
rem Close Olympus Stream first.  docs\glovebox_scanner.md
rem
rem   run_scanner.bat                     the scanner's window
rem   run_scanner.bat sort RUN [...]      sort a finished scan again without the window (python -m flakeid.gscan.collect)
rem
rem The original scan program is still there: run_flakeid.bat (also for: check, calibrate, zcal, flat fields).
setlocal
set ROOT=%~dp0..\..
set PYTHONUTF8=1
set FLAKEID_ORIG_PATH=%PATH%
set PYTHONPATH=%ROOT%
cd /d "%ROOT%"
if not exist "%ROOT%\scan_settings.json" copy /Y "%~dp0scan_settings.json" "%ROOT%\scan_settings.json" >nul
if /i "%~1"=="sort" (
  py -3.9 -m flakeid.gscan.collect %2 %3 %4 %5 %6 %7 %8 %9
) else (
  py -3.9 -m flakeid.gscan --settings "%ROOT%\scan_settings.json" %*
)
endlocal
