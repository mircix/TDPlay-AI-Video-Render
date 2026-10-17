@echo off
rem Opens TDPlay AI Video Render in your browser, with this window showing the log.
rem (The Start menu and desktop shortcuts open it in its own window instead.)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo TDPlay AI Video Render isn't installed yet. Double-click install.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m tdplay
pause
