@echo off
if not exist "%~dp0.app-env\Scripts\pythonw.exe" (
  echo Traffic AI is not set up yet. Double-click Install_TrafficAI.bat first.
  pause
  exit /b 1
)
start "" "%~dp0.app-env\Scripts\pythonw.exe" "%~dp0TrafficAI.pyw"
