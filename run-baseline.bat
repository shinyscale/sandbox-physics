@echo off
REM Baseline runner for sandpipe-body-webgpu.html on powerhouse
REM Double-click to launch Chrome against the local file
REM Uses simple python http.server for static file serving

cd /d F:\sandbox-physics

echo Starting http.server on port 8765...
start /b C:\Python313\python.exe -m http.server 8765

timeout /t 2 /nobreak > nul

echo.
echo Opening sandpipe-body-webgpu.html in Chrome with WebGPU enabled...
echo.
echo BASELINE INSTRUCTIONS:
echo   1. Load a video clip (try "lieberman" from the gallery).
echo   2. Let it run for ~60 seconds.
echo   3. Note FPS and any stutter.
echo   4. Open DevTools (F12) and check Console for errors.
echo   5. Close this window when done (stops the server).
echo.

start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" ^
  --enable-unsafe-webgpu ^
  --new-window ^
  "http://localhost:8765/sandpipe-body-webgpu.html"

pause

taskkill /f /im python.exe > nul 2>&1
