@echo off
REM Super Alita Simple Startup Script for Windows
REM This batch file starts the Super Alita agent and opens the chat interface

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    🤖 Super Alita Agent                     ║
echo ║              AI-Powered Development Assistant                ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo 🚀 Starting Super Alita Agent System...
echo.

REM Check if we're in the right directory
if not exist "app.py" (
    echo ❌ Error: app.py not found. Please run this script from the Super Alita root directory.
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH.
    pause
    exit /b 1
)

REM Start the main server
echo Starting Super Alita server...
echo.

REM Use start command to run in background and open browser after delay
start /B python -m uvicorn app:app --host 127.0.0.1 --port 8080 --reload

REM Wait a few seconds for server to start
echo Waiting for server to start...
timeout /t 5 /nobreak >nul

REM Try to check if server is responding
curl -s http://127.0.0.1:8080/health >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Server may still be starting up...
) else (
    echo ✅ Server is responding!
)

REM Open the chat interface in default browser
echo.
echo 🌐 Opening chat interface in browser...
start http://127.0.0.1:8080

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                     🎉 Super Alita Ready!                   ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo 🌐 Chat Interface: http://127.0.0.1:8080
echo 📊 Health Check: http://127.0.0.1:8080/health
echo 📚 API Docs: http://127.0.0.1:8080/docs
echo.
echo 📝 The server is running in the background.
echo    Close this window or press Ctrl+C to stop the server.
echo.

REM Keep the window open
pause
