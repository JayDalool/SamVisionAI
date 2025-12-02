@echo off
cd /d %~dp0

REM Activate virtual environment
call venv\Scripts\activate

REM Run the main launcher
python run_samvision.py
