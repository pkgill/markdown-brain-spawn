@echo off
cd /d "%~dp0"
start "" /min pythonw main.py --config config.yaml
