@echo off
chcp 65001 >nul
title Проверка окружения
cd /d "%~dp0"
python main.py --check
pause
