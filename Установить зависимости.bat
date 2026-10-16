@echo off
chcp 65001 >nul
title Установка зависимостей
cd /d "%~dp0"
echo Установка Python-зависимостей...
python -m pip install -r requirements.txt
echo.
echo Проверка окружения:
python main.py --check
pause
