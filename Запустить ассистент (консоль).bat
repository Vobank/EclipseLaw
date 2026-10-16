@echo off
chcp 65001 >nul
title Ассистент Eclipse RP (консоль)
cd /d "%~dp0"
python main.py
pause
