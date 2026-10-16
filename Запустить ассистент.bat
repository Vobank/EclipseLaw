@echo off
chcp 65001 >nul
title Ассистент Eclipse RP
cd /d "%~dp0"
start "" pythonw main.py
