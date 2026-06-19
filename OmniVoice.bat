@echo off
title OmniVoice TTS
setlocal

set PROJECT=%~dp0
if "%PROJECT:~-1%"=="\" set PROJECT=%PROJECT:~0,-1%
set VENV=%PROJECT%\.venv\Scripts
set PORT=8001

cd /d "%PROJECT%"

:: Check if already running
netstat -aon 2>nul | findstr ":%PORT% " | findstr "LISTENING" >nul 2>&1
if %errorlevel%==0 (
    echo.
    echo  OmniVoice ja esta rodando na porta %PORT%.
    echo  Abrindo no navegador...
    start http://localhost:%PORT%
    pause
    exit /b
)

echo.
echo  ============================================
echo       OmniVoice - Text to Speech
echo  ============================================
echo.
echo  Carregando modelo (primeira vez demora mais)...
echo  O navegador vai abrir automaticamente.
echo  Para parar: feche esta janela.
echo.

:: Open browser after delay
start /b cmd /c "timeout /t 20 /nobreak >nul && start http://localhost:%PORT%"

"%VENV%\python.exe" -m omnivoice.cli.demo --ip 0.0.0.0 --port %PORT%

if %errorlevel% neq 0 (
    echo.
    echo  ERRO ao iniciar o OmniVoice.
    echo  Verifique se o ambiente virtual esta instalado.
    pause
)
