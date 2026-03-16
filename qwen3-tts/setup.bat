@echo off
echo ============================================
echo   Qwen3-TTS - Setup Ambiente Virtuale
echo ============================================
echo.

REM Controlla che Python sia installato
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRORE] Python non trovato. Installa Python 3.12+ e riprova.
    pause
    exit /b 1
)

echo [1/3] Creazione ambiente virtuale...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERRORE] Impossibile creare il venv.
    pause
    exit /b 1
)

echo [2/3] Attivazione ambiente virtuale...
call venv\Scripts\activate.bat

echo [3/3] Installazione dipendenze (potrebbe richiedere diversi minuti)...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ============================================
echo   Setup completato!
echo ============================================
echo.
echo Per attivare il venv:   venv\Scripts\activate.bat
echo Script base:            python basic_tts.py
echo Script GUI:             python gui_tts.py
echo.
pause
