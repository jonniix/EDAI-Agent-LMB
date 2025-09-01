@echo off
setlocal

REM Crea venv se non esiste
if not exist ".venv" (
    python -m venv .venv
)

REM Attiva venv
call .venv\Scripts\activate

REM Installa dipendenze
pip install --upgrade pip
pip install -r requirements.txt

REM Avvia Streamlit
streamlit run main.py

endlocal
