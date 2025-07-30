@echo on
cd /d %~dp0

echo === Activation de l'environnement virtuel ===
call .\venv\Scripts\activate

echo === Lancement de FastAPI ===
uvicorn src.tags_suggester.api.main:app --reload --host 127.0.0.1 --port 8000

pause
