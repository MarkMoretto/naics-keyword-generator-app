@ECHO Off

::SET PYTHONPATH=%CD%;%PYTHONPATH%

::SET PRODUCTION=1
@REM SET PROD_LOG_LEVEL=info
@REM SET DEV_LOG_LEVEL=debug

@REM SET LOG_FOLDER=src\logging

::SET STARTCMD=python -m src
::CALL venv\Scripts\activate && %STARTCMD%
::PAUSE



::SET port=8000
::SET hostdev=127.0.0.1
:: SET hostprod=0.0.0.0
::SET GUNICORN_CMD_ARGS="--bind=0.0.0.0 --workers=3 --log-level 'info'"
CD server-side
SET entrypoint=src.main:app
SET startdev=uvicorn %entrypoint% --reload
::PAUSE

::SET startdev=uvicorn --host %hostdev% --port %port% --reload --log-config %LOG_FOLDER%\config.yaml --log-level %DEV_LOG_LEVEL% --backlog 2048 %entrypoint%
::SET startprod=gunicorn -b %hostprod%:%port% -w 2 --log-level '%LOG_LEVEL%' --error-logfile %LOG_FOLDER%\err.log %entrypoint%


CALL venv\Scripts\activate && %startdev%
PAUSE