@echo off
rem Starts the ollama server with project-local model storage, so the ~2GB
rem model lands next to the data instead of in the user profile.
rem
rem %~dp0 is this script's own folder - set NLTM_DATA_DIR first if your data
rem lives somewhere else.

if "%NLTM_DATA_DIR%"=="" set NLTM_DATA_DIR=%~dp0
if "%NLTM_OLLAMA_HOST%"=="" set NLTM_OLLAMA_HOST=127.0.0.1:11434

set OLLAMA_MODELS=%NLTM_DATA_DIR%\ollama_models
set OLLAMA_HOST=%NLTM_OLLAMA_HOST%

echo models : %OLLAMA_MODELS%
echo host   : %OLLAMA_HOST%

rem if 11434 refuses to bind, it may sit inside a reserved windows port
rem range - check with:  netsh interface ipv4 show excludedportrange protocol=tcp
rem then set NLTM_OLLAMA_HOST to a free port and re-run.

"%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
