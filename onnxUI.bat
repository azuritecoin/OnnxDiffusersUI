@echo off

set first_run=0
set venv_path="virtualenv"
set main_script="onnxUI.py"
set default_model="runwayml/stable-diffusion-v1-5"

:: check if programs are installed
python --version 1> NUL 2> NUL
if %errorlevel% NEQ 0 goto NoPython
git --version 1> NUL 2> NUL
if %errorlevel% NEQ 0 goto NoGit
pip --version 1> NUL 2> NUL
if %errorlevel% NEQ 0 goto NoPip

:: parse the parameters
set model=
set cpu_only=0

:ArgParseLoop
if not "%1"=="" (
    if "%1"=="-model" (
        set model=%2
        shift
    )
    if "%1"=="-cpu-only" (
        set cpu_only=1
        shift
    )
    shift
    goto ArgParseLoop
)
if "%model%"=="" set model=%default_model%

echo %model%
echo %cpu_only%

:: check if virtual environment is present, else create it and install requirements
if exist %venv_path%\Scripts\activate.bat goto SkipCreateVEnv

echo no %venv_path% folder detected, creating virtual environment
set first_run=1
python -m venv %venv_path%

:SkipCreateVEnv
call %venv_path%\Scripts\activate.bat
:: using wget from python library instead of standalone wget for Windows
if not exist %venv_path%\Lib\site-packages\wget.py pip install wget

if %first_run% NEQ 0 goto FirstRunInstall

:AfterFirstRun
:: check for token
rem onnxUI.py --model="stable_diffusion_onnx-v1.5"

pause
goto:eof


:FirstRunInstall
:: download files from OnnxDiffusersUI GitHub
echo installing Python packages

if not exist requirements.txt %venv_path%\Lib\site-packages\wget.py https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/requirements.txt
pip install -r requirements.txt

rem if not exist onnxUI.py %venv_path%\Lib\site-packages\wget.py https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/onnxUI.py

goto AfterFirstRun


:NoPython
echo python not found
goto CheckError

:NoGit
echo git not found
goto CheckError

:NoPip
echo pip not found
goto CheckError

:CheckError
pause
goto:eof
