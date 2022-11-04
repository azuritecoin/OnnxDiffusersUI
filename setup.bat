@echo off

set first_run=0
set venv_path="virtualenv"

:: check if programs are installed
python --version 1> NUL 2> NUL
if %errorlevel% NEQ 0 goto NoPython
git --version 1> NUL 2> NUL
if %errorlevel% NEQ 0 goto NoGit
pip --version 1> NUL 2> NUL
if %errorlevel% NEQ 0 goto NoPip

:: parse the parameters
set update=0

:ArgParseLoop
if not "%1"=="" (
    if "%1"=="-update" (
        set update=1
    )
    shift
    goto ArgParseLoop
)

:: check if virtual environment is present, else create it and install requirements
if exist %venv_path%\Scripts\activate.bat goto ActivateVirtEnv

echo no %venv_path% folder detected, creating virtual environment
set first_run=1
python -m venv %venv_path%

:ActivateVirtEnv
call %venv_path%\Scripts\activate.bat

:: for the first run, get requirements and install packages
if %first_run%==0 goto ScriptDownload
python -m pip install --upgrade pip
pip install wheel
:: using wget from python library instead of standalone wget for Windows
pip install wget

echo installing Python packages
if not exist requirements.txt python -m wget https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/requirements.txt
pip install -r requirements.txt
:: install onnxruntime-directml separately
pip install "protobuf<=3.20.1" onnxruntime-directml
:: install latest version of protobuf v3.x
pip install --no-deps --ignore-installed "protobuf<4"

:ScriptDownload
if not exist onnxUI.py python -m wget https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/onnxUI.py

:: update the python packages and redownload onnxUI.py
if %first_run% NEQ 0 goto FinishSetup
if %update%==0 goto FinishSetup
python -m pip install --upgrade pip
pip install -U wget wheel
pip uninstall --yes protobuf
if exist requirements.txt del requirements.txt
python -m wget https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/requirements.txt
pip install -U -r requirements.txt
pip install -U "protobuf<=3.20.1" onnxruntime-directml
pip install --no-deps --ignore-installed "protobuf<4"

if exist onnxUI.py del onnxUI.py
python -m wget https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/onnxUI.py

:FinishSetup
echo setup complete
pause
deactivate
goto :eof


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
goto :eof
