@echo off

set first_run=0
set venv_path="virtualenv"
set model_path="model"

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
python -m venv %venv_path% --upgrade-deps

:ActivateVirtEnv
call %venv_path%\Scripts\activate.bat

:: for the first run, get requirements and install packages
if %first_run%==0 goto ScriptDownload
pip install wheel
:: using wget from python library instead of standalone wget for Windows
pip install wget

echo installing Python packages
if not exist requirements.txt python -m wget https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/requirements.txt
pip install -r requirements.txt
:: install onnxruntime-directml separately after onnxruntime to enable DmlExecutionProvider
pip install onnxruntime-directml

:ScriptDownload
if not exist onnxUI.py python -m wget https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/onnxUI.py
if not exist txt2img_onnx.py python -m wget https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/txt2img_onnx.py
if not exist lpw_pipe.py python -m wget https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/lpw_pipe.py

:: get version number from pip
for /F "delims= eol=|" %%f in ('pip show diffusers ^| findstr "Version"') do set version_tag=%%f
set version_tag=v%version_tag:~9%

if not exist convert_original_stable_diffusion_to_diffusers.py (
    python -m wget https://raw.githubusercontent.com/huggingface/diffusers/%version_tag%/scripts/convert_original_stable_diffusion_to_diffusers.py -o convert_original_stable_diffusion_to_diffusers.py
)
if not exist convert_stable_diffusion_checkpoint_to_onnx.py (
    python -m wget https://raw.githubusercontent.com/huggingface/diffusers/%version_tag%/scripts/convert_stable_diffusion_checkpoint_to_onnx.py -o convert_stable_diffusion_checkpoint_to_onnx.py
)
if not exist v1-inference.yaml (
    python -m wget https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml -o v1-inference.yaml
)

if not exist %model_path% mkdir %model_path%

:: update the python packages and redownload onnxUI.py
if %first_run% NEQ 0 goto FinishSetup
if %update%==0 goto FinishSetup
python -m pip install --upgrade pip
pip install -U wget wheel
:: need to uninstall these two packages to upgrade
pip uninstall --yes onnxruntime-directml
if exist requirements.txt del requirements.txt
python -m wget https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/requirements.txt
pip install -U -r requirements.txt
pip install -U onnxruntime-directml

for /F "delims= eol=|" %%f in ('pip show diffusers ^| findstr "Version"') do set version_tag=%%f
set version_tag=v%version_tag:~9%

if exist onnxUI.py del onnxUI.py
python -m wget https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/onnxUI.py
if exist txt2img_onnx.py del txt2img_onnx.py
python -m wget https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/txt2img_onnx.py
if exist lpw_pipe.py del lpw_pipe.py
python -m wget https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/lpw_pipe.py
if exist convert_original_stable_diffusion_to_diffusers.py del convert_original_stable_diffusion_to_diffusers.py
python -m wget https://raw.githubusercontent.com/huggingface/diffusers/%version_tag%/scripts/convert_original_stable_diffusion_to_diffusers.py -o convert_original_stable_diffusion_to_diffusers.py
if exist convert_stable_diffusion_checkpoint_to_onnx.py del convert_stable_diffusion_checkpoint_to_onnx.py
python -m wget https://raw.githubusercontent.com/huggingface/diffusers/%version_tag%/scripts/convert_stable_diffusion_checkpoint_to_onnx.py -o convert_stable_diffusion_checkpoint_to_onnx.py
if exist v1-inference.yaml del v1-inference.yaml
python -m wget https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml -o v1-inference.yaml

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
