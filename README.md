# OnnxDiffusersUI

I’ve been helping people setup Stable Diffusion and run it on their AMD graphics card on Windows. I’ve also wrote a basic UI for this version. This guide is a consolidation of what I’ve learned and hopefully will help other people setup their PC to run Stable Diffusion too.

## Credits

A lot of this document is based on other guides. I've listed them below:
- https://www.travelneil.com/stable-diffusion-windows-amd.html
- https://gist.github.com/harishanand95/75f4515e6187a6aa3261af6ac6f61269#file-stable_diffusion-md
- https://rentry.org/ayymd-stable-diffustion-v1_4-guide

## Pre-requisites

You'll need to install a few things:
- Python: any version between 3.6 to 3.10 will work. I'll be using 3.10 in this guide
- Git: used by huggingface-cli for token authentication
- [huggingface.co](https://huggingface.co) account

To check if they’re installed properly open up command prompt and run the following commands:  
```
python --version
git --version
pip --version
```  
There shouldn't be any "not recognized as an internal or external command" errors.

## Creating a Workspace

Start by creating a folder somewhere to store your project. I named mine `stable_diff`. Open up command prompt (or PowerShell) and navigate to your folder.

Create a Python virtual environment:  
`python -m venv .\virtualenv`

Activate the virtual environment:  
`.\virtualenv\Scripts\activate.bat`

At this point you should be in your virtual environment and your prompt should have a `(virtualenv)` at the begining of the line. To exit the virtual environment just run `deactivate` at any time.

## Installing Packages

First, update `pip`:  
`python -m pip install --upgrade pip`

Install the following packages:  
```
pip install diffusers==0.4.1
pip install transformers
pip install onnxruntime
pip install onnx
pip install torch
pip install gradio
```

Go to <https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/PyPI/ort-nightly-directml/overview/> and download the latest version of DirectML for your version of Python. Save the file into your working folder.  
> If you are on Python 3.7 download the file that ends with **-cp37-cp37m-win_amd64.whl  
> If you are on Python 3.8 download the file that ends with **-cp38-cp38-win_amd64.whl  
> If you are on Python 3.9 download the file that ends with **-cp39-cp39-win_amd64.whl  
> If you are on Python 3.10 download the file that ends with **-cp310-cp310-win_amd64.whl  

Install the downloaded file using `pip`. Note the `--force-reinstall` is needed:  
`pip install ort_nightly_<whatever_version_you_got>.whl --force-reinstall`

## Download Model and Convert to ONNX

Login to huggingface:  
`huggingface-cli.exe login`  
When it prompts you for your token, copy and paste your token from the huggingface website then press enter. NOTE: when pasting, the command prompt looks like nothing has happened. This is normal behaviour, just press enter and it should update.

Go to <https://huggingface.co/CompVis/stable-diffusion-v1-4> and accept the terms for the model.

Go to <https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py> and download the script. Save the file into your working folder. NOTE: make sure you save this as a `.py` file and not as `.py.txt`.

Run the Python script to download and convert:  
`python convert_stable_diffusion_checkpoint_to_onnx.py --model_path="CompVis/stable-diffusion-v1-4" --output_path=".\stable_diffusion_onnx"`

## Basic Script and Setup Check

Download <https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/txt2img.py> and save the file into your working folder.

Run the Python script and check if any images were generated in the output folder. NOTE: some warnings may show up but it should be working as long as an output image is generated:  
`python .\txt2img.py`

If an image was generated and it's not just a blank image then you're ready to generate art! You can use the `txt2img.py` script to input your own prompt:  
`python .\txt2img.py --prompt="tire swing hanging from a tree" --height=512 --width=512`

## Running The GUI

Download <https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/onnxUI.py> and save the file into your working folder.
Run the Python script and wait for everything to load:  
`python .\onnxUI.py`

Once you see "Running on local URL:" open up your browser and go to "127.0.0.1:7860". You should be able to generate images using the web UI. To close the program, go back to the command prompt and hit `ctrl-C`.


