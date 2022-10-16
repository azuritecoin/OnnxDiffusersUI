# DiffusersOnnxUI

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

