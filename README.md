# OnnxDiffusersUI

I’ve been helping people setup Stable Diffusion and run it on their AMD graphics card (or CPU) on Windows. I’ve also wrote a basic UI for the diffusers library version to make it more user friendly. This guide is a consolidation of what I’ve learned and hopefully will help other people setup their PC to run Stable Diffusion too.

The intent of this UI is to get people started running Stable Diffusion on Windows. As such this UI won't be as feature rich as other UI, nor will it be as fast as running Stable Diffusion on Linux and ROCm.

**WIP Update:** Known issues:
- There's been a bug where DDIM and LMS schedulers will cause an error. See [this issue page](https://github.com/huggingface/diffusers/issues/967)
- img2img output is sub-standard on PNDM scheduler, not sure if this is expected
- the seed input for img2img does not generate the same image. See [this this page](https://github.com/huggingface/diffusers/pull/932)

Example screenshot:  
![example screenshot using waifu diffusion model](images/Screenshot2.png)

## Credits

A lot of this document is based on other guides. I've listed them below:
- https://www.travelneil.com/stable-diffusion-windows-amd.html
- https://gist.github.com/harishanand95/75f4515e6187a6aa3261af6ac6f61269#file-stable_diffusion-md
- https://rentry.org/ayymd-stable-diffustion-v1_4-guide
- https://gist.github.com/averad/256c507baa3dcc9464203dc14610d674

## Prerequisites

You'll need to have a few things prepared first:
- Install Python: any version between 3.6 to 3.10 will work. I'll be using 3.10 in this guide
- Install Git: used by huggingface-cli for token authentication
- Have a [huggingface.co](https://huggingface.co) account

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
`python -m venv virtualenv`

Activate the virtual environment:  
`.\virtualenv\Scripts\activate.bat`

At this point you should be in your virtual environment and your prompt should have a `(virtualenv)` at the begining of the line. To exit the virtual environment just run `deactivate` at any time.

To restart the virtual environment after closing the command prompt window, `cd` back into the `stable_diff` folder and run the `.\virtualenv\Scripts\activate.bat` batch file again.

## Installing Packages

First, update `pip`:  
`python -m pip install --upgrade pip`

Download the requirements file and save it into your `stable_diff` folder: <https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/requirements.txt>

Then install the requirements using this command:  
`pip install -r requirements.txt`

## Download Model and Convert to ONNX

Login to huggingface:  
`huggingface-cli.exe login`  
When it prompts you for your token, copy and paste your token from the huggingface website then press enter. NOTE: when pasting, the command prompt looks like nothing has happened. This is normal behaviour, just press enter and it should update.

Go to <https://huggingface.co/runwayml/stable-diffusion-v1-5> and accept the terms for the model.

### Option 1

Go to <https://raw.githubusercontent.com/huggingface/diffusers/89d124945add51b4218cf0a9028a3966cc9dfd47/scripts/convert_stable_diffusion_checkpoint_to_onnx.py> and download the script. Save the file into your `stable_diff` folder. NOTE: make sure you save this as a `.py` file and not as `.py.txt`.

Run the Python script to download and convert:  
`python convert_stable_diffusion_checkpoint_to_onnx.py --model_path="runwayml/stable-diffusion-v1-5" --output_path="./stable_diffusion_onnx"`

### Option 2

Althernatively, you could download the pre-converted version of the model using git:  
`git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 --branch onnx --single-branch stable_diffusion_onnx`

## Basic Script and Setup Check

Download <https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/txt2img_onnx.py> and save the file into your `stable_diff` folder.

Run the Python script and check if any images were generated in the output folder. NOTE: some warnings may show up but it should be working as long as an output image is generated:  
`python txt2img_onnx.py`

If an image was generated and it's not just a blank image then you're ready to generate art! You can use the `txt2img.py` script to input your own prompt for example:  
`python txt2img_onnx.py --prompt="tire swing hanging from a tree" --height=512 --width=512`

## Running The GUI

Download <https://raw.githubusercontent.com/azuritecoin/OnnxDiffusersUI/main/onnxUI.py> and save the file into your working folder.
Run the Python script and wait for everything to load:  
`python onnxUI.py`

Once you see "Running on local URL:" open up your browser and go to "http[]()://127.0.0.1:7860". You should be able to generate images using the web UI. To close the program, go back to the command prompt and hit `ctrl-C`.

## Using Other Models

If the model is on the hugging face website and it's using the diffusers library, then you can use the same convert script from the guide. In this example I'll use waifu-diffusion.  
`python convert_stable_diffusion_checkpoint_to_onnx.py --model_path="hakurei/waifu-diffusion" --output_path="./waifu_diffusion_onnx"`

If the pretrained model is a `.ckpt` file, then you'll need to do a two step conversion. You first will need to convert from .ckpt to diffusers, then from diffusers to ONNX.

Download the following files and the `.ckpt` model of your choice and put them in your `stable_diff` folder:  
<https://raw.githubusercontent.com/huggingface/diffusers/89d124945add51b4218cf0a9028a3966cc9dfd47/scripts/convert_stable_diffusion_checkpoint_to_onnx.py>  
<https://raw.githubusercontent.com/huggingface/diffusers/d9cfe325a53502641f16ce4f839391c5b0d0a684/scripts/convert_original_stable_diffusion_to_diffusers.py>  
<https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml>

Run the first conversion script, using trinart2_step115000.ckpt in this example:  
`python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path="./trinart2_step115000.ckpt" --dump_path="./trinart2_step115000_diffusers"`  
Then run the second conversion script:  
`python convert_stable_diffusion_checkpoint_to_onnx.py --model_path="./trinart2_step115000_diffusers" --output_path="./trinart2_step115000_onnx"`  
NOTE: make sure the `--dump_path` in the first script and the `--model_path` is the same folder name.

Once you have your newly converted model, you can pass it to the scripts using the `--model` parameter:  
`python onnxUI.py --model="./waifu_diffusion_onnx"`

## Running Stable Diffusion on CPUs

If you don't have a graphics card with enough VRAM or you only have onboard graphics, you can still run Stable Diffusion with the CPU. Simply add a `--cpu-only` flag to your command line:  
`python txt2img_onnx.py --cpu-only`

## Running Other Schedulers

**WIP Update:** There has been an [issue raised](https://github.com/huggingface/diffusers/issues/967) to fix the bug but it's not in v0.6.0.

Currently the diffusers library supports PNDM, DDIM, and LMS Discrete schedulers. By default the scripts I've provided only uses PNDM. In the mean time you can edit the diffusers library directly. There's two file you need to modify: `pipeline_onnx_stable_diffusion.py` and `pipeline_onnx_stable_diffusion_img2img.py`

### Editing `pipeline_onnx_stable_diffusion.py`

Go open the first file in notepad `stable_diff\virtualenv\Lib\site-packages\diffusers\pipelines\stable_diffusion\pipeline_onnx_stable_diffusion.py`.

- On line 4, this line:  
```python
import numpy as np
```  
add this line underneath:  
```python
import numpy as np
import torch
```

- On line 153 change this:  
```python
latents = latents * self.scheduler.init_noise_sigma
```  
to this:  
```python
latents = latents * np.array(self.scheduler.init_noise_sigma)
```

- On line 171 change this:  
```python
sample=latent_model_input, timestep=np.array([t]), encoder_hidden_states=text_embeddings
```  
to this:  
```python
sample=latent_model_input, timestep=np.array([t], dtype=np.int64), encoder_hidden_states=text_embeddings
```

- On line 181, this line:  
```python
latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
```  
add this line above:  
```python
latents = torch.tensor(latents)
latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
```

### Editing `pipeline_onnx_stable_diffusion_img2img.py`

Go open the second file in notepad `stable_diff\virtualenv\Lib\site-packages\diffusers\pipelines\stable_diffusion\pipeline_onnx_stable_diffusion_img2img.py`.

- On line 327 change this:  
```python
sample=latent_model_input, timestep=np.array([t]), encoder_hidden_states=text_embeddings
```  
to this:  
```python
sample=latent_model_input, timestep=np.array([t], dtype=np.int64), encoder_hidden_states=text_embeddings
```

- On line 336, this line:  
```python
latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
```  
add this line above:  
```python
latents = torch.tensor(latents)
latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
```

### Undoing Your Changes

If you messed something up editing the two files, you need to re-install diffusers using `pip`:  
```
pip uninstall diffusers
pip install diffusers==0.6.0
```
