import argparse
import functools
import gc
import os
import re
import time
import cv2
from typing import Optional, Tuple
from math import ceil
import tempfile
import signal
import shutil

from diffusers import (
    OnnxRuntimeModel,
    OnnxStableDiffusionPipeline,
    OnnxStableDiffusionImg2ImgPipeline,
    OnnxStableDiffusionInpaintPipeline,
    OnnxStableDiffusionInpaintPipelineLegacy,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler
)
from diffusers import __version__ as _df_version
import gradio as gr
import numpy as np
from packaging import version
import PIL

import lpw_pipe

# We want to save data to PNG
from PIL import Image, PngImagePlugin

# gradio function
def run_diffusers(
    prompt: str,
    neg_prompt: Optional[str],
    init_image: Optional[PIL.Image.Image],
    init_mask: Optional[PIL.Image.Image],
    iteration_count: int,
    batch_size: int,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    eta: float,
    denoise_strength: Optional[float],
    seed: str,
    image_format: str,
    legacy: bool,
    savemask: bool,
    loopback: bool,
    preprocess: bool,
) -> Tuple[list, str]:
    global model_name
    global controlnet_name
    global controlnet
    global current_pipe
    global pipe
    global original_steps
    conditioning_scale_t3 = denoise_strength

    prompt.strip("\n")
    neg_prompt.strip("\n")

    # generate seeds for iterations
    if seed == "":
        rng = np.random.default_rng()
        seed = rng.integers(np.iinfo(np.uint32).max)
    else:
        try:
            seed = int(seed) & np.iinfo(np.uint32).max
        except ValueError:
            seed = hash(seed) & np.iinfo(np.uint32).max

    # use given seed for the first iteration
    seeds = np.array([seed], dtype=np.uint32)

    if iteration_count > 1:
        seed_seq = np.random.SeedSequence(seed)
        seeds = np.concatenate((seeds, seed_seq.generate_state(iteration_count - 1)))

    # create and parse output directory
    output_path = "output"
    os.makedirs(output_path, exist_ok=True)
    dir_list = os.listdir(output_path)
    if len(dir_list):
        pattern = re.compile(r"([0-9][0-9][0-9][0-9][0-9][0-9])-([0-9][0-9])\..*")
        match_list = [pattern.match(f) for f in dir_list]
        next_index = max([int(m[1]) if m else -1 for m in match_list]) + 1
    else:
        next_index = 0

    sched_name = pipe.scheduler.__class__.__name__
    neg_prompt = None if neg_prompt == "" else neg_prompt
    images = []
    time_taken = 0
    for i in range(iteration_count):
        print(f"iteration {i + 1}/{iteration_count}")

        # create generator object from seed
        rng = np.random.RandomState(seeds[i])

        if current_pipe == "txt2img":
            start = time.time()
            batch_images = pipe(
                prompt,
                negative_prompt=neg_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                eta=eta,
                num_images_per_prompt=batch_size,
                generator=rng).images
            finish = time.time()
        elif current_pipe == "img2img":
            start = time.time()
            if loopback is True:
                try:
                    loopback_image
                except UnboundLocalError:
                    loopback_image = None

                if loopback_image is not None:
                    batch_images = pipe(
                        prompt,
                        negative_prompt=neg_prompt,
                        image=loopback_image,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        eta=eta,
                        strength=denoise_strength,
                        num_images_per_prompt=batch_size,
                        generator=rng,
                    ).images
                elif loopback_image is None:
                    batch_images = pipe(
                        prompt,
                        negative_prompt=neg_prompt,
                        image=init_image,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        eta=eta,
                        strength=denoise_strength,
                        num_images_per_prompt=batch_size,
                        generator=rng,
                    ).images
            elif loopback is False:
                batch_images = pipe(
                    prompt,
                    negative_prompt=neg_prompt,
                    image=init_image,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    strength=denoise_strength,
                    num_images_per_prompt=batch_size,
                    generator=rng,
                ).images
            finish = time.time()
        elif current_pipe == "inpaint":
            start = time.time()
            if legacy is True:
                batch_images = pipe(
                    prompt,
                    negative_prompt=neg_prompt,
                    image=init_image,
                    mask_image=init_mask,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_images_per_prompt=batch_size,
                    generator=rng,
                ).images
            else:
                batch_images = pipe(
                    prompt,
                    negative_prompt=neg_prompt,
                    image=init_image,
                    mask_image=init_mask,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_images_per_prompt=batch_size,
                    generator=rng,
                ).images
            finish = time.time()
        elif current_pipe == "controlnet":
            if preprocess:
                cnet_image=init_image
            else:
                if controlnet_type == "canny":
                    image = np.array(init_image)
                    low_threshold = 100
                    high_threshold = 200

                    image = cv2.Canny(image, low_threshold, high_threshold)
                    image = image[:, :, None]
                    image = np.concatenate([image, image, image], axis=2)
                    cnet_image = PIL.Image.fromarray(image)
                elif controlnet_type == "openpose":
                    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
                    cnet_image = openpose(init_image)
                    del openpose
                    gc.collect() 
                elif controlnet_type == "scribble":
                    hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
                    cnet_image = hed(init_image, scribble=True)
                    del hed
                    gc.collect()   
                elif controlnet_type == "depth":
                    depth_estimator = pipeline('depth-estimation')
                    image = depth_estimator(init_image)['depth']
                    image = np.array(image)
                    image = image[:, :, None]
                    image = np.concatenate([image, image, image], axis=2)
                    cnet_image = PIL.Image.fromarray(image)
                #cnet_image.save("./tmp.png")
            start = time.time()
            batch_images = pipe(
                prompt,
                negative_prompt=neg_prompt,
                image=cnet_image,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                eta=eta,
                num_images_per_prompt=batch_size,
                generator=rng,
                controlnet_conditioning_scale=conditioning_scale_t3,
            ).images
            finish = time.time()

        if current_pipe == "img2img" or "inpaint":
            steps = original_steps

        info = (
            f"{next_index + i:06} | "
            f"prompt: {prompt} "
            f"negative prompt: {neg_prompt} | "
            f"scheduler: {sched_name} "
            f"model: {model_name} "
            f"iteration size: {iteration_count} "
            f"batch size: {batch_size} "
            f"steps: {steps} "
            f"scale: {guidance_scale} "
            f"height: {height} "
            f"width: {width} "
            f"eta: {eta} "
            f"seed: {seeds[i]}"
        )
        info_png = (
            f"{prompt} "
            f"Negative prompt: {neg_prompt} "
            f"Steps: {steps}, "
            f"Sampler: {sched_name}, "
            f"CFG scale: {guidance_scale}, "
            f"Seed: {seeds[i]}, "
            f"Model: {model_name}, "
            f"Iteration size: {iteration_count}, "
            f"batch size: {batch_size}, "
            f"eta: {eta}, "
        )
        if current_pipe == "img2img":
            info = info + f" denoise: {denoise_strength}"
            info_png = info_png + f" denoise: {denoise_strength}"
        if current_pipe == controlnet:
            info = info + f"controlnet: {controlnet_name}"
            info_png = info_png + f"controlnet: {controlnet_name}"
        with open(os.path.join(output_path, "history.txt"), "a") as log:
            log.write(info + "\n")

        short_prompt = prompt.strip("<>:\"/\\|?*\n\t")
        short_prompt = re.sub(r'[\\/*?:"<>|\n\t]', "", short_prompt)
        short_prompt = short_prompt[:99] if len(short_prompt) > 100 else short_prompt
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("parameters",info_png)

        # save mask
        if savemask is True and current_pipe == "inpaint":
            saved_mask = PIL.ImageOps.invert(init_mask)
            saved_mask.save(
                os.path.join(
                    output_path + f"/masks/",
                    f"{next_index + i:06}-00.{short_prompt} mask.{image_format}",
                ),
                optimize=True,
                pnginfo=metadata,
            )

        if loopback is True:
            loopback_image = batch_images[0]

            # png output
            if image_format == "png":
                loopback_image.save(
                    os.path.join(
                        output_path,
                        f"{next_index + i:06}-00.{short_prompt}.{image_format}",
                    ),
                    optimize=True,
                    pnginfo=metadata,
                )
            # jpg output
            elif image_format == "jpg":
                loopback_image.save(
                    os.path.join(
                        output_path,
                        f"{next_index + i:06}-00.{short_prompt}.{image_format}",
                    ),
                    quality=95,
                    subsampling=0,
                    optimize=True,
                    progressive=True,
                )
        elif loopback is False:
            # png output
            if image_format == "png":
                for j in range(batch_size):
                    batch_images[j].save(
                        os.path.join(
                            output_path,
                            f"{next_index + i:06}-{j:02}.{short_prompt}.{image_format}",
                        ),
                        optimize=True,
                        pnginfo=metadata,
                    )
            # jpg output
            elif image_format == "jpg":
                for j in range(batch_size):
                    batch_images[j].save(
                        os.path.join(
                            output_path,
                            f"{next_index + i:06}-{j:02}.{short_prompt}.{image_format}",
                        ),
                        quality=95,
                        subsampling=0,
                        optimize=True,
                        progressive=True,
                    )

        images.extend(batch_images)
        time_taken = time_taken + (finish - start)

    time_taken = time_taken / 60.0
    if iteration_count > 1:
        status = (
            f"Run indexes {next_index:06} "
            f"to {next_index + iteration_count - 1:06} "
            f"took {time_taken:.1f} minutes "
            f"to generate {iteration_count} "
            f"iterations with batch size of {batch_size}. "
            f"seeds: " + np.array2string(seeds, separator=",")
        )
    else:
        status = (
            f"Run index {next_index:06} "
            f"took {time_taken:.1f} minutes "
            f"to generate a batch size of {batch_size}. "
            f"seed: {seeds[0]}"
        )

    return images, status


def resize_and_crop(input_image: PIL.Image.Image, height: int, width: int):
    input_width, input_height = input_image.size

    # nearest neighbor for upscaling
    if (input_width * input_height) < (width * height):
        resample_type = Image.NEAREST
    # lanczos for downscaling
    else:
        resample_type = Image.LANCZOS

    if height / width > input_height / input_width:
        adjust_width = int(input_width * height / input_height)
        input_image = input_image.resize((adjust_width, height),
                                         resample=resample_type)
        left = (adjust_width - width) // 2
        right = left + width
        input_image = input_image.crop((left, 0, right, height))
    else:
        adjust_height = int(input_height * width / input_width)
        input_image = input_image.resize((width, adjust_height),
                                         resample=resample_type)
        top = (adjust_height - height) // 2
        bottom = top + height
        input_image = input_image.crop((0, top, width, bottom))
    return input_image


def step_adjustment(unadjusted_steps, denoise, pipeline):
    # adjust step count to account for denoise in img2img
    if pipeline == "img2img":
        steps_old = unadjusted_steps
        steps = ceil(unadjusted_steps / denoise)
        if (steps > 1000) and (sch_t1 == "DPMSM" or "DPMSS" or "DEIS"):
            steps_unreduced = steps
            steps = 1000
            print()
            print(
                f"Adjusting steps to account for denoise. From {steps_old} "
                f"to {steps_unreduced} steps internally."
            )
            print(
                f"Without adjustment the actual step count would be "
                f"~{ceil(steps_old * denoise)} steps."
            )
            print()
            print(
                f"INTERNAL STEP COUNT EXCEEDS 1000 MAX FOR DPMSM, DPMSS, "
                f"or DEIS. INTERNAL STEPS WILL BE REDUCED TO 1000."
            )
            print()
        else:
            print()
            print(
                f"Adjusting steps to account for denoise. From {steps_old} "
                f"to {steps} steps internally."
            )
            print(
                f"Without adjustment the actual step count would be "
                f"~{ceil(steps_old * denoise)} steps."
            )
            print()
    elif pipeline == "inpaint":
        # adjust steps to account for legacy inpaint only using ~80% of set steps
        steps_old = unadjusted_steps
        if unadjusted_steps < 5:
            steps = unadjusted_steps + 1
        elif unadjusted_steps >= 5:
            steps = int((unadjusted_steps / 0.7989) + 1)
        print()
        print(
            f"Adjusting steps for legacy inpaint. From {steps_old} "
            f"to {steps} internally."
        )
        print(
            f"Without adjustment the actual step count would be "
            f"~{int(steps_old * 0.8)} steps."
        )
        print()

    return steps


def clear_click():
    global current_tab
    if current_tab == 0:
        return {
            prompt_t0: "",
            neg_prompt_t0: "",
            sch_t0: "PNDM",
            iter_t0: 1,
            batch_t0: 1,
            steps_t0: 16,
            guid_t0: 7.5,
            height_t0: 512,
            width_t0: 512,
            eta_t0: 0.0,
            seed_t0: "",
            fmt_t0: "png",
        }
    elif current_tab == 1:
        return {
            prompt_t1: "",
            neg_prompt_t1: "",
            sch_t1: "PNDM",
            image_t1: None,
            iter_t1: 1,
            batch_t1: 1,
            steps_t1: 16,
            guid_t1: 7.5,
            height_t1: 512,
            width_t1: 512,
            eta_t1: 0.0,
            denoise_t1: 0.8,
            seed_t1: "",
            fmt_t1: "png",
            loopback_t1: False,
        }
    elif current_tab == 2:
        return {
            prompt_t2: "",
            neg_prompt_t2: "",
            sch_t2: "PNDM",
            legacy_t2: False,
            savemask_t2: False,
            image_t2: None,
            mask_t2: None,
            iter_t2: 1,
            batch_t2: 1,
            steps_t2: 16,
            guid_t2: 7.5,
            height_t2: 512,
            width_t2: 512,
            eta_t2: 0.0,
            seed_t2: "",
            fmt_t2: "png",
        }
    elif current_tab == 3:
        return {
            prompt_t3: "",
            neg_prompt_t3: "",
            sch_t3: "PNDM",
            preprocess_t3: False,
            image_t3: None, 
            iter_t3: 1,
            batch_t3: 1,
            steps_t3: 16,
            guid_t3: 7.5,
            height_t3: 512,
            width_t3: 512,
            eta_t3: 0.0,
            seed_t3: "",
            fmt_t3: "png",
        }


def generate_click(
    model_drop,
    controlnet_drop,
    prompt_t0,
    neg_prompt_t0,
    sch_t0,
    iter_t0,
    batch_t0,
    steps_t0,
    guid_t0,
    height_t0,
    width_t0,
    eta_t0,
    seed_t0,
    fmt_t0,
    prompt_t1,
    neg_prompt_t1,
    image_t1,
    sch_t1,
    iter_t1,
    batch_t1,
    steps_t1,
    guid_t1,
    height_t1,
    width_t1,
    eta_t1,
    denoise_t1,
    seed_t1,
    fmt_t1,
    loopback_t1,
    prompt_t2,
    neg_prompt_t2,
    sch_t2,
    legacy_t2,
    savemask_t2,
    image_t2,
    mask_t2,
    iter_t2,
    batch_t2,
    steps_t2,
    guid_t2,
    height_t2,
    width_t2,
    eta_t2,
    seed_t2,
    fmt_t2,
    prompt_t3,
    neg_prompt_t3,
    image_t3,
    sch_t3,
    preprocess_t3,
    conditioning_scale_t3,
    iter_t3,
    batch_t3,
    steps_t3,
    guid_t3,
    height_t3,
    width_t3,
    eta_t3,
    seed_t3,
    fmt_t3,
):
    global model_name
    global controlnet_name
    global provider
    global current_tab
    global current_pipe
    global current_legacy
    global release_memory_after_generation
    global release_memory_on_change
    global scheduler
    global controlnet_type
    global pipe
    global controlnet
    global original_steps

    # reset scheduler and pipeline if model is different
    if model_name != model_drop:
        model_name = model_drop
        scheduler = None
        pipe = None
        gc.collect()
    model_path = os.path.join("model", model_name)
    
    if controlnet_name != controlnet_drop:
        controlnet_name = controlnet_drop
        controlnet = None
        gc.collect()
    if controlnet_name != "default":
        controlnet_path = os.path.join("controlnet", controlnet_name)
        if "canny" in controlnet_name:
            controlnet_type = "canny"
        elif "openpose" in controlnet_name:
            controlnet_type = "openpose"
        elif "scribble" in controlnet_name:
            controlnet_type = "scribble"
        elif "depth" in controlnet_name:
            controlnet_type = "depth"
    else:
        if "canny" in model_name:
            controlnet_type = "canny"
        elif "openpose" in model_name:
            controlnet_type = "openpose"
        elif "scribble" in model_name:
            controlnet_type = "scribble"
        elif "depth" in model_name:
            controlnet_type = "depth"

    # select which scheduler depending on current tab
    if current_tab == 0:
        sched_name = sch_t0
    elif current_tab == 1:
        sched_name = sch_t1
    elif current_tab == 2:
        sched_name = sch_t2
    elif current_tab == 3:
        sched_name = sch_t3
    else:
        raise Exception("Unknown tab")

    if sched_name == "PNDM" and type(scheduler) is not PNDMScheduler:
        scheduler = PNDMScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "LMS" and type(scheduler) is not LMSDiscreteScheduler:
        scheduler = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "DDIM" and type(scheduler) is not DDIMScheduler:
        scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "Euler" and type(scheduler) is not EulerDiscreteScheduler:
        scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "EulerA" and type(scheduler) is not EulerAncestralDiscreteScheduler:
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "DPMS_ms" and type(scheduler) is not DPMSolverMultistepScheduler:
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "DPMS_ss" and type(scheduler) is not DPMSolverSinglestepScheduler:
        scheduler = DPMSolverSinglestepScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "DEIS" and type(scheduler) is not DEISMultistepScheduler:
        scheduler = DEISMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "HEUN" and type(scheduler) is not HeunDiscreteScheduler:
        scheduler = HeunDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "KDPM2" and type(scheduler) is not KDPM2DiscreteScheduler:
        scheduler = KDPM2DiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif sched_name == "UniPC" and type(scheduler) is not UniPCMultistepScheduler:
        scheduler = UniPCMultistepScheduler.from_pretrained(model_path, subfolder="scheduler") 

    # select which pipeline depending on current tab
    if current_tab == 0:
        controlnet = None
        gc.collect()
        if current_pipe == ("img2img" or "inpaint") and release_memory_on_change:
            pipe = None
            gc.collect()
        if current_pipe != "txt2img" or pipe is None:
            if textenc_on_cpu and vaedec_on_cpu:
                cputextenc = OnnxRuntimeModel.from_pretrained(
                    model_path + "/text_encoder")
                cpuvaedec = OnnxRuntimeModel.from_pretrained(
                    model_path + "/vae_decoder")
                pipe = OnnxStableDiffusionPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    text_encoder=cputextenc,
                    vae_decoder=cpuvaedec,
                    vae_encoder=None
                )
            elif textenc_on_cpu:
                cputextenc = OnnxRuntimeModel.from_pretrained(
                    model_path + "/text_encoder")
                pipe = OnnxStableDiffusionPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    text_encoder=cputextenc)
            elif vaedec_on_cpu:
                cpuvaedec = OnnxRuntimeModel.from_pretrained(
                    model_path + "/vae_decoder")
                pipe = OnnxStableDiffusionPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    vae_decoder=cpuvaedec,
                    vae_encoder=None
                )
            else:
                pipe = OnnxStableDiffusionPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler)
        current_pipe = "txt2img"
    elif current_tab == 1:
        controlnet = None
        gc.collect()
        if current_pipe == ("txt2img" or "inpaint") and release_memory_on_change:
            pipe = None
            gc.collect()
        if current_pipe != "img2img" or pipe is None:
            if textenc_on_cpu and vaedec_on_cpu:
                cputextenc = OnnxRuntimeModel.from_pretrained(
                    model_path + "/text_encoder")
                cpuvaedec = OnnxRuntimeModel.from_pretrained(
                    model_path + "/vae_decoder")
                pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    text_encoder=cputextenc,
                    vae_decoder=cpuvaedec)
            elif textenc_on_cpu:
                cputextenc = OnnxRuntimeModel.from_pretrained(
                    model_path + "/text_encoder")
                pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    text_encoder=cputextenc)
            elif vaedec_on_cpu:
                cpuvaedec = OnnxRuntimeModel.from_pretrained(
                    model_path + "/vae_decoder")
                pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler,
                    vae_decoder=cpuvaedec)
            else:
                pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=scheduler)
        current_pipe = "img2img"
    elif current_tab == 2:
        controlnet = None
        gc.collect()
        if current_pipe == ("txt2img" or "img2img") and release_memory_on_change:
            pipe = None
            gc.collect()
        if current_pipe != "inpaint" or pipe is None or current_legacy != legacy_t2:
            if legacy_t2:
                if textenc_on_cpu and vaedec_on_cpu:
                    cputextenc = OnnxRuntimeModel.from_pretrained(
                        model_path + "/text_encoder")
                    cpuvaedec = OnnxRuntimeModel.from_pretrained(
                        model_path + "/vae_decoder")
                    pipe = OnnxStableDiffusionInpaintPipelineLegacy.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        text_encoder=cputextenc,
                        vae_decoder=cpuvaedec)
                elif textenc_on_cpu:
                    cputextenc = OnnxRuntimeModel.from_pretrained(
                        model_path + "/text_encoder")
                    pipe = OnnxStableDiffusionInpaintPipelineLegacy.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        text_encoder=cputextenc)
                elif vaedec_on_cpu:
                    cpuvaedec = OnnxRuntimeModel.from_pretrained(
                        model_path + "/vae_decoder")
                    pipe = OnnxStableDiffusionInpaintPipelineLegacy.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        vae_decoder=cpuvaedec)
                else:
                    pipe = OnnxStableDiffusionInpaintPipelineLegacy.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler)
            else:
                if textenc_on_cpu and vaedec_on_cpu:
                    cputextenc = OnnxRuntimeModel.from_pretrained(
                        model_path + "/text_encoder")
                    cpuvaedec = OnnxRuntimeModel.from_pretrained(
                        model_path + "/vae_decoder")
                    pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        text_encoder=cputextenc,
                        vae_decoder=cpuvaedec)
                elif textenc_on_cpu:
                    cputextenc = OnnxRuntimeModel.from_pretrained(
                        model_path + "/text_encoder")
                    pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        text_encoder=cputextenc)
                elif vaedec_on_cpu:
                    cpuvaedec = OnnxRuntimeModel.from_pretrained(
                        model_path + "/vae_decoder")
                    pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        vae_decoder=cpuvaedec)
                else:
                    pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler)
        current_pipe = "inpaint"
        current_legacy = legacy_t2
    elif current_tab == 3:
        disable_controlnet = False
        if os.path.exists("./modules/pipeline_onnx_stable_diffusion_controlnet.py"):
            if current_pipe != "controlnet" and release_memory_on_change:
                pipe = None
                gc.collect()
            if current_pipe != "controlnet" or pipe is None:
                if controlnet == None:
                    if controlnet_name != "default":
                        controlnet = OnnxRuntimeModel.from_pretrained(
                            controlnet_path + "/controlnet", provider=provider)
                    else:
                        controlnet = OnnxRuntimeModel.from_pretrained(
                            model_path + '/controlnet', provider=provider)
                if textenc_on_cpu and vaedec_on_cpu:
                    cputextenc = OnnxRuntimeModel.from_pretrained(
                        model_path + "/text_encoder")
                    cpuvaedec = OnnxRuntimeModel.from_pretrained(
                        model_path + "/vae_decoder")
                    pipe = OnnxStableDiffusionControlNetPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        text_encoder=cputextenc,
                        vae_decoder=cpuvaedec,
                        controlnet=controlnet)
                elif textenc_on_cpu:
                    cputextenc = OnnxRuntimeModel.from_pretrained(
                        model_path + "/text_encoder")
                    pipe = OnnxStableDiffusionControlNetPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        text_encoder=cputextenc,
                        controlnet=controlnet)
                elif vaedec_on_cpu:
                    cpuvaedec = OnnxRuntimeModel.from_pretrained(
                        model_path + "/vae_decoder")
                    pipe = OnnxStableDiffusionControlNetPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        vae_decoder=cpuvaedec,
                        controlnet=controlnet)
                else:
                    pipe = OnnxStableDiffusionControlNetPipeline.from_pretrained(
                        model_path,
                        provider=provider,
                        scheduler=scheduler,
                        controlnet=controlnet)
            else:
                if controlnet == None:
                    pipe.controlnet = None
                    gc.collect()
                    if controlnet_name != "default":
                        controlnet = OnnxRuntimeModel.from_pretrained(
                            controlnet_path + "/controlnet", provider=provider)
                    else:
                        controlnet = OnnxRuntimeModel.from_pretrained(
                            model_path + '/controlnet', provider=provider)
                    pipe.controlnet = controlnet
        else:
            disable_controlnet = True
        current_pipe = "controlnet"

    # manual garbage collection
    gc.collect()

    # modifying the methods in the pipeline object
    if type(pipe.scheduler) is not type(scheduler):
        pipe.scheduler = scheduler
    if version.parse(_df_version) >= version.parse("0.8.0"):
        safety_checker = None
    else:
        safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    pipe.safety_checker = safety_checker
    pipe._encode_prompt = functools.partial(lpw_pipe._encode_prompt, pipe)

    # run the pipeline with the correct parameters
    if current_tab == 0:
        original_steps = steps_t0
        
        images, status = run_diffusers(
            prompt_t0,
            neg_prompt_t0,
            None,
            None,
            iter_t0,
            batch_t0,
            steps_t0,
            guid_t0,
            height_t0,
            width_t0,
            eta_t0,
            0,
            seed_t0,
            fmt_t0,
            False,
            False,
            False,
            False,
        )
    elif current_tab == 1:
        # input image resizing
        input_image = image_t1.convert("RGB")
        input_image = resize_and_crop(input_image, height_t1, width_t1)

        # adjust steps to account for denoise
        original_steps = steps_t1
        steps_t1 = step_adjustment(steps_t1, denoise_t1, "img2img")

        images, status = run_diffusers(
            prompt_t1,
            neg_prompt_t1,
            input_image,
            None,
            iter_t1,
            batch_t1,
            steps_t1,
            guid_t1,
            height_t1,
            width_t1,
            eta_t1,
            denoise_t1,
            seed_t1,
            fmt_t1,
            False,
            False,
            loopback_t1,
            False,
        )
    elif current_tab == 2:
        input_image = image_t2["image"].convert("RGB")
        input_image = resize_and_crop(input_image, height_t2, width_t2)
        
        original_steps = steps_t2

        if mask_t2 is not None:
            print("using uploaded mask")
            input_mask = mask_t2.convert("RGB")
            input_mask = resize_and_crop(input_mask, height_t2, width_t2)
        else:
            print("using painted mask")
            input_mask = image_t2["mask"].convert("RGB")
            input_mask = resize_and_crop(input_mask, height_t2, width_t2)

        # adjust steps to account for legacy inpaint only using ~80% of set steps
        if legacy_t2 is True:
            steps_t2 = step_adjustment(steps_t2, 0, "inpaint")

        images, status = run_diffusers(
            prompt_t2,
            neg_prompt_t2,
            input_image,
            input_mask,
            iter_t2,
            batch_t2,
            steps_t2,
            guid_t2,
            height_t2,
            width_t2,
            eta_t2,
            0,
            seed_t2,
            fmt_t2,
            legacy_t2,
            savemask_t2,
            False,
            False,
        )
    elif current_tab == 3:
        if disable_controlnet == False:
            input_image = image_t3.convert("RGB")
            input_image = resize_and_crop(input_image, height_t3, width_t3)
            original_steps = steps_t3
            
            images, status = run_diffusers(
                prompt_t3,
                neg_prompt_t3,
                input_image,
                None,
                iter_t3,
                batch_t3,
                steps_t3,
                guid_t3,
                height_t3,
                width_t3,
                eta_t3,
                conditioning_scale_t3,
                seed_t3,
                fmt_t3,
                False,
                False,
                False,
                preprocess_t3,
            )


    if release_memory_after_generation:
        pipe = None
        gc.collect()

    return images, status


def select_tab0():
    global current_tab
    current_tab = 0


def select_tab1():
    global current_tab
    current_tab = 1


def select_tab2():
    global current_tab
    current_tab = 2
    
def select_tab3():
    global current_tab
    current_tab = 3


def choose_sch(sched_name: str):
    if sched_name == "DDIM":
        return gr.update(interactive=True)
    else:
        return gr.update(interactive=False)


def clear_temp_files(sig, frame):
    print(f"Cleaning temporary files...", flush=True)
    shutil.rmtree(tempfile.gettempdir(), ignore_errors=True, onerror=None)
    exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gradio interface for ONNX based Stable Diffusion")
    parser.add_argument("--cpu-only", action="store_true", default=False, help="run ONNX with CPU")
    parser.add_argument(
        "--cpu-textenc", action="store_true", default=False,
        help="Run Text Encoder on CPU, saves VRAM by running Text Encoder on CPU")
    parser.add_argument(
        "--cpu-vaedec", action="store_true", default=False,
        help="Run VAE Decoder on CPU, saves VRAM by running VAE Decoder on CPU")
    parser.add_argument(
        "--release-memory-after-generation", action="store_true", default=False,
        help="de-allocate the pipeline and release memory after generation")
    parser.add_argument(
        "--release-memory-on-change", action="store_true", default=False,
        help="de-allocate the pipeline and release memory allocated when changing pipelines",
    )
    args = parser.parse_args()

    # variables for ONNX pipelines
    model_name = None
    controlnet_name = None
    provider = "CPUExecutionProvider" if args.cpu_only else "DmlExecutionProvider"
    current_tab = 0
    current_pipe = "txt2img"
    current_legacy = False
    release_memory_after_generation = args.release_memory_after_generation
    release_memory_on_change = args.release_memory_on_change
    textenc_on_cpu = args.cpu_textenc
    vaedec_on_cpu = args.cpu_vaedec

    # diffusers objects
    scheduler = None
    pipe = None
    controlnet = None

    # check versions
    is_v_0_12 = version.parse(_df_version) >= version.parse("0.12.0")
    is_v_0_14 = version.parse(_df_version) >= version.parse("0.14.0")
    is_v_dev = version.parse(_df_version).is_prerelease

    # prerelease version use warning
    if is_v_dev:
        print(
            "You are using diffusers " + str(version.parse(_df_version)) + " (prerelease)\n" +
            "If you experience unexpected errors please run `pip install diffusers --force-reinstall`.")

    # custom css
    custom_css = """
    #gen_button {height: 90px}
    #image_init {min-height: 400px}
    #image_init [data-testid="image"], #image_init [data-testid="image"] > div {min-height: 400px}
    #image_inpaint {min-height: 400px}
    #image_inpaint [data-testid="image"], #image_inpaint [data-testid="image"] > div {min-height: 400px}
    #image_inpaint .touch-none {display: flex}
    #image_inpaint img {display: block; max-width: 84%}
    #image_inpaint canvas {max-width: 84%; object-fit: contain}
    """

    # search the model folder
    model_dir = "model"
    model_list = []
    with os.scandir(model_dir) as scan_it:
        for entry in scan_it:
            if entry.is_dir():
                model_list.append(entry.name)
    default_model = model_list[0] if len(model_list) > 0 else None
    
    controlnet_visible = False
    controlnet_list = ['default']
    default_controlmodel = "default"

    if is_v_0_12:
        from diffusers import (
            DPMSolverSinglestepScheduler,
            DEISMultistepScheduler,
            HeunDiscreteScheduler,
            KDPM2DiscreteScheduler
        )
        
        if is_v_0_14:
            if os.path.exists("./modules/pipeline_onnx_stable_diffusion_controlnet.py"):
                controlnet_visible = True
                if not os.path.isdir("./controlnet"):
                    os.mkdir("./controlnet")
                controlnet_dir = "controlnet"
                with os.scandir(controlnet_dir) as scan_it:
                    for entry in scan_it:
                        if entry.is_dir():
                            controlnet_list.append(entry.name)
                default_controlmodel = controlnet_list[0] if len(controlnet_list) > 0 else None
                from modules.pipeline_onnx_stable_diffusion_controlnet import OnnxStableDiffusionControlNetPipeline
                from controlnet_aux import OpenposeDetector, HEDdetector
                from transformers import pipeline
            from diffusers import UniPCMultistepScheduler
            sched_list = ["DPMS_ms", "DPMS_ss", "EulerA", "Euler", "DDIM", "LMS", "PNDM", "DEIS", "HEUN", "KDPM2", "UniPC"]
        else:
            sched_list = ["DPMS_ms", "DPMS_ss", "EulerA", "Euler", "DDIM", "LMS", "PNDM", "DEIS", "HEUN", "KDPM2"]
    else:
        sched_list = ["DPMS_ms", "EulerA", "Euler", "DDIM", "LMS", "PNDM"]

    # create gradio block
    title = "Stable Diffusion ONNX"
    with gr.Blocks(title=title, css=custom_css) as demo:
        with gr.Row():
            with gr.Column(scale=13, min_width=650):
                model_drop = gr.Dropdown(model_list, value=default_model, label="model folder", interactive=True)
                controlnet_drop = gr.Dropdown(controlnet_list, value=default_controlmodel, 
                    label="controlnet folder", interactive=True, visible=controlnet_visible)
            with gr.Column(scale=11, min_width=550):
                with gr.Row():
                    gen_btn = gr.Button("Generate", variant="primary", elem_id="gen_button")
                    clear_btn = gr.Button("Clear", elem_id="gen_button")
        with gr.Row():
            with gr.Column(scale=13, min_width=650):
                with gr.Tab(label="txt2img") as tab0:
                    prompt_t0 = gr.Textbox(value="", lines=2, label="prompt")
                    neg_prompt_t0 = gr.Textbox(value="", lines=2, label="negative prompt")
                    sch_t0 = gr.Radio(sched_list, value="PNDM", label="scheduler")
                    with gr.Row():
                        iter_t0 = gr.Slider(1, 24, value=1, step=1, label="iteration count")
                        batch_t0 = gr.Slider(1, 4, value=1, step=1, label="batch size")
                    steps_t0 = gr.Slider(1, 300, value=16, step=1, label="steps")
                    guid_t0 = gr.Slider(0, 50, value=7.5, step=0.1, label="guidance")
                    height_t0 = gr.Slider(256, 2048, value=512, step=64, label="height")
                    width_t0 = gr.Slider(256, 2048, value=512, step=64, label="width")
                    eta_t0 = gr.Slider(0, 1, value=0.0, step=0.01, label="DDIM eta", interactive=False)
                    seed_t0 = gr.Textbox(value="", max_lines=1, label="seed")
                    fmt_t0 = gr.Radio(["png", "jpg"], value="png", label="image format")
                with gr.Tab(label="img2img") as tab1:
                    prompt_t1 = gr.Textbox(value="", lines=2, label="prompt")
                    neg_prompt_t1 = gr.Textbox(value="", lines=2, label="negative prompt")
                    sch_t1 = gr.Radio(sched_list, value="PNDM", label="scheduler")
                    image_t1 = gr.Image(label="input image", type="pil", elem_id="image_init")
                    with gr.Row():
                        iter_t1 = gr.Slider(1, 24, value=1, step=1, label="iteration count")
                        batch_t1 = gr.Slider(1, 4, value=1, step=1, label="batch size")
                    with gr.Row():
                        loopback_t1 = gr.Checkbox(value=False, label="loopback (use iteration count)")
                    steps_t1 = gr.Slider(1, 300, value=16, step=1, label="steps")
                    guid_t1 = gr.Slider(0, 50, value=7.5, step=0.1, label="guidance")
                    height_t1 = gr.Slider(256, 2048, value=512, step=64, label="height")
                    width_t1 = gr.Slider(256, 2048, value=512, step=64, label="width")
                    eta_t1 = gr.Slider(0, 1, value=0.0, step=0.01, label="DDIM eta", interactive=False)
                    denoise_t1 = gr.Slider(0, 1, value=0.8, step=0.01, label="denoise strength")
                    seed_t1 = gr.Textbox(value="", max_lines=1, label="seed")
                    fmt_t1 = gr.Radio(["png", "jpg"], value="png", label="image format")
                with gr.Tab(label="inpainting") as tab2:
                    prompt_t2 = gr.Textbox(value="", lines=2, label="prompt")
                    neg_prompt_t2 = gr.Textbox(value="", lines=2, label="negative prompt")
                    sch_t2 = gr.Radio(sched_list, value="PNDM", label="scheduler")
                    legacy_t2 = gr.Checkbox(value=False, label="legacy inpaint")
                    savemask_t2 = gr.Checkbox(
                        value=False, label="save painted mask"
                    )
                    image_t2 = gr.Image(
                        source="upload", tool="sketch", label="input image", type="pil", elem_id="image_inpaint")
                    mask_t2 = gr.Image(
                        source="upload",
                        label="input mask",
                        type="pil",
                        invert_colors=True,
                        elem_id="mask_inpaint",
                    )
                    with gr.Row():
                        iter_t2 = gr.Slider(1, 24, value=1, step=1, label="iteration count")
                        batch_t2 = gr.Slider(1, 4, value=1, step=1, label="batch size")
                    steps_t2 = gr.Slider(1, 300, value=16, step=1, label="steps")
                    guid_t2 = gr.Slider(0, 50, value=7.5, step=0.1, label="guidance")
                    height_t2 = gr.Slider(256, 2048, value=512, step=64, label="height")
                    width_t2 = gr.Slider(256, 2048, value=512, step=64, label="width")
                    eta_t2 = gr.Slider(0, 1, value=0.0, step=0.01, label="DDIM eta", interactive=False)
                    seed_t2 = gr.Textbox(value="", max_lines=1, label="seed")
                    fmt_t2 = gr.Radio(["png", "jpg"], value="png", label="image format")
                with gr.Tab(label="controlnet") as tab3:
                    prompt_t3 = gr.Textbox(value="", lines=2, label="prompt")
                    neg_prompt_t3 = gr.Textbox(value="", lines=2, label="negative prompt")
                    sch_t3 = gr.Radio(sched_list, value="PNDM", label="scheduler")
                    preprocess_t3 = gr.Checkbox(value=False, label="Don't preprocess image")
                    image_t3 = gr.Image(label="input image", type="pil", elem_id="image_init")
                    with gr.Row():
                        iter_t3 = gr.Slider(1, 24, value=1, step=1, label="iteration count")
                        batch_t3 = gr.Slider(1, 1, value=1, step=1, label="batch size")
                    steps_t3 = gr.Slider(1, 300, value=16, step=1, label="steps")
                    guid_t3 = gr.Slider(0, 50, value=7.5, step=0.1, label="guidance")
                    height_t3 = gr.Slider(192, 1536, value=512, step=64, label="height")
                    width_t3 = gr.Slider(192, 1536, value=512, step=64, label="width")
                    eta_t3 = gr.Slider(0, 1, value=0.0, step=0.01, label="DDIM eta", interactive=False)
                    conditioning_scale_t3 = gr.Slider(0, 2, value=1.0, step=0.01, label="controlnet conditioning scale")
                    seed_t3 = gr.Textbox(value="", max_lines=1, label="seed")
                    fmt_t3 = gr.Radio(["png", "jpg"], value="png", label="image format")
            with gr.Column(scale=11, min_width=550):
                image_out = gr.Gallery(value=None, label="output images")
                status_out = gr.Textbox(value="", label="status")

        # config components
        tab0_inputs = [
            prompt_t0,
            neg_prompt_t0,
            sch_t0,
            iter_t0,
            batch_t0,
            steps_t0,
            guid_t0,
            height_t0,
            width_t0,
            eta_t0,
            seed_t0,
            fmt_t0,
        ]
        tab1_inputs = [
            prompt_t1,
            neg_prompt_t1,
            image_t1,
            sch_t1,
            iter_t1,
            batch_t1,
            steps_t1,
            guid_t1,
            height_t1,
            width_t1,
            eta_t1,
            denoise_t1,
            seed_t1,
            fmt_t1,
            loopback_t1,
        ]
        tab2_inputs = [
            prompt_t2,
            neg_prompt_t2,
            sch_t2,
            legacy_t2,
            savemask_t2,
            image_t2,
            mask_t2,
            iter_t2,
            batch_t2,
            steps_t2,
            guid_t2,
            height_t2,
            width_t2,
            eta_t2,
            seed_t2,
            fmt_t2,
        ]
        tab3_inputs = [
            prompt_t3,
            neg_prompt_t3,
            image_t3,
            sch_t3,
            preprocess_t3,
            conditioning_scale_t3,
            iter_t3,
            batch_t3,
            steps_t3,
            guid_t3,
            height_t3,
            width_t3,
            eta_t3,
            seed_t3,
            fmt_t3,
        ]
        all_inputs = [model_drop]
        all_inputs.extend([controlnet_drop])
        all_inputs.extend(tab0_inputs)
        all_inputs.extend(tab1_inputs)
        all_inputs.extend(tab2_inputs)
        all_inputs.extend(tab3_inputs)

        clear_btn.click(fn=clear_click, inputs=None, outputs=all_inputs, queue=False)
        gen_btn.click(fn=generate_click, inputs=all_inputs, outputs=[image_out, status_out])

        tab0.select(fn=select_tab0, inputs=None, outputs=None)
        tab1.select(fn=select_tab1, inputs=None, outputs=None)
        tab2.select(fn=select_tab2, inputs=None, outputs=None)
        tab3.select(fn=select_tab3, inputs=None, outputs=None)

        sch_t0.change(fn=choose_sch, inputs=sch_t0, outputs=eta_t0, queue=False)
        sch_t1.change(fn=choose_sch, inputs=sch_t1, outputs=eta_t1, queue=False)
        sch_t2.change(fn=choose_sch, inputs=sch_t2, outputs=eta_t2, queue=False)
        sch_t3.change(fn=choose_sch, inputs=sch_t3, outputs=eta_t3, queue=False)

        image_out.style(grid=2)
        image_t1.style(height=402)
        image_t2.style(height=402)
        image_t3.style(height=402)

    # change the default temp folder and handle cleaning it when stopping the ui
    os.makedirs("temp", exist_ok=True)
    tempfile.tempdir = os.path.abspath(os.path.join("temp"))
    signal.signal(signal.SIGINT, clear_temp_files)

    # start gradio web interface on local host
    demo.launch()

    # use the following to launch the web interface to a private network
    # demo.queue(concurrency_count=1)
    # demo.launch(server_name="0.0.0.0")
