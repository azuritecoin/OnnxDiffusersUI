import argparse
import os
import re
import time

from diffusers import OnnxStableDiffusionPipeline
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler
)
import numpy as np


def get_latents_from_seed(seed: int, batch_size: int, height: int, width: int) -> np.ndarray:
    latents_shape = (batch_size, 4, height // 8, width // 8)
    # Gotta use numpy instead of torch, because torch's randn() doesn't support DML
    rng = np.random.default_rng(seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents



parser = argparse.ArgumentParser(description="simple interface for ONNX based Stable Diffusion")
parser.add_argument(
    "--model", dest="model_path", default="model/stable_diffusion_onnx", help="path to the model directory")
parser.add_argument(
    "--prompt", dest="prompt", default="a photo of an astronaut riding a horse on mars",
    help="input text prompt to generate image")
parser.add_argument(
    "--guidance-scale", type=float, dest="guidance_scale", default=7.5, help="guidance value for the generator")
parser.add_argument("--steps", dest="steps", type=int, default=25, help="number of steps for the generator")
parser.add_argument("--height", dest="height", type=int, default=384, help="height of the image")
parser.add_argument("--width", dest="width", type=int, default=384, help="width of the image")
parser.add_argument("--seed", dest="seed", default="", help="seed for the generator")
parser.add_argument("--cpu-only", action="store_true", default=False, help="run ONNX with CPU")
args = parser.parse_args()
pndm = PNDMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
lms = LMSDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")
ddim = DDIMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
ddpm = DDPMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
euler = EulerDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")
eulera = EulerAncestralDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")
dpms = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder="scheduler")
parser.add_argument("--scheduler", dest="scheduler", default=pndm, help="schedulers: pndm, lms, ddim, ddpm, euler, eulera, dpms")
args = parser.parse_args()


provider = "CPUExecutionProvider" if args.cpu_only else "DmlExecutionProvider"
pipe = OnnxStableDiffusionPipeline.from_pretrained(
    args.model_path, provider=provider, scheduler=args.scheduler, safety_checker=None)

# generate seeds for iterations
if args.seed == "":
    rng = np.random.default_rng()
    seed = rng.integers(np.iinfo(np.uint32).max)
else:
    try:
        seed = int(args.seed) & np.iinfo(np.uint32).max
    except ValueError:
        seed = hash(args.seed) & np.iinfo(np.uint32).max

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

sched_name = str(pipe.scheduler._class_name)
info = f"{next_index:06} | prompt: {args.prompt} | scheduler: {sched_name} model: {args.model_path} steps: " + \
       f"{args.steps} scale: {args.guidance_scale} height: {args.height} width: {args.width} seed: {seed}\n"
with open(os.path.join(output_path, "history.txt"), "a") as log:
    log.write(info)

# Generate our own latents so that we can provide a seed.
latents = get_latents_from_seed(seed, 1, args.height, args.width)

start = time.time()
images = pipe(
    args.prompt, height=args.height, width=args.width, num_inference_steps=args.steps,
    guidance_scale=args.guidance_scale, latents=latents).images
finish = time.time()

images[0].save(os.path.join(output_path, f"{next_index:06}-00.png"))

time_taken = (finish - start) / 60.0
status = f"Run index {next_index:06} took {time_taken:.1f} minutes to generate an image. seed: {seed}"
print(status)