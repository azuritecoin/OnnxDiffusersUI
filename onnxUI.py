import argparse
import os
import re
import time

from diffusers import StableDiffusionOnnxPipeline, DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
import diffusers
import gradio as gr
import numpy as np


def get_latents_from_seed(seed: int, batch_size: int, height: int, width: int) -> np.ndarray:
    latents_shape = (batch_size, 4, height // 8, width // 8)
    # Gotta use numpy instead of torch, because torch's randn() doesn't support DML
    rng = np.random.default_rng(seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents


# gradio function
def generate(
    prompt: str,
    neg_prompt: str,
    iteration_size: int,
    batch_size: int,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    eta: float,
    seed: str,
    image_format: str
) -> (list, str):
    global pipe
    global model_path
    sched_name = str(pipe.scheduler._class_name)

    # generate seeds for iterations
    if seed == "":
        rng = np.random.default_rng()
        seed = rng.integers(np.iinfo(np.uint32).max)
    else:
        try:
            seed = int(seed) & np.iinfo(np.uint32).max
        except ValueError:
            seed = hash(seed) & np.iinfo(np.uint32).max
    seeds = np.array([seed], dtype=np.uint32)  # use given seed for the first iteration
    if iteration_size > 1:
        seed_seq = np.random.SeedSequence(seed)
        seeds = np.concatenate((seeds, seed_seq.generate_state(iteration_size - 1)))

    # create and parse output directory
    output_path = "output"
    os.makedirs(output_path, exist_ok=True)
    dir_list = os.listdir(output_path)
    if len(dir_list):
        pattern = re.compile(r"([0-9][0-9][0-9][0-9][0-9][0-9])-([0-9][0-9])\.\w*")
        match_list = [pattern.match(f) for f in dir_list]
        next_index = max([int(m[1]) if m else -1 for m in match_list]) + 1
    else:
        next_index = 0

    prompts = [prompt]*batch_size
    neg_prompts = [neg_prompt]*batch_size if neg_prompt != "" else None
    images = []
    time_taken = 0
    for i in range(iteration_size):
        log = open(os.path.join(output_path, "history.txt"), "a")
        info = f"{next_index+i:06} | prompt: {prompt} negative prompt: {neg_prompt} | scheduler: {sched_name} " + \
            f"model: {model_path} iteration size: {iteration_size} batch size: {batch_size} steps: {steps} " + \
            f"scale: {guidance_scale} height: {height} width: {width} eta: {eta} seed: {seeds[i]} \n"
        log.write(info)
        log.close()

        # Generate our own latents so that we can provide a seed.
        latents = get_latents_from_seed(seeds[i], batch_size, height, width)

        start = time.time()
        batch_images = pipe(prompts, negative_prompt=neg_prompts, height=height, width=width,
                            num_inference_steps=steps, guidance_scale=guidance_scale, eta=eta, latents=latents).images
        finish = time.time()

        for j in range(batch_size):
            batch_images[j].save(os.path.join(output_path, f"{next_index+i:06}-{j:02}.{image_format}"))

        images.extend(batch_images)
        time_taken = time_taken + (finish - start)

    time_taken = time_taken / 60.0
    if iteration_size > 1:
        status = f"Run indexes {next_index:06} to {next_index+iteration_size-1:06} took {time_taken:.1f} minutes " + \
            f"to generate {iteration_size} iterations with batch size of {batch_size}. seeds: " + \
            np.array2string(seeds, separator=",")
    else:
        status = f"Run index {next_index:06} took {time_taken:.1f} minutes to generate a batch size of " + \
            f"{batch_size}. seed: {seeds[0]}"

    return images, status


def clear_input():
    return "", "", 1, 1, 16, 7.5, 512, 512, 0.0, "", "png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gradio interface for ONNX based Stable Diffusion")
    parser.add_argument('--model', dest='model_path', default="./stable_diffusion_onnx",
                        help='path to the model directory')
    args = parser.parse_args()

    model_path = args.model_path
    scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    pipe = StableDiffusionOnnxPipeline.from_pretrained(model_path, provider="DmlExecutionProvider", scheduler=scheduler)
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))  # Disable the safety checker

    # check versions
    is_DDIM = type(scheduler) == DDIMScheduler
    diff_ver = diffusers.__version__.split(".")
    is_v_0_4 = (int(diff_ver[0]) > 0) or (int(diff_ver[1]) >= 4)

    # create gradio block
    title = "Stable Diffusion ONNX"
    with gr.Blocks(title=title) as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                pt_tb = gr.Textbox(value="a photo of an astronaut riding a horse on mars", lines=2, label="prompt")
                np_tb = gr.Textbox(value="", lines=2, label="negative prompt", visible=is_v_0_4)
                it_sl = gr.Slider(1, 24, value=1, step=1, label="iteration size")
                bs_sl = gr.Slider(1, 4, value=1, step=1, label="batch size")
                st_sl = gr.Slider(1, 100, value=16, step=1, label="steps")
                gs_sl = gr.Slider(0, 50, value=7.5, step=0.1, label="guidance")
                ht_sl = gr.Slider(384, 768, value=512, step=64, label="height")
                wd_sl = gr.Slider(384, 768, value=512, step=64, label="width")
                et_sl = gr.Slider(0, 1, value=0.0, step=0.01, label="eta", visible=is_DDIM)
                sd_tb = gr.Textbox(value="", max_lines=1, label="seed")
                if_rd = gr.Radio(["png", "jpg"], value="png", label="image format")
                with gr.Row():
                    cl_bt = gr.Button("Clear")
                    gn_bt = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1, min_width=600):
                ot_gl = gr.Gallery(value=None, label="images")
                st_tb = gr.Textbox(value="", label="status")

        # config components
        cl_bt.click(fn=clear_input, inputs=None, outputs=[pt_tb, np_tb, it_sl, bs_sl, st_sl, gs_sl, ht_sl, wd_sl,
                    et_sl, sd_tb, if_rd], queue=False)
        gn_bt.click(fn=generate, inputs=[pt_tb, np_tb, it_sl, bs_sl, st_sl, gs_sl, ht_sl, wd_sl, et_sl, sd_tb, if_rd],
                    outputs=[ot_gl, st_tb])

        ot_gl.style(grid=2)

    # start gradio web interface on local host
    demo.launch()

    # use the following to launch the web interface to a private network
    # demo.queue(concurrency_count=1)
    # demo.launch(server_name="0.0.0.0")

