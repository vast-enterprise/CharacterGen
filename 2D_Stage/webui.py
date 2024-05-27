import gradio as gr
from PIL import Image
import glob

import io
import argparse
import inspect
import os
import random
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import numpy as np

import torch
import torch.utils.checkpoint

from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import check_min_version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from torchvision import transforms

from tuneavideo.models.unet_mv2d_condition import UNetMV2DConditionModel
from tuneavideo.models.unet_mv2d_ref import UNetMV2DRefModel
from tuneavideo.models.PoseGuider import PoseGuider
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.util import shifted_noise
from einops import rearrange
from PIL import Image
from torchvision.utils import save_image
import PIL
from PIL import Image
import json

check_min_version("0.24.0")

logger = get_logger(__name__, log_level="INFO")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_bg_color(bg_color):
    if bg_color == 'white':
        bg_color = np.array([1., 1., 1.], dtype=np.float32)
    elif bg_color == 'black':
        bg_color = np.array([0., 0., 0.], dtype=np.float32)
    elif bg_color == 'gray':
        bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    elif bg_color == 'random':
        bg_color = np.random.rand(3)
    elif isinstance(bg_color, float):
        bg_color = np.array([bg_color] * 3, dtype=np.float32)
    else:
        raise NotImplementedError
    return bg_color

def process_image(image, totensor):
    # image
    if not image.mode == "RGBA":
        image = image.convert("RGBA")
    bg_color = get_bg_color("gray")
    # paste to center
    max_dim = max(image.width, image.height)
    new_image = Image.new("RGBA", (max_dim, max_dim), tuple(bg_color))
    left = (max_dim - image.width) // 2
    top = (max_dim - image.height) // 2
    new_image.paste(image, (left, top))

    image = new_image.resize((768, 768), resample=PIL.Image.BICUBIC)
    image = np.array(image)
    image = image.astype(np.float32) / 255.
    assert image.shape[-1] == 4  # RGBA
    alpha = image[..., 3:4]
    image = image[..., :3] * alpha + bg_color * (1 - alpha)
    # save image
    new_image = Image.fromarray((image * 255).astype(np.uint8))
    new_image.save("input.png")
    return totensor(image)

class Inference_API:

    def __init__(self):
        self.validation_pipeline = None

    @torch.no_grad()
    def inference(self, input_image, vae, feature_extractor, image_encoder, unet, ref_unet, tokenizer, text_encoder, pretrained_model_path, generator, validation, val_width, val_height, unet_condition_type,
                    pose_guider=None, use_noise=True, use_shifted_noise=False, noise_d=256, crop=False, seed=100, timestep=20):
        set_seed(seed)
        # Get the validation pipeline
        if self.validation_pipeline is None:
            noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
            if use_shifted_noise:
                print(f"enable shifted noise for {val_height} to {noise_d}")
                betas = shifted_noise(noise_scheduler.betas, image_d=val_height, noise_d=noise_d)
                noise_scheduler.betas = betas
                noise_scheduler.alphas = 1 - betas
                noise_scheduler.alphas_cumprod = torch.cumprod(noise_scheduler.alphas, dim=0)
            self.validation_pipeline = TuneAVideoPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, ref_unet=ref_unet,feature_extractor=feature_extractor,image_encoder=image_encoder,
                scheduler=noise_scheduler
            )
            self.validation_pipeline.enable_vae_slicing()
            self.validation_pipeline.set_progress_bar_config(disable=True)

        totensor = transforms.ToTensor()

        metas = json.load(open("./material/pose.json", "r"))
        cameras = []
        pose_images = []
        input_path = "./material"
        for lm in metas:
            cameras.append(torch.tensor(np.array(lm[0]).reshape(4, 4).transpose(1,0)[:3, :4]).reshape(-1))
            if not crop:
                pose_images.append(totensor(np.asarray(Image.open(os.path.join(input_path, lm[1])).resize(
                    (val_height, val_width), resample=PIL.Image.BICUBIC)).astype(np.float32) / 255.))
            else:
                pose_image = Image.open(os.path.join(input_path, lm[1]))
                crop_area = (128, 0, 640, 768)
                pose_images.append(totensor(np.array(pose_image.crop(crop_area)).astype(np.float32)) / 255.)
        camera_matrixs = torch.stack(cameras).unsqueeze(0).to("cuda")
        pose_imgs_in = torch.stack(pose_images).to("cuda")
        prompts = "high quality, best quality"
        prompt_ids = tokenizer(
            prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        ).input_ids[0]

        # (B*Nv, 3, H, W)
        B = 1
        weight_dtype = torch.bfloat16
        imgs_in = process_image(input_image, totensor)
        imgs_in = rearrange(imgs_in.unsqueeze(0).unsqueeze(0), "B Nv C H W -> (B Nv) C H W")
                
        with torch.autocast("cuda", dtype=weight_dtype):
            imgs_in = imgs_in.to("cuda")
            # B*Nv images
            out = self.validation_pipeline(prompt=prompts, image=imgs_in.to(weight_dtype), generator=generator, 
                                        num_inference_steps=timestep,
                                        camera_matrixs=camera_matrixs.to(weight_dtype), prompt_ids=prompt_ids, 
                                        height=val_height, width=val_width, unet_condition_type=unet_condition_type, 
                                        pose_guider=None, pose_image=pose_imgs_in, use_noise=use_noise, 
                                        use_shifted_noise=use_shifted_noise, **validation).videos
            out = rearrange(out, "B C f H W -> (B f) C H W", f=validation.video_length)

        image_outputs = []
        for bs in range(4):
            img_buf = io.BytesIO()
            save_image(out[bs], img_buf, format='PNG')
            img_buf.seek(0)
            img = Image.open(img_buf)
            image_outputs.append(img)
        torch.cuda.empty_cache()
        return image_outputs 

@torch.no_grad()
def main(
    pretrained_model_path: str,
    image_encoder_path: str,
    ckpt_dir: str,
    validation: Dict,
    local_crossattn: bool = True,
    unet_from_pretrained_kwargs=None,
    unet_condition_type=None,
    use_pose_guider=False,
    use_noise=True,
    use_shifted_noise=False,
    noise_d=256
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    device = "cuda"

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
    feature_extractor = CLIPImageProcessor()
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNetMV2DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", local_crossattn=local_crossattn, **unet_from_pretrained_kwargs)
    ref_unet = UNetMV2DRefModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", local_crossattn=local_crossattn, **unet_from_pretrained_kwargs)
    if use_pose_guider:
        pose_guider = PoseGuider(noise_latent_channels=4).to("cuda")
    else:
        pose_guider = None

    unet_params = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu")
    if use_pose_guider:
        pose_guider_params = torch.load(os.path.join(ckpt_dir, "pytorch_model_1.bin"), map_location="cpu")
        ref_unet_params = torch.load(os.path.join(ckpt_dir, "pytorch_model_2.bin"), map_location="cpu")
        pose_guider.load_state_dict(pose_guider_params)
    else:
        ref_unet_params = torch.load(os.path.join(ckpt_dir, "pytorch_model_1.bin"), map_location="cpu")
    unet.load_state_dict(unet_params)
    ref_unet.load_state_dict(ref_unet_params)

    weight_dtype = torch.float16

    text_encoder.to(device, dtype=weight_dtype)
    image_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    ref_unet.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    ref_unet.requires_grad_(False)

    generator = torch.Generator(device="cuda")
    inferapi = Inference_API()
    def gen4views(image, width, height, seed, timestep=20):
        return inferapi.inference(
            image, vae, feature_extractor, image_encoder, unet, ref_unet, tokenizer, text_encoder, pretrained_model_path,
            generator, validation, width, height, unet_condition_type,
            pose_guider=pose_guider, use_noise=use_noise, use_shifted_noise=use_shifted_noise, noise_d=noise_d,
            crop=True, seed=seed, timestep=timestep
        )

    with gr.Blocks() as demo:
        gr.Markdown("# [SIGGRAPH'24] CharacterGen: Efficient 3D Character Generation from Single Images with Multi-View Pose Calibration")
        gr.Markdown("# 2D Stage: One Image to Four Views of Character Image")
        gr.Markdown("**Please Upload the Image without background, and the pictures uploaded should preferably be full-body frontal photos.**")
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="pil", label="Upload Image(without background)", image_mode="RGBA")
                gr.Examples(
                    label="Example Images",
                    examples=glob.glob("./material/examples/*.png"),
                    inputs=[img_input]
                )
                with gr.Row():
                    width_input = gr.Number(label="Width", value=512)
                    height_input = gr.Number(label="Height", value=768)
                    seed_input = gr.Number(label="Seed", value=2333)
                timestep = gr.Slider(minimum=10, maximum=60, step=1, value=40, label="Timesteps")
            with gr.Column():
                button = gr.Button(value="Generate")
                output = gr.Gallery(label="4 views of Character Image")
        
        button.click(
            fn=gen4views,
            inputs=[img_input, width_input, height_input, seed_input, timestep],
            outputs=[output]
        )

    demo.launch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/infer.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))