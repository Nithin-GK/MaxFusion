
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from PIL import Image

controlnet_libs = {
    'pose': "lllyasviel/sd-controlnet-openpose",
    'scribble': "lllyasviel/sd-controlnet-scribble",
    'canny': "lllyasviel/sd-controlnet-canny",
    'hed': "lllyasviel/sd-controlnet-hed",
    'depth': "lllyasviel/sd-controlnet-depth"

}


def load_pipeline(mode_1, mode_2,device='cuda'):
    controlnet_1 = ControlNetModel.from_pretrained(controlnet_libs[mode_1])
    controlnet_2 = ControlNetModel.from_pretrained(controlnet_libs[mode_2])

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet_2
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.controlnet1=controlnet_1
    pipe.controlnet2= controlnet_2

    pipe=pipe.to("cuda")

    return pipe