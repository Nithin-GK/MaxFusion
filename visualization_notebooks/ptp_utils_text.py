# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
import torch.nn.functional as F
from einops import rearrange
import os
from fused_forward import forwardfused
def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img



def view_images(images, num_rows=1, offset_ratio=0.02,results_dir='results'):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)
    if os.path.exists(results_dir)==False:
        os.makedirs(results_dir)
    img_name=len(os.listdir(results_dir))
    pil_img.save(os.path.join(results_dir,str(img_name)+'.png'))

def prepare_image(
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height))
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

@torch.no_grad()
def diffusion_step(model, latents, context, t, guidance_scale, low_resource=False):

    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        guess_mode=False
        latents = latents
        latents_input = torch.cat([latents] * 2)

        (noise_pred,variancelist_encoder,variancelist_decoder) = model.unet(
                    latents_input,
                    t,
                    encoder_hidden_states=context,
                    cross_attention_kwargs={}
                )
        # variancelist_encoder=noise_pred.variance_encoder
        # variancelist_decoder=noise_pred.variance_decoder

        # noise_pred=noise_pred.noise_pred

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        # for _ in variancelist_decoder:
        #   print(_.shape)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t.to(noise_pred.device), latents.to(noise_pred.device))["prev_sample"]
    return latents,variancelist_encoder,variancelist_decoder


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents

import PIL.Image



@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    negative_prompts: List[str],
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False
):
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        negative_prompts, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    # set timesteps
    extra_set_kwargs = {}
    model.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    all_variance_encoder=[]
    all_variance_decoder=[]
    for t in tqdm(model.scheduler.timesteps):
        latents, variance_encoder,variance_decoder = diffusion_step(model, latents, context, t, guidance_scale,low_resource = low_resource)
        all_variance_encoder.append(variance_encoder)
        all_variance_decoder.append(variance_decoder)

    image = latent2image(model.vae, latents)
  
    return image, latent, all_variance_encoder, all_variance_decoder

@torch.no_grad()
def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward_diffusers(
            hidden_states=None,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale=1.0,
        ):
            residual = hidden_states

            args = () 

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states, *args)

            if encoder_hidden_states is None:
                is_cross=False
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                is_cross=True

                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states, *args)
            value =self.to_v(encoder_hidden_states, *args)
            is_self= (key.shape[1]==query.shape[1])
            is_cross= not is_self

            key_u,key_c=key.chunk(2)
            query_u,query_c=query.chunk(2)
            value_u,value_c=value.chunk(2)

            if not is_cross:
                query_u, key_u, value_u = controller.replace_self_attention_feats(query_u, key_u, value_u, place_in_unet)
                key_u, value_u = controller.replace_key_value_feats(key_u, value_u, place_in_unet)
                query_u, key_u = controller.replace_self_attention(query_u, key_u,  place_in_unet)

                query_c, key_c, value_c = controller.replace_self_attention_feats(query_c, key_c, value_c, place_in_unet)
                key_c, value_c = controller.replace_key_value_feats(key_c, value_c, place_in_unet)
                query_c, key_c = controller.replace_self_attention(query_c, key_c,  place_in_unet)

            else:
                query_u= controller.replace_cross_attention(query_u,  place_in_unet)
                query_c = controller.replace_cross_attention(query_c,  place_in_unet)

            query_u = self.head_to_batch_dim(query_u)
            key_u = self.head_to_batch_dim(key_u)
            value_u = self.head_to_batch_dim(value_u)

            query_c = self.head_to_batch_dim(query_c)
            key_c = self.head_to_batch_dim(key_c)
            value_c = self.head_to_batch_dim(value_c)

            assert torch.is_tensor(query_u)
            assert torch.is_tensor(key_u)
            assert torch.is_tensor(value_u)

            attention_probs_u,hidden_states_u=controller.get_attention(query_u, key_u,value_u, self.scale, place_in_unet, is_cross)
            attention_probs_c, hidden_states_c=controller.get_attention(query_c, key_c, value_c,self.scale, place_in_unet, is_cross)
 
            attention_probs=torch.cat([attention_probs_u,attention_probs_c],0)
            attention_probs = controller(attention_probs, is_cross, place_in_unet)
            hidden_states=torch.cat([hidden_states_u,hidden_states_c],0)

            hidden_states= self.batch_to_head_dim(hidden_states)
            hidden_states = self.to_out[0](hidden_states, *args)
            hidden_states = self.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states
        return forward_diffusers

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        # print(net_.__class__.__name__)
   
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        # print( net[0])
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    # print(cross_att_count)
    controller.num_att_layers = cross_att_count

    
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words

