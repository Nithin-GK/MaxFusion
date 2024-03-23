from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalControlnetMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
)
from diffusers.models.unet_2d_condition import UNet2DConditionModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def return_maxvar_feat_intra(controlnet_cond_pose,controlnet_cond_depth):
        A= controlnet_cond_pose
        B= controlnet_cond_depth
        mean_A = torch.mean(A, dim=1, keepdim=True)
        mean_B = torch.mean(B, dim=1, keepdim=True)
        A_demeaned = A - mean_A
        B_demeaned = B - mean_B
        covariance = torch.sum(A_demeaned * B_demeaned, dim=1)
        std_A = torch.sqrt(torch.sum(A_demeaned ** 2, dim=1))
        std_B = torch.sqrt(torch.sum(B_demeaned ** 2, dim=1))
        correlation = covariance / (std_A * std_B)
        A_flat = A.view(A.size(0), A.size(1), -1)  
        B_flat = B.view(B.size(0), B.size(1), -1) 

        cosine_sim = (F.cosine_similarity(A_flat, B_flat, dim=1)+1)/2

        cosine_sim = cosine_sim.view(A.size(0), A.size(2), A.size(3))

 
        std_A = torch.std(A, dim=1, keepdim=True)
        std_B = torch.std(B, dim=1, keepdim=True)
  
        var_A = torch.var(A , dim=1, keepdim=True)
        var_B = torch.var(B , dim=1, keepdim=True)
   

        var_A=var_A /torch.sum(var_A )
        var_B=var_B/torch.sum(var_B)
    
        high_sim_threshold =0.7 
        average=(A+B)/2

        fuse_based_on_variance1 = torch.where(var_A >= var_B, A, torch.div(B*std_A,std_B))  
        fuse_based_on_variance2 = torch.where(var_A >= var_B, torch.div(A*std_B,std_A), B)   


        # Decide which values to take based on the cosine similarity
        fused_tensor1 = torch.where(correlation.unsqueeze(1) > high_sim_threshold, average, fuse_based_on_variance1)
        fused_tensor2 = torch.where(correlation.unsqueeze(1) > high_sim_threshold, average, fuse_based_on_variance2)

        controlnet_cond1=fused_tensor1
        controlnet_cond2=fused_tensor2

        return controlnet_cond1, controlnet_cond2

def return_maxvar_feat_intra_sd(controlnet_cond_pose,controlnet_cond_depth):
        
        A= controlnet_cond_pose
        B= controlnet_cond_depth
        mean_A = torch.mean(A, dim=1, keepdim=True)
        mean_B = torch.mean(B, dim=1, keepdim=True)
        A_demeaned = A - mean_A
        B_demeaned = B - mean_B
        covariance = torch.sum(A_demeaned * B_demeaned, dim=1)
        std_A = torch.sqrt(torch.sum(A_demeaned ** 2, dim=1))
        std_B = torch.sqrt(torch.sum(B_demeaned ** 2, dim=1))
        correlation = covariance / (std_A * std_B)
        A_flat = A.view(A.size(0), A.size(1), -1)  
        B_flat = B.view(B.size(0), B.size(1), -1)  

        cosine_sim = (F.cosine_similarity(A_flat, B_flat, dim=1)+1)/2
        cosine_sim = cosine_sim.view(A.size(0), A.size(2), A.size(3))

        std_A = torch.std(A, dim=1, keepdim=True)
        std_B = torch.std(B, dim=1, keepdim=True)

        var_A = torch.var(A , dim=1, keepdim=True)
        var_B = torch.var(B , dim=1, keepdim=True)
        var_A=var_A /torch.sum(var_A )
        var_B=var_B/torch.sum(var_B)
  
        high_sim_threshold =0.7  
        average=(A+B)/2
   
        fuse_based_on_variance = torch.where(var_A >= var_B, A, B) 
        fused_tensor = torch.where(correlation.unsqueeze(1) > high_sim_threshold, average, fuse_based_on_variance)


        return fused_tensor



@dataclass
class ControlNetOutput(BaseOutput):
    """
    The output of [`ControlNetModel`].

    Args:
        down_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, height //resolution, width // resolution)`. Output can be
            used to condition the original UNet's downsampling activations.
        mid_down_block_re_sample (`torch.Tensor`):
            The activation of the midde block (the lowest sample resolution). Each tensor should be of shape
            `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`.
            Output can be used to condition the original UNet's middle block activation.
    """

    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor

def forwardfused(
        model1,
        model2,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond_pose: torch.FloatTensor,
        controlnet_cond_depth: torch.FloatTensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple]:
 
        channel_order = model1.config.controlnet_conditioning_channel_order

        if channel_order == "rgb":
            # in rgb order by default
            ...
        elif channel_order == "bgr":
            controlnet_cond_pose = torch.flip(controlnet_cond_pose, dims=[1])
            controlnet_cond_depth= torch.flip(controlnet_cond_depth, dims=[1])

        else:
            raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        t_emb1 = model1.time_proj(timesteps)
        t_emb2 = model2.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb1= t_emb1.to(dtype=sample.dtype)
        # print(model1.device, t_emb1.device)
        emb1 = model1.time_embedding(t_emb1, timestep_cond)
        emb2 = model2.time_embedding(t_emb2, timestep_cond)

        aug_emb1 = None
        aug_emb2 = None

        if model1.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if model1.config.class_embed_type == "timestep":
                class_labels1 = model1.time_proj(class_labels)
                class_labels2 = model2.time_proj(class_labels)

            class_emb1 =model1.class_embedding(class_labels1).to(dtype=model1.dtype)
            class_emb2 =model2.class_embedding(class_labels2).to(dtype=model2.dtype)

            emb1 = emb1 + class_emb1
            emb2 = emb2 + class_emb2

        if model1.config.addition_embed_type is not None:
            if model1.config.addition_embed_type == "text":
                aug_emb1 = model1.add_embedding(encoder_hidden_states)
                aug_emb2 = model2.add_embedding(encoder_hidden_states)

            elif model1.config.addition_embed_type == "text_time":
             
                text_embeds1 = added_cond_kwargs.get("text_embeds")
                text_embeds2 = added_cond_kwargs.get("text_embeds")

              
                time_ids1 = added_cond_kwargs.get("time_ids")
                time_embeds1 = model1.add_time_proj(time_ids1.flatten())
                time_embeds1 = time_embeds1.reshape((text_embeds1.shape[0], -1))

                add_embeds1 = torch.concat([text_embeds1, time_embeds1], dim=-1)
                add_embeds1 = add_embeds1.to(emb1.dtype)
                aug_emb1 = model1.add_embedding(add_embeds1)


                  
                time_ids2 = added_cond_kwargs.get("time_ids")
                time_embeds2 = model2.add_time_proj(time_ids2.flatten())
                time_embeds2 = time_embeds2.reshape((text_embeds2.shape[0], -1))

                add_embeds = torch.concat([text_embeds2, time_embeds2], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb2 =model2.add_embedding(add_embeds)

        emb1 = emb1 + aug_emb1 if aug_emb1 is not None else emb1
        emb2 = emb2 + aug_emb2 if aug_emb2 is not None else emb2

        sample1 = model1.conv_in(sample)
        sample2 = model2.conv_in(sample)
       
        controlnet_cond1 = model1.controlnet_cond_embedding(controlnet_cond_pose)
        controlnet_cond2 = model2.controlnet_cond_embedding(controlnet_cond_depth)

        sample1,sample2=return_maxvar_feat_intra(sample1,sample2)
        controlnet_cond_pose,controlnet_cond_depth=return_maxvar_feat_intra(controlnet_cond_pose,controlnet_cond_depth)

        sample1 = sample1 + controlnet_cond1
        sample2 = sample2 + controlnet_cond2

        down_block_res_samples1 = (sample1,)
        down_block_res_samples2= (sample2,)

        for downsample_block1,downsample_block2 in zip(model1.down_blocks,model2.down_blocks):
            if hasattr(downsample_block1, "has_cross_attention") and downsample_block1.has_cross_attention:
                sample1, res_samples1 = downsample_block1(
                    hidden_states=sample1,
                    temb=emb1,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
                sample2, res_samples2 = downsample_block2(
                    hidden_states=sample2,
                    temb=emb2,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

                sample1,sample2=return_maxvar_feat_intra(sample1,sample2)
                set1,set2=(),()

                for a,b in zip(res_samples1,res_samples2):
                    a,b = return_maxvar_feat_intra(a,b)
                    set1=set1+(a,)
                    set2=set2+(b,)

                res_samples1,res_samples2=set1,set2
            else:
                sample1, res_samples1 = downsample_block1(hidden_states=sample1, temb=emb1)
                sample2, res_samples2 = downsample_block2(hidden_states=sample2, temb=emb2)
                sample1,sample2=return_maxvar_feat_intra(sample1,sample2)
                set1,set2=(),()
                for a,b in zip(res_samples1,res_samples2):
                    a,b = return_maxvar_feat_intra(a,b)
                    set1=set1+(a,)
                    set2=set2+(b,)
                res_samples1,res_samples2=set1,set2

            down_block_res_samples1 += res_samples1
            down_block_res_samples2 += res_samples2

        if model1.mid_block is not None:
            sample1 = model1.mid_block(
                sample1,
                emb1,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )
            sample2 = model2.mid_block(
                sample2,
                emb2,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )
            sample1,sample2=return_maxvar_feat_intra(sample1,sample2)


        controlnet_down_block_res_samples1 = ()
        controlnet_down_block_res_samples2 = ()

        for down_block_res_sample1, controlnet_block1,down_block_res_sample2, controlnet_block2 in zip(down_block_res_samples1, model1.controlnet_down_blocks,down_block_res_samples2, model2.controlnet_down_blocks):
            down_block_res_sample1 = controlnet_block1(down_block_res_sample1)
            down_block_res_sample2 = controlnet_block2(down_block_res_sample2)
            down_block_res_sample1,down_block_res_sample2=return_maxvar_feat_intra_sd(down_block_res_sample1,down_block_res_sample2)

            controlnet_down_block_res_samples1 = controlnet_down_block_res_samples1 + (down_block_res_sample1,)
            controlnet_down_block_res_samples2 = controlnet_down_block_res_samples2 + (down_block_res_sample2,)

        down_block_res_samples1 = controlnet_down_block_res_samples1
        down_block_res_samples2 = controlnet_down_block_res_samples2

        mid_block_res_sample1 = model1.controlnet_mid_block(sample1)
        mid_block_res_sample2 = model2.controlnet_mid_block(sample2)
        mid_block_res_sample1,mid_block_res_sample2=return_maxvar_feat_intra_sd(mid_block_res_sample1,mid_block_res_sample2)

        # 6. scaling
        if guess_mode and not model1.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0
            scales = scales * conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
        else:
            down_block_res_samples1 = [sample * conditioning_scale for sample in down_block_res_samples1]
            mid_block_res_sample1 = mid_block_res_sample1 * conditioning_scale
            down_block_res_samples2 = [sample * conditioning_scale for sample in down_block_res_samples2]
            mid_block_res_sample2 = mid_block_res_sample2 * conditioning_scale

        if model1.config.global_pool_conditions:
            # stop
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        if not return_dict:
            return (down_block_res_samples1, mid_block_res_sample2)

        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples1, mid_block_res_sample=mid_block_res_sample2
        )


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module