# MaxFusion: Plug & Play multimodal generation in text to image diffusion models


### [Project Page]()&ensp;&ensp;&ensp;[Paper]()


## Prompt Edits





## Attention Control Options
 * `cross_replace_steps`: specifies the fraction of steps to edit the cross attention maps. Can also be set to a dictionary `[str:float]` which specifies fractions for different words in the prompt.
 * `self_replace_steps`: specifies the fraction of steps to replace the self attention maps.
 * `local_blend` (optional):  `LocalBlend` object which is used to make local edits. `LocalBlend` is initialized with the words from each prompt that correspond with the region in the image we want to edit.
 * `equalizer`: used for attention Re-weighting only. A vector of coefficients to multiply each cross-attention weight

## Citation

``` bibtex

}
```
