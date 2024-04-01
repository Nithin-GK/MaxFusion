

<h2 align="center"> <a href="">MaxFusion: Plug & Play multimodal generation in text to image diffusion models</a></h2>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>



<h5 align="center">
    
[![hf_space](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)]()
[![project page]()
[![arXiv](https://img.shields.io/badge/Arxiv-.svg?logo=arXiv)] <br>



## Applications
<img src="./assets/img1.png" width="100%">

<img src="./assets/img2.png" width="100%">




This repository contains the implementation of the paper:
> **Unite and Conquer: Plug & Play Multi-Modal Synthesis using Diffusion Models**<br>
> [Nithin Gopalakrishnan Nair](https://nithin-gk.github.io/), [Chaminda Bandara](https://www.wgcban.com/), [Vishal M Patel](https://engineering.jhu.edu/vpatel36/vishal-patel/)

IEEE/CVF International Conference on Computer Vision (**CVPR**), 2023

From [VIU Lab](https://engineering.jhu.edu/vpatel36/), Johns Hopkins University

[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Nair_Unite_and_Conquer_Plug__Play_Multi-Modal_Synthesis_Using_Diffusion_CVPR_2023_paper.pdf)] |
[[Project Page](https://nithin-gk.github.io/projectpages/Multidiff)] |
[[Video](https://youtu.be/N4EOwnhNzIk)]

Keywords: Multimodal Generation, Semantic Face Generation, Multimodal Face generation, Text to image generation, Diffusion based Face Generation, Text to Image Generation, Text to Face Generation

## Applications
<img src="./utils/faces.png" width="100%">

<img src="./utils/natural.png" width="100%">

We propose **MaxFusion**, a plug and play framework for multimodal generation using text to image diffusion models.
    *(a) Multimodal generation*. We address the problem of conflicting spatial conditioning for text to iamge models .
    *(b) Saliency in variance maps*. We discover that the variance maps of different feature layers expresses the strength og conditioning.

<br>


### Contributions:

- We tackle the need for training with paired data for multi-task conditioning using diffusion models.
- We propose a novel variance-based feature merging strategy for diffusion models.
- Our method allows us to use combined information to influence the output, unlike individual models that are limited to a single condition.
- Unlike previous solutions, our approach is easily scalable and can be added on top of off-the-shelf models.

<!-- <p align="center">
  <img src="./utils/intropng.png" alt="Centered Image" style="width: 50%;">
</p> -->

## Environment setup 


```
conda env create -f environment.yml
```


## Code demo:

A notebook for differnt demo conditions is provided in demo.ipynb


# Testing On custom datasets 

Will be released shortly

##  Instructions for Interactive Demo

An intractive demo can be run locally using

```
python gradio_maxfusion.py

```

## Citation
5. If you use our work, please use the following citation
```

```

This code is reliant on:
```
https://github.com/google/prompt-to-prompt/
```

## Citation

``` 
```
