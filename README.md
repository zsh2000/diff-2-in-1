> # [ICLR 2025] Diff-2-in-1: Bridging Generation and Dense Perception with Diffusion Models <br>
> [Paper](https://openreview.net/pdf?id=ZYd5wJSaMs)

> [Website](https://zsh2000.github.io/diff-2-in-1.github.io/)

This repository contains a PyTorch implementation of our paper "Diff-2-in-1: Bridging Generation and Dense Perception with Diffusion Models". We take the surface normal prediction task on [Bae et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Bae_Estimating_and_Exploiting_the_Aleatoric_Uncertainty_in_Surface_Normal_Estimation_ICCV_2021_paper.pdf) as an example to show how our Diff-2-in-1 works.

## Installation

#### Tested on a single NVIDIA A100 GPU with 40GB memory.

Our package requirements follow the works [Grounded Diffusion](https://github.com/Lipurple/Grounded-Diffusion) and [Bae et al.](https://github.com/baegwangbin/surface_normal_uncertainty).

To install the dependencies, a conda environment named `diff-2-in-1` can be created and activated with:

```
conda env create -f environment.yaml
conda activate diff-2-in-1
```

## Dataset

The NYUv2 surface normal dataset can be prepared in the same way as [Bae et al.](https://github.com/baegwangbin/surface_normal_uncertainty). It can be accessed via the [Google Drive link](https://drive.google.com/drive/folders/1Ku25Am69h_HrbtcCptXn4aetjo7sB33F?usp=sharing). In our work, we adopt the official train/test split that contains 795/654 images. Prepare the dataset under the `./dataset/nyu/` folder, so that `./dataset/nyu/train` and `./datasets/nyu/test/` contain the train/test data of NYUv2.

Except for preparing the RGB images and surface normal maps, the captions for the images are also needed. We use [ClipCap](https://github.com/rmokady/CLIP_prefix_caption) to generate captions for each image. Finally, the dataset structure looks like this:

```
dataset
  - nyu
     - train
        - img
           - 000002.png
           - 000003.png
           - 000004.png
           ...
           - 001439.png
        - norm
           - 000002.png
           - 000003.png
           - 000004.png
           ...
           - 001439.png
        - caption
           - 000002.txt
           - 000003.txt
           - 000004.txt
           ...
           - 001439.txt
     - test
        - img
           - 000000.png
           - 000001.png
           - 000008.png
           ...
           - 001448.png
        - norm
           - 000000.png
           - 000001.png
           - 000008.png
           ...
           - 001448.png
        - caption
           - 000000.txt
           - 000001.txt
           - 000008.txt
           ...
           - 001448.txt
```

## Checkpoints

Our work uses stable diffusion v1.5 as the diffusion model backbone. Download the checkpoint `model.ckpt` in [Huggingface](https://huggingface.co/ShuhongZheng/diff-2-in-1), and put inside `./models/ldm/stable-diffusion-v1`.

For inference, the pretrained weights can also be accessed at [Huggingface](https://huggingface.co/ShuhongZheng/diff-2-in-1).


## Training

For the warm-up stage for the task head, run the coommand:

```
python training_1st_stage.py
```

Then we create the synthetic multi-modal data with the command:

```
python data_creation.py
```

Afterwards, the created data needs to be combined with the original data, and save in the directory `./dataset/nyu_nyu_T600/` in the same format as `./dataset/nyu/`.

Finally, we can perform the self-improving stage with the command:

```
python training_2nd_stage.py --sn_ckpt [sn_ckpt_path] --bae_ckpt [bae_ckpt_path]
```

`sn_ckpt_path` and `bae_ckpt_path` are the paths for the checkpoints obtained from the warm-up stage.

## Testing
The inference process can be performed with the command:

```
python test.py --sn_ckpt [sn_ckpt_path] --bae_ckpt [bae_ckpt_path]
```

## Citation
If you find our work useful, please consider citing:
```BibTeX
@inproceedings{zheng2025diff2in1,
  title={Diff-2-in-1: Bridging Generation and Dense Perception with Diffusion Models},
  author={Zheng, Shuhong and Bao, Zhipeng and Zhao, Ruoyu and Hebert, Martial and Wang, Yu-Xiong},
  booktitle={ICLR},
  year={2025}
}
```

### Acknowledgement
The codes are largely borrowed from the following repositories:

https://github.com/Lipurple/Grounded-Diffusion

https://github.com/baegwangbin/surface_normal_uncertainty

This work was supported in part by NSF Grant 2106825, NIFA Award 2020-67021-32799, the Toyota Research Institute, the IBM-Illinois Discovery Accelerator Institute, the Amazon-Illinois Center on AI for Interactive Conversational Experiences, Snap Inc., and the Jump ARCHES endowment through the Health Care Engineering Systems Center at Illinois and the OSF Foundation. This work used computational resources, including the NCSA Delta and DeltaAI supercomputers through allocations CIS220014 and CIS230012 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, as well as the TACC Frontera supercomputer and Amazon Web Services (AWS) through the National Artificial Intelligence Research Resource (NAIRR) Pilot.