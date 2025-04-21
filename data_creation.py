import warnings
from pytorch_lightning import seed_everything
warnings.filterwarnings("ignore")
import argparse, os
import PIL
import torch
from datetime import datetime

from ldm.data.nyu import NYU_Dataset
import torchvision.datasets as datasets

import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import torchvision
import random
from ldm.util import instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler
import torch.optim as optim
from ldm.modules.diffusionmodules.openaimodel import clear_feature_dic,get_feature_dic
from ldm.models.sn_module import SNmodule

from ldm.models.decoder import Decoder

import cv2

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
        
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(h,w)
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--save_name",
        type=str,
        help="the save dir name",
        default=""
    )
    
    parser.add_argument(
        "--class_split",
        type=int,
        help="the class split: 1,2,3",
        default=1
    )
    parser.add_argument(
        "--train_data",
        type=str,
        help="the type of training data: single, two, random",
        default="random"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="the batch size",
        default=1
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        help="the number of workers",
        default=4
    )

    parser.add_argument(
        "--epoch",
        type=int,
        help="number of epoch",
        default=1
    )

    parser.add_argument(
        "--save_interval",
        type=int,
        help="the interval of saving the model",
        default=5
    )

    parser.add_argument(
        "--ema_update_interval",
        type=int,
        help="interval of ema update",
        default=10
    )

    parser.add_argument(
        "--sn_ckpt",
        type=str,
        help="the path of sn checkpoint",
        default="checkpoint_50.pth"
    )

    parser.add_argument(
        "--bae_ckpt",
        type=str,
        help="the path of bae checkpoint",
        default="checkpoint_bae_50.pth"
    )
    
    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    starttime = datetime.now()

    sn_module = SNmodule().to(device)

    bae_module = Decoder().to(device)

    sn_module.load_state_dict(torch.load(opt.sn_ckpt, map_location="cpu"), strict=True)
    bae_module.load_state_dict(torch.load(opt.bae_ckpt, map_location="cpu"), strict=True)


    model = load_model_from_config(config, f"{opt.ckpt}")
    endtime = datetime.now()
    print ("the time of load_model_from_config is",(endtime - starttime).seconds)
    
    model = model.to(device)

    
    sampler = DDIMSampler(model)

    batch_size = opt.batch_size


    print('***********************   begin   **********************************')
    
    total_epoch = opt.epoch  # generate 1 sample for each image in the original dataset


    train_dataset = NYU_Dataset()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True, drop_last=True)


    print("Start training with maximum {0} iterations.".format(total_epoch))
    
    start_code = None
    if opt.fixed_code:
        print('start_code')
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)  

    
    cnt = 0
    for j in range(total_epoch):
        print('Epoch ' +  str(j) + '/' + str(total_epoch))
        for iteration, samples in enumerate(train_loader):
            gt_imgs = samples["image"]
            prompts = samples["caption"]
            
            with torch.no_grad():
                clear_feature_dic()
   


                c = model.get_learned_conditioning(prompts)
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                uc = None

                feature_in = model.encode_first_stage(gt_imgs.float().cuda())
            
                z = model.get_first_stage_encoding(feature_in).detach()

                t = torch.ones((gt_imgs.shape[0],), device="cuda").long() * 600
                noise = None
                noise = default(noise, lambda: torch.randn_like(z))
                x_noisy = model.q_sample(x_start=z, t=t, noise=noise)


                start_code = x_noisy
                samples_ddim,_, _ = sampler.sample(S=opt.ddim_steps,
                                                conditioning=c,
                                                batch_size=opt.n_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=opt.scale,
                                                unconditional_conditioning=uc,
                                                eta=opt.ddim_eta,
                                                x_T=start_code,
                                                ddpm_num_steps=t)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                diffusion_features=get_feature_dic()

                feature_1, feature_2, feature_3, feature_4, feature_5 = sn_module(diffusion_features)

                norm_out_list, _, _ = bae_module(feature_1, feature_2, feature_3, feature_4, feature_5, gt_norm_mask=None, mode='test')
                norm_out = norm_out_list[-1]

                pred_norm = norm_out[:, :3, :, :]

                for i in range(x_samples_ddim.shape[0]):
                    cnt += 1
                    rgb_map = np.clip(np.transpose(x_samples_ddim[i].detach().cpu().numpy(), (1, 2, 0)), -1, 1)
                    cv2.imwrite("./n600/rgb_syn_"+str(cnt)+".png", (rgb_map[:, :, [2,1,0]]+1)/2*255.)

                for i in range(pred_norm.shape[0]):
                    sn_map = np.clip(np.transpose(pred_norm[i].detach().cpu().numpy(), (1, 2, 0)), -1, 1)
                    cv2.imwrite("./n600/sn_syn_"+str(cnt)+".png", (sn_map[:, :, [2,1,0]]+1)/2*255.)
        
if __name__ == "__main__":
    main()
