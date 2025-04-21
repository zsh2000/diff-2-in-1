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
from losses import compute_loss


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


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
        default=5.0,
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
        default=8
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        help="the number of workers",
        default=8
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        help="the learning rate",
        default=0.000357   # learning rate inherited from the original surface normal uncertainty paper (https://arxiv.org/abs/2109.09881) 
    )

    parser.add_argument(
        "--epoch",
        type=int,
        help="number of epoch",
        default=50
    )

    parser.add_argument(
        "--save_interval",
        type=int,
        help="the interval of saving the model",
        default=5
    )
    
    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    starttime = datetime.now()
    sn_module=SNmodule().to(device)
    bae_module = Decoder().to(device)
    model = load_model_from_config(config, f"{opt.ckpt}")
    endtime = datetime.now()
    print ("the time of load_model_from_config is",(endtime - starttime).seconds)
    
    model = model.to(device)


    batch_size = opt.batch_size
    num_workers = opt.num_workers


    print('***********************   begin   **********************************')
    
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    save_dir = 'checkpoint'
    os.makedirs(save_dir, exist_ok=True)
    learning_rate = opt.learning_rate
    total_epoch = opt.epoch
    
    ckpt_dir = os.path.join(save_dir, opt.save_name+'run-'+current_time)
    os.makedirs(ckpt_dir, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'logs'))
    os.makedirs(os.path.join(ckpt_dir, 'training'), exist_ok=True)
    
    g_optim = optim.Adam(
                [{"params": list(sn_module.parameters()) + list(bae_module.parameters())},],
                lr=learning_rate
              )

    loss_fn = compute_loss("UG_NLL_ours")

    train_dataset = NYU_Dataset()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)


    print("Start training with maximum {0} iterations.".format(total_epoch))
    
    start_code = None
    if opt.fixed_code:
        print('start_code')
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)  

    
    batch_size = opt.n_samples

    for epoch_num in range(total_epoch):
        print('Epoch ' +  str(epoch_num) + '/' + str(total_epoch))
        for iteration, samples in enumerate(train_loader):
            gt_imgs = samples["image"]
            gt_sns = samples["sns"]
            gt_norm_mask = samples["sn_masks"]

            clear_feature_dic()
            uc = None
            
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(gt_imgs.shape[0] * [""])
            feature_in = model.encode_first_stage(gt_imgs.float().cuda())
            
            z = model.get_first_stage_encoding(feature_in).detach()
            
            t = torch.zeros((gt_imgs.shape[0],), device="cuda").long()
            noise = None
            noise = default(noise, lambda: torch.randn_like(z))
            x_noisy = model.q_sample(x_start=z, t=t, noise=noise)
            model_output = model.apply_model(x_noisy, t, uc)
            diffusion_features = get_feature_dic()
            feature_1, feature_2, feature_3, feature_4, feature_5 = sn_module(diffusion_features)
            gt_norm_mask = np.transpose(gt_norm_mask, (0, 3, 1, 2))

            norm_out_list, pred_list, coord_list = bae_module(feature_1, feature_2, feature_3, feature_4, feature_5, gt_norm_mask=gt_norm_mask, mode='train')
            loss = loss_fn(pred_list, coord_list, gt_sns.cuda(), gt_norm_mask.cuda())

            g_optim.zero_grad()
            loss.backward()
            g_optim.step()


        if epoch_num % opt.save_interval == 0:
            print("Saving latest checkpoint to",ckpt_dir)
            torch.save(sn_module.state_dict(), os.path.join(ckpt_dir, 'checkpoint_'+str(epoch_num)+'.pth'))
            torch.save(bae_module.state_dict(), os.path.join(ckpt_dir, 'checkpoint_bae_'+str(epoch_num)+'.pth'))
        
if __name__ == "__main__":
    main()
