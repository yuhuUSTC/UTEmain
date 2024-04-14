import argparse, os
import cv2
import torch
import random
import numpy as np
import PIL
from omegaconf import OmegaConf
from PIL import Image
from imwatermark import WatermarkEncoder
from itertools import islice
#from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler
from transformers import AutoFeatureExtractor


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False



def final_disentangle_attributes(opt):
    steps = 50  # DDIM, PLMS Sampling steps in stable-diffusion
    seed = opt.seed
    seed_torch(seed)
    # seed_everything(seed)
    original_text = opt.c1
    new_text = opt.c2
    lambda_t = [opt.lambda_t_default_1] * opt.lambda_t_star + [opt.lambda_t_default_2] * (steps - opt.lambda_t_star)  # lambda_t initialization
    lambda_t = torch.tensor(lambda_t)
    lambda_save_path = os.path.join(opt.outdir, "lambda")
    #image_save_path = os.path.join(opt.outdir, "image")
    image_save_path = opt.outdir
    
    if not os.path.exists(lambda_save_path):
        os.makedirs(lambda_save_path, exist_ok=True)
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path, exist_ok=True)
    # image_output_path = "output"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    ####SD V1.4
    model = load_model_from_config(OmegaConf.load("/mnt/workspace/workgroup/yuhu/code/Disentanglement/configs/stable-diffusion/v1-inference.yaml"),
                                   "/mnt/workspace/workgroup/yuhu/model/stable-diffusion/model.ckpt",).to(device)
    
    for param in model.parameters():
        # Check if parameter dtype is  Float (float32)
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.float16)
            
    # model = load_model_from_config(OmegaConf.load("/mnt/workspace/workgroup/yuhu/code/Disentanglement/configs/stable-diffusion/v1-inference.yaml"),
    #                                "/mnt/workspace/workgroup/yuhu/code/guided-diffusion/model/256x256_diffusion_uncond.pt",).to(device)
    sampler = PLMSSampler(model)
    num_samples = 1  # batch size


    noised_image_encode = None
    shape = [4, 64, 64]
    with torch.autocast("cuda"):
        c1 = model.get_learned_conditioning(num_samples * [original_text])
        c2 = model.get_learned_conditioning(num_samples * [new_text])
        c3 = model.get_learned_conditioning(num_samples * [opt.c3])
        start_code = torch.randn([num_samples, 4, 64, 64], device=device)
        start_code = start_code.half()
        uc = model.get_learned_conditioning(num_samples * [""])

        sampler.sample_optimize_intrinsic(
            S=50,
            conditioning1=c1,
            conditioning2=c2,
            conditioning3=c3,
            batch_size=num_samples,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=7.5,
            unconditional_conditioning=uc,
            eta=0.0,
            x_T=start_code,
            input_image=noised_image_encode,
            noise_save_path=None,
            lambda_t=lambda_t,
            lambda_save_path=lambda_save_path,
            image_save_path=image_save_path,
            original_text=original_text,
            new_text=new_text,
            otext=original_text,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--c1",
        type=str,
        nargs="?",
        default="A photo of person",
        help="The text to synthesize original image.",
    )
    parser.add_argument(
        "--c2",
        type=str,
        nargs="?",
        default="A photo of person, smiling",
        help="The text modifies from c1, containing target attribute.",
    )
    parser.add_argument(
        "--c3",
        type=str,
        nargs="?",
        default="A photo of person, smiling",
        help="The text modifies from c1, containing target attribute.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="?",
        default=42,
        help="The seed. Particularly useful to control original image.",
    )
    parser.add_argument(
        "--lambda_t_star",
        type=int,
        nargs="?",
        default=20,
        help="lambda initialization.",
    )
    parser.add_argument(
        "--lambda_t_default_1",
        type=float,
        nargs="?",
        default=1.0,
        help="lambda initialization.",
    )
    parser.add_argument(
        "--lambda_t_default_2",
        type=float,
        nargs="?",
        default=0.0,
        help="lambda initialization.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/disentangle",
    )
    
    opt = parser.parse_args()
    final_disentangle_attributes(opt)
