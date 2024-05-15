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
from pytorch_lightning import seed_everything
from sklearn.decomposition import PCA
import cv2
import numpy as np
from einops import rearrange

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



def PCAright(condition, index):
    condition = condition[0,:,:].cpu().numpy()
    X_mean = np.mean(condition, axis=0)
    dataMat = condition - X_mean
    covMat = np.mat(np.cov(dataMat, rowvar=0, bias=True))
    eigVal, eigVect = np.linalg.eig(covMat)
    eigValInd = np.argsort(eigVal)
    eigValInd = eigValInd[-index:-(index+1):-1]  # 取前N个较大的特征值
    main_eigVect = eigVect[:, eigValInd]  # *N维投影矩阵
    new_dataMat = dataMat * main_eigVect  # 投影变换后的新矩阵
    
    main_eigVect = torch.tensor(main_eigVect).cuda().unsqueeze(0)
    new_dataMat = torch.tensor(new_dataMat).cuda().unsqueeze(0)
    return new_dataMat.real.to(torch.float32)
    
def PCAleft(condition, index):
    condition = condition[0,:,:].cpu().numpy()
    X_mean = np.mean(condition, axis=0)
    dataMat = condition - X_mean
    covMat = np.mat(np.cov(dataMat, rowvar=1, bias=True))
    eigVal, eigVect = np.linalg.eig(covMat)
    eigValInd = np.argsort(eigVal)
    eigValInd = eigValInd[-index:-(index+1):-1]  # 取前N个较大的特征值
    main_eigVect = eigVect[eigValInd, :]  # *N维投影矩阵
    new_dataMat = main_eigVect * dataMat   # 投影变换后的新矩阵
        
    main_eigVect = torch.tensor(main_eigVect).cuda().unsqueeze(0)
    new_dataMat = torch.tensor(new_dataMat).cuda().unsqueeze(0)
    return new_dataMat.real.to(torch.float32)
    
def SVD(condition, k):
    condition = condition[0,:,:].cpu().numpy()
    U, Sigma, VT = np.linalg.svd(condition)
    new_dataMat = U[:,:k].dot(np.diag(Sigma[:k])).dot(VT[:k,:])
    new_dataMat = torch.tensor(new_dataMat).cuda().unsqueeze(0)
    return new_dataMat
    


def final_disentangle_attributes(opt):
    steps = 50  # DDIM, PLMS Sampling steps in stable-diffusion
    seed = opt.seed
    seed_everything(seed)
    original_text = opt.c1
    new_text = opt.c2

    image_save_path = opt.outdir
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path, exist_ok=True)
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


        if opt.mode == "swap":
            a = 1
            c1[:,4:5,:] = a * c2[:,4:5,:] + (1-a) * c1[:,4:5,:]     # 4 is the index of the meaningful token you want to swap
        elif opt.mode == "fader":
            c1[:,4:5,:] = 1.1 * c1[:,4:5,:]
        elif opt.mode == "PCA":
            #----------PCA_right---------#
            PCA_right = PCAright(c1, index=1)     # [1, 77, 1]
            PCA_right_repeat = torch.repeat_interleave(PCA_right, 768, dim=2)
            c1[:,:,0:1] += 0.1 * PCA_right
            #c1 -= 0.001 * PCA_right_repeat

            #----------PCA_left---------#
            # PCA_left = PCAleft(c1, index=1)       # [1, 1, 768]
            # PCA_left_repeat = torch.repeat_interleave(PCA_left, 77, dim=1)
            # c1 += 0.05 * PCA_left_repeat
        else:
            raise ValueError(f"Cannot find {code/Disentanglement} mode, mode type has to be within 'swap', 'fader', or 'PCA' ")


        start_code = torch.randn([num_samples, 4, 64, 64], device=device)
        start_code = start_code.half()
        uc = model.get_learned_conditioning(num_samples * [""])

        img = sampler.sample_optimize_intrinsic(
            S=50,
            conditioning=c1,
            batch_size=num_samples,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=7.5,
            unconditional_conditioning=uc,
            eta=0.0,
            x_T=start_code,
            input_image=noised_image_encode,
            noise_save_path=None,
            original_text=original_text,
            new_text=new_text,
            otext=original_text,
        )
        img = model.decode_first_stage(img)
        img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
        img = img.cpu().permute(0, 2, 3, 1).permute(0, 3, 1, 2)
        # save image
        with torch.no_grad():
            x_sample = 255.0 * rearrange(img[0].detach().cpu().numpy(), "c h w -> h w c")
            imgsave = Image.fromarray(x_sample.astype(np.uint8))
            imgsave.save(os.path.join(image_save_path, "{}.png".format(original_text)))

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
        "--mode",
        type=str,
        nargs="?",
        default="swap",
        choices=["swap", "fader", "PCA"],
        help="Operation mode on the text embedding",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="?",
        default=42,
        help="The seed. Particularly useful to control original image.",
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
