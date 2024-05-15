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
    
def find_diff_indices(a, b):
    words_a = a.split()
    words_b = b.split()

    diff_indices = []
    min_len = min(len(words_a), len(words_b))

    for i in range(min_len):
        if words_a[i] != words_b[i]:
            diff_indices.append(i+1)
    return diff_indices


def readprompt(prompt):
    with open(prompt, 'r') as file:
        content = file.readlines()

    source_prompts = []
    target_prompts = []
    seeds = []
    ddim_steps = []
    scales = []
    for line in content:
        if 'source_prompt:' in line:
            source_prompt = line.split('source_prompt:')[1].strip()
            source_prompts.append(source_prompt)
        if line.strip().startswith("-"):
            target_prompts.append(line.strip()[2:])  # 提取prompt并添加到prompts列表中（去掉"- "）
        if 'seed:' in line:
            seed = line.split('seed:')[1].strip()
            seeds.append(int(seed))
        if 'scale:' in line:
            scale = line.split('scale:')[1].strip()
            scales.append(float(scale))
        if ' ddim_steps:' in line:
            ddim_step = line.split('ddim_steps:')[1].strip()
            ddim_steps.append(int(ddim_step))
    return source_prompts, target_prompts, seeds, ddim_steps, scales



def final_disentangle_attributes(opt):
    source_prompts, target_prompts, seeds, ddim_steps, scales =readprompt(opt.prompt)
    
    image_save_path = opt.outdir
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    ####SD V1.4
    model = load_model_from_config(OmegaConf.load("/mnt/workspace/workgroup/yuhu/code/Disentanglement/configs/stable-diffusion/v1-inference.yaml"),
                                   "/mnt/workspace/workgroup/yuhu/model/stable-diffusion/model.ckpt",).to(device)
    
    for param in model.parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.float16)

    sampler = PLMSSampler(model)
    num_samples = 1  # batch size


    noised_image_encode = None
    shape = [4, 64, 64]
    with torch.autocast("cuda"):
        for i in range(len(source_prompts)):
            original_text = source_prompts[i]

            for j in range(5):
                seed = seeds[i]
                step = ddim_steps[i]
                scale = scales[i]
                seed_everything(seed)
                index_of_target = i * 5 + j 
                new_text = target_prompts[index_of_target]

                c1 = model.get_learned_conditioning(num_samples * [original_text])
                c2 = model.get_learned_conditioning(num_samples * [new_text])
                
                index = find_diff_indices(original_text, new_text)

                for k in range(len(index)):
                    print(index[k])
                    a = 1.0
                    c1[:,index[k]:index[k]+1,:] = a * c2[:,index[k]:index[k]+1,:] + (1-a) * c1[:,index[k]:index[k]+1,:]
                
                start_code = torch.randn([num_samples, 4, 64, 64], device=device)
                start_code = start_code.half()
                uc = model.get_learned_conditioning(num_samples * [""])

                img = sampler.sample_optimize_intrinsic(
                    S=step,
                    conditioning=c1,
                    batch_size=num_samples,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=scale,
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
                    imgsave.save(os.path.join(image_save_path, f"{i}_{j+1}.png"))



                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="data",
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
