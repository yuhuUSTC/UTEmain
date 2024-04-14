"""SAMPLING ONLY."""
import os
import torch
from torch import nn
import torchvision
import numpy as np
from tqdm import tqdm
from functools import partial
from PIL import Image
import shutil
from torch import optim
from tqdm.auto import tqdm
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler   
import torch.utils.checkpoint as checkpoint

from ldm.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
)
import clip
from einops import rearrange
import random
from models.facial_recognition.model_irse import Backbone


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode="bilinear", size=(224, 224), align_corners=False)
            target = self.transform(target, mode="bilinear", size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class DCLIPLoss(torch.nn.Module):
    def __init__(self):
        super(DCLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=16)

    def forward(self, image1, image2, text1, text2):
        text1 = clip.tokenize([text1]).to("cuda")
        text2 = clip.tokenize([text2]).to("cuda")
        image1 = image1.unsqueeze(0).cuda()
        image2 = image2.unsqueeze(0)
        image1 = self.avg_pool(self.upsample(image1))
        image2 = self.avg_pool(self.upsample(image2))
        image1_feat = self.model.encode_image(image1)
        image2_feat = self.model.encode_image(image2)
        text1_feat = self.model.encode_text(text1)
        text2_feat = self.model.encode_text(text2)
        d_image_feat = image1_feat - image2_feat
        d_text_feat = text1_feat - text2_feat
        similarity = torch.nn.CosineSimilarity()(d_image_feat, d_text_feat)
        return 1 - similarity


class PLMSSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(
        self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True
    ):
        if ddim_eta != 0:
            raise ValueError("ddim_eta must be 0 for PLMS")
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert (alphas_cumprod.shape[0] == self.ddpm_num_timesteps), "alphas have to be defined for each timestep"
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),)
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod.cpu())))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu())))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)),)

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=0.0,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer("ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def plms_sampling(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = (
                self.ddpm_num_timesteps
                if ddim_use_original_steps
                else self.ddim_timesteps
            )
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = (int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1)
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = (
            list(reversed(range(0, timesteps))) if ddim_use_original_steps else np.flip(timesteps))
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running PLMS Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="PLMS Sampler", total=total_steps)
        old_eps = []

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full(
                (b,),
                time_range[min(i + 1, len(time_range) - 1)],
                device=device,
                dtype=torch.long,
            )

            if mask is not None:
                assert x0 is not None
                # import ipdb; ipdb.set_trace()
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1.0 - mask) * img

            outs = self.p_sample_plms(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                old_eps=old_eps,
                t_next=ts_next,
            )
            img, pred_x0, e_t = outs
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            if index % 1 == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_plms(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        old_eps=None,
        t_next=None,
    ):
        b, *_, device = *x.shape, x.device

        def get_model_output(x, t):
            if (
                unconditional_conditioning is None
                or unconditional_guidance_scale == 1.0
            ):
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            return e_t

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (
            self.model.alphas_cumprod_prev
            if use_original_steps
            else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.model.ddim_sigmas_for_original_num_steps
            if use_original_steps
            else self.ddim_sigmas
        )

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full(
                (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
            )

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1.0 - a_prev - sigma_t ** 2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.0:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = get_model_output(x, t)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (
                55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]
            ) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t

    ###### Above are original stable-diffusion code ############

    ###### Encode Image ########################################

    @torch.no_grad()
    def sample_encode_save_noise(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        input_image=None,
        noise_save_path=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        assert conditioning is not None
        assert not isinstance(conditioning, dict)

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f"Data shape for PLMS sampling is {size}")

        samples, intermediates = self.plms_sampling_enc_save_noise(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            input_image=input_image,
            noise_save_path=noise_save_path,
        )
        return samples, intermediates

    @torch.no_grad()
    def plms_sampling_enc_save_noise(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        input_image=None,
        noise_save_path=None,
    ):
        device = self.model.betas.device

        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = (
                self.ddpm_num_timesteps
                if ddim_use_original_steps
                else self.ddim_timesteps
            )
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = (int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1)
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = (
            list(reversed(range(0, timesteps)))
            if ddim_use_original_steps
            else np.flip(timesteps)
        )
        time_range = list(range(0, timesteps)) if ddim_use_original_steps else timesteps
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running PLMS Sampling with {total_steps} timesteps")

        # iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
        iterator = tqdm(time_range[:-1], desc="PLMS Sampler", total=total_steps)
        old_eps = []

        for each_time in time_range:
            noised_image = self.model.q_sample(input_image, torch.tensor([each_time]).to(device))
            torch.save(noised_image, noise_save_path + "_image_time%d.pt" % (each_time))

        x0_loop = input_image.clone()
        alphas = (self.model.alphas_cumprod if ddim_use_original_steps else self.ddim_alphas)
        alphas_prev = (
            self.model.alphas_cumprod_prev
            if ddim_use_original_steps
            else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if ddim_use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.model.ddim_sigmas_for_original_num_steps
            if ddim_use_original_steps
            else self.ddim_sigmas
        )

        def get_model_output(x, t):
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, cond])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            return e_t

        def get_x_prev_and_pred_x0(e_t, index, curr_x0):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full(
                (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
            )

            # current prediction for x_0
            pred_x0 = (curr_x0 - sqrt_one_minus_at * e_t) / a_t.sqrt()

            a_t = torch.full((b, 1, 1, 1), alphas[index + 1], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index + 1], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index + 1], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index + 1], device=device)

            dir_xt = (1.0 - a_t - sigma_t ** 2).sqrt() * e_t

            x_prev = a_t.sqrt() * pred_x0 + dir_xt

            return x_prev, pred_x0

        for i, step in enumerate(iterator):
            index = i
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full(
                (b,),
                time_range[min(i + 1, len(time_range) - 1)],
                device=device,
                dtype=torch.long,
            )
            e_t = get_model_output(x0_loop, ts)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index, x0_loop)
            x0_loop = x_prev
        torch.save(x0_loop, noise_save_path + "_final_latent.pt")

        # Reconstruction
        img = x0_loop.clone()
        time_range = (
            list(reversed(range(0, timesteps)))
            if ddim_use_original_steps
            else np.flip(timesteps)
        )
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running PLMS Sampling with {total_steps} timesteps")
        iterator = tqdm(time_range, desc="PLMS Sampler", total=total_steps)
        old_eps = []
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full(
                (b,),
                time_range[min(i + 1, len(time_range) - 1)],
                device=device,
                dtype=torch.long,
            )

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1.0 - mask) * img

            outs = self.p_sample_plms_dec_save_noise(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                old_eps=old_eps,
                t_next=ts_next,
                input_image=input_image,
                noise_save_path=noise_save_path,
            )
            img, pred_x0, e_t = outs

            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_plms_dec_save_noise(
        self,
        x,
        c1,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        old_eps=None,
        t_next=None,
        input_image=None,
        noise_save_path=None,
    ):
        b, *_, device = *x.shape, x.device

        def get_model_output(x, t):
            if (unconditional_conditioning is None or unconditional_guidance_scale == 1.0):
                e_t = self.model.apply_model(x, t, c1)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c1])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            return e_t

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (
            self.model.alphas_cumprod_prev
            if use_original_steps
            else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.model.ddim_sigmas_for_original_num_steps
            if use_original_steps
            else self.ddim_sigmas
        )

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1.0 - a_prev - sigma_t ** 2).sqrt() * e_t
            time_curr = index * 20 + 1
            img_prev = torch.load(noise_save_path + "_image_time%d.pt" % (time_curr))
            noise = img_prev - a_prev.sqrt() * pred_x0 - dir_xt
            torch.save(noise, noise_save_path + "_time%d.pt" % (time_curr))

            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = get_model_output(x, t)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t

    
    
    
################# Encode Image End ###############################
    def p_sample_plms_sampling1(
        self,
        x,
        c1,
        c2,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        old_eps=None,
        t_next=None,
        input_image=None,
        optimizing_weight=None,
        noise_save_path=None,
    ):
        b, *_, device = *x.shape, x.device

        def optimize_model_output(x, t):
            # weight_for_pencil = torch.nn.Sigmoid()(optimizing_weight)
            # condition = weight_for_pencil * c1 + (1 - weight_for_pencil) * c2
            condition = optimizing_weight * c1 + (1 - optimizing_weight) * c2
            if (unconditional_conditioning is None or unconditional_guidance_scale == 1.0):
                e_t = self.model.apply_model(x, t, condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, condition])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            return e_t

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev)
        sqrt_one_minus_alphas = (self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas)
        sigmas = (self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas)

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1.0 - a_prev - sigma_t ** 2).sqrt() * e_t
            time_curr = index * 20 + 1
            if noise_save_path and index > 16:
                noise = torch.load(noise_save_path + "_time%d.pt" % (time_curr))[:1]
            else:
                noise = torch.zeros_like(dir_xt)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = optimize_model_output(x, t)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            # e_t_next = get_model_output(x_prev, t_next)
            e_t_next = optimize_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t


    def p_sample_plms_sampling(
        self,
        x,
        condition,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        old_eps=None,
        t_next=None,
        input_image=None,
        optimizing_weight=None,
        noise_save_path=None,
    ):
        b, *_, device = *x.shape, x.device

        def optimize_model_output(x, t, condition):
            if (unconditional_conditioning is None or unconditional_guidance_scale == 1.0):
                e_t = self.model.apply_model(x, t, condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, condition])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            return e_t

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev)
        sqrt_one_minus_alphas = (self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas)
        sigmas = (self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas)

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1.0 - a_prev - sigma_t ** 2).sqrt() * e_t
            time_curr = index * 20 + 1
            if noise_save_path and index > 16:
                noise = torch.load(noise_save_path + "_time%d.pt" % (time_curr))[:1]
            else:
                noise = torch.zeros_like(dir_xt)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = optimize_model_output(x, t, condition)
        if len(old_eps) == 0:
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = optimize_model_output(x_prev, t_next, condition)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t





    ################## Edit Input Image ###############################

    def sample_optimize_intrinsic_edit(
        self,
        S,
        batch_size,
        shape,
        conditioning1=None,
        conditioning2=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        input_image=None,
        noise_save_path=None,
        lambda_t=None,
        lambda_save_path=None,
        image_save_path=None,
        original_text=None,
        new_text=None,
        otext=None,
        noise_saved_path=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        assert conditioning1 is not None
        assert conditioning2 is not None

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f"Data shape for PLMS sampling is {size}")

        self.plms_sampling_optimize_intrinsic_edit(
            conditioning1,
            conditioning2,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            input_image=input_image,
            noise_save_path=noise_save_path,
            lambda_t=lambda_t,
            lambda_save_path=lambda_save_path,
            image_save_path=image_save_path,
            original_text=original_text,
            new_text=new_text,
            otext=otext,
            noise_saved_path=noise_saved_path,
        )
        return None

    def plms_sampling_optimize_intrinsic_edit(
        self,
        cond1,
        cond2,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        input_image=None,
        noise_save_path=None,
        lambda_t=None,
        lambda_save_path=None,
        image_save_path=None,
        original_text=None,
        new_text=None,
        otext=None,
        noise_saved_path=None,
    ):
        # Different from above, the intrinsic edit version needs
        device = self.model.betas.device

        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        img_clone = img.clone()

        if timesteps is None:
            timesteps = (self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps)
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = (int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1)
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = (list(reversed(range(0, timesteps))) if ddim_use_original_steps else np.flip(timesteps))

        weighting_parameter = lambda_t
        weighting_parameter.requires_grad = True
        from torch import optim

        optimizer = optim.Adam([weighting_parameter], lr=0.05)

        
        
        
        import numpy
        #aa = numpy.linspace(0, 0.5, num=6)
        aa = [76]
        print(aa)
        for a in aa:
            cond = cond1.clone()
            #cond[:,4:5,:] = a * cond1[:,4:5,:]
            cond[:,int(a):77,:] = cond2[:,int(a):77,:]
            with torch.no_grad():
                img = img_clone.clone()
                total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
                iterator = time_range
                old_eps = []
                for i, step in enumerate(iterator):
                    index = total_steps - i - 1
                    ts = torch.full((b,), step, device=device, dtype=torch.long)
                    ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long,)


                    outs = self.p_sample_plms_sampling(
                        img,
                        cond,
                        #cond2,
                        ts,
                        index=index,
                        use_original_steps=ddim_use_original_steps,
                        quantize_denoised=quantize_denoised,
                        temperature=temperature,
                        noise_dropout=noise_dropout,
                        score_corrector=score_corrector,
                        corrector_kwargs=corrector_kwargs,
                        unconditional_guidance_scale=unconditional_guidance_scale,
                        unconditional_conditioning=unconditional_conditioning,
                        old_eps=old_eps,
                        t_next=ts_next,
                        input_image=input_image,
                        optimizing_weight=torch.ones(50)[i], #   torch.ones(50)[i],  weighting_parameter[i], 
                        noise_save_path=noise_saved_path,
                    )
                    img, pred_x0, e_t = outs
                    old_eps.append(e_t)
                    if len(old_eps) >= 4:
                        old_eps.pop(0)
                img_temp = self.model.decode_first_stage(img)
                img_temp_ddim = torch.clamp((img_temp + 1.0) / 2.0, min=0.0, max=1.0)
                img_temp_ddim = img_temp_ddim.cpu().permute(0, 2, 3, 1).permute(0, 3, 1, 2)
                    # save image
                with torch.no_grad():
                    x_sample = 255.0 * rearrange(img_temp_ddim[0].detach().cpu().numpy(), "c h w -> h w c")
                    imgsave = Image.fromarray(x_sample.astype(np.uint8))
                    imgsave.save(image_save_path + "church{}.png".format(a))
            torch.cuda.empty_cache()
        
        
        
        
        
        
        
#         print("Original image")
#         with torch.no_grad():
#             img = img_clone.clone()
#             total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
#             iterator = time_range
#             old_eps = []
        
#             #a = 0.3
#             #cond1[:,4:5,:] = a * cond1[:,4:5,:] + (1-a) * cond2[:,4:5,:]
#             cond1[:,4:5,:] = 1.2 * cond1[:,4:5,:]
#             for i, step in enumerate(iterator):
#                 index = total_steps - i - 1
#                 ts = torch.full((b,), step, device=device, dtype=torch.long)
#                 ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long,)

                
#                 outs = self.p_sample_plms_sampling(
#                     img,
#                     cond1,
#                     #cond2,
#                     ts,
#                     index=index,
#                     use_original_steps=ddim_use_original_steps,
#                     quantize_denoised=quantize_denoised,
#                     temperature=temperature,
#                     noise_dropout=noise_dropout,
#                     score_corrector=score_corrector,
#                     corrector_kwargs=corrector_kwargs,
#                     unconditional_guidance_scale=unconditional_guidance_scale,
#                     unconditional_conditioning=unconditional_conditioning,
#                     old_eps=old_eps,
#                     t_next=ts_next,
#                     input_image=input_image,
#                     optimizing_weight=torch.ones(50)[i], #   torch.ones(50)[i],  weighting_parameter[i], 
#                     noise_save_path=noise_saved_path,
#                 )
#                 img, pred_x0, e_t = outs
#                 old_eps.append(e_t)
#                 if len(old_eps) >= 4:
#                     old_eps.pop(0)
#             img_temp = self.model.decode_first_stage(img)
#             img_temp_ddim = torch.clamp((img_temp + 1.0) / 2.0, min=0.0, max=1.0)
#             img_temp_ddim = img_temp_ddim.cpu().permute(0, 2, 3, 1).permute(0, 3, 1, 2)
#             # save image
#             with torch.no_grad():
#                 x_sample = 255.0 * rearrange(img_temp_ddim[0].detach().cpu().numpy(), "c h w -> h w c")
#                 imgsave = Image.fromarray(x_sample.astype(np.uint8))
#                 imgsave.save(image_save_path + "original.png")
#             readed_image = (torchvision.io.read_image(image_save_path + "original.png").float() / 255)
        
        
        print("Optimizing start")
        for epoch in tqdm(range(0)):
            img = img_clone.clone()
            total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
            iterator = time_range
            old_eps = []

            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                ts_next = torch.full(
                    (b,),
                    time_range[min(i + 1, len(time_range) - 1)],
                    device=device,
                    dtype=torch.long,
                )

                outs = self.p_sample_plms_sampling1(
                    img,
                    cond1,
                    cond2,
                    ts,
                    index=index,
                    use_original_steps=ddim_use_original_steps,
                    quantize_denoised=quantize_denoised,
                    temperature=temperature,
                    noise_dropout=noise_dropout,
                    score_corrector=score_corrector,
                    corrector_kwargs=corrector_kwargs,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    old_eps=old_eps,
                    t_next=ts_next,
                    input_image=input_image,
                    optimizing_weight=weighting_parameter[i],
                    noise_save_path=noise_saved_path,
                )
                img, pred_x0, e_t = outs
                old_eps.append(e_t)
                if len(old_eps) >= 4:
                    old_eps.pop(0)
            img_temp = self.model.decode_first_stage(img)
            img_temp_ddim = torch.clamp((img_temp + 1.0) / 2.0, min=0.0, max=1.0)
            img_temp_ddim = img_temp_ddim.cpu()

            
            ##save image
            with torch.no_grad():
                x_sample = 255.0 * rearrange(img_temp_ddim[0].detach().cpu().numpy(), "c h w -> h w c")
                imgsave = Image.fromarray(x_sample.astype(np.uint8))
                imgsave.save(image_save_path + "/%d.png" % (epoch))

                
            loss1 = VGGPerceptualLoss()(img_temp_ddim[0], readed_image)
            loss2 = DCLIPLoss()(readed_image, img_temp_ddim[0].float().cuda(), otext, new_text)
            loss = 0.05 * loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # torch.save(weighting_parameter, lambda_save_path + "/weightingParam%d.pt" % (epoch))
            if epoch < 9:
                del img
            else:
                # save image
                with torch.no_grad():
                    x_sample = 255.0 * rearrange(img_temp_ddim[0].detach().cpu().numpy(), "c h w -> h w c")
                    imgsave = Image.fromarray(x_sample.astype(np.uint8))
                    imgsave.save(image_save_path + "/final.png")
                torch.save(weighting_parameter, lambda_save_path + "/weightingParam_final.pt")

            torch.cuda.empty_cache()
        shutil.rmtree("noise")
        return None

    ################ Edit Image End ######################

    ################ Disentangle #########################

    def sample_optimize_intrinsic(
        self,
        S,
        batch_size,
        shape,
        conditioning1=None,
        conditioning2=None,
        conditioning3=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        input_image=None,
        noise_save_path=None,
        lambda_t=None,
        lambda_save_path=None,
        image_save_path=None,
        original_text=None,
        new_text=None,
        otext=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        assert conditioning1 is not None
        assert conditioning2 is not None

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f"Data shape for PLMS sampling is {size}")

        self.plms_sampling_optimize_intrinsic(
            conditioning1,
            conditioning2,
            conditioning3,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            input_image=input_image,
            noise_save_path=noise_save_path,
            lambda_t=lambda_t,
            lambda_save_path=lambda_save_path,
            image_save_path=image_save_path,
            original_text=original_text,
            new_text=new_text,
            otext=otext,
        )
        return None

    def PCA(self, condition, n_components):
        from sklearn.decomposition import PCA
        import cv2
        import numpy as np

        pca = PCA(n_components=n_components)
        condition = condition[0,:,:].cpu().numpy()
        condition = pca.fit_transform(condition)
        condition = torch.tensor(condition).cuda().unsqueeze(0)
        return condition

    def PCA_right(self, condition, n_components):
        condition = condition[0,:,:].cpu().numpy()
        X_mean = np.mean(condition, axis=0)
        dataMat = condition - X_mean
        covMat = np.mat(np.cov(dataMat, rowvar=0, bias=True))
        eigVal, eigVect = np.linalg.eig(covMat)
        eigValInd = np.argsort(eigVal)
        eigValInd = eigValInd[-10:-11:-1]  # 取前N个较大的特征值
        main_eigVect = eigVect[:, eigValInd]  # *N维投影矩阵
        new_dataMat = dataMat * main_eigVect  # 投影变换后的新矩阵
        
        main_eigVect = torch.tensor(main_eigVect).cuda().unsqueeze(0)
        new_dataMat = torch.tensor(new_dataMat).cuda().unsqueeze(0)
        return new_dataMat.real.to(torch.float32)
    
    def PCA_left(self, condition, n_components):
        condition = condition[0,:,:].cpu().numpy()
        X_mean = np.mean(condition, axis=0)
        dataMat = condition - X_mean
        covMat = np.mat(np.cov(dataMat, rowvar=1, bias=True))
        eigVal, eigVect = np.linalg.eig(covMat)
        eigValInd = np.argsort(eigVal)
        eigValInd = eigValInd[-3:-4:-1]  # 取前N个较大的特征值
        main_eigVect = eigVect[eigValInd, :]  # *N维投影矩阵
        new_dataMat = main_eigVect * dataMat   # 投影变换后的新矩阵
        
        main_eigVect = torch.tensor(main_eigVect).cuda().unsqueeze(0)
        new_dataMat = torch.tensor(new_dataMat).cuda().unsqueeze(0)
        return new_dataMat.real.to(torch.float32)
    
    def SVD(self, condition, k):
        condition = condition[0,:,:].cpu().numpy()
        U, Sigma, VT = np.linalg.svd(condition)
        new_dataMat = U[:,:k].dot(np.diag(Sigma[:k])).dot(VT[:k,:])
        new_dataMat = torch.tensor(new_dataMat).cuda().unsqueeze(0)
        return new_dataMat
    
    def condition(self, c1, c2, weighting_parameter):
        condition = weighting_parameter * c1 + (1 - weighting_parameter) * c2
        # PCA_condition = self.PCA(c1, n_components)
        # PCA_condition = torch.repeat_interleave(PCA_condition, 71, dim=1)
        # condition[:,:,0:1] += 0.1 * PCA_condition
        # condition[:,6:77,:] += 0.01 * PCA_condition
        # condition[:,4:5,:] = 0.94 * condition[:,4:5,:] + 0.06 * c2[:,4:5,:]

            
        # from scipy import spatial
        # for i in range(77):
        #     cos_sim = 1 - spatial.distance.cosine(condition[0,i,:].cpu().numpy(), PCA_condition[0,0,:].cpu().numpy())
        #     print(cos_sim)
        
        return condition
            
    def plms_sampling_optimize_intrinsic(
        self,
        cond1,
        cond2,
        cond3,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        input_image=None,
        noise_save_path=None,
        lambda_t=None,
        lambda_save_path=None,
        image_save_path=None,
        original_text=None,
        new_text=None,
        otext=None,
    ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        img_clone = img.clone()

        if timesteps is None:
            timesteps = (self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps)
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = (int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1)
            timesteps = self.ddim_timesteps[:subset_end]
        time_range = (list(reversed(range(0, timesteps))) if ddim_use_original_steps else np.flip(timesteps))
        #weighting_parameter = lambda_t
        weighting_parameter = torch.ones(77)
        weighting_parameter[4] = 0
        weighting_parameter.requires_grad = True
        optimizer = optim.Adam([weighting_parameter], lr=0.05)

#         import numpy
#         aa = numpy.linspace(0, 0.2, num=3)
#         print(aa)
#         for w in aa:
#             cond = cond1.clone()
#             cond_2 = cond2.clone()
#             #cond[:,2:3,:] *= w
#             #cond[:,5:6,:] = w * cond[:,5:6,:] + (1-w) * cond_2[:,5:6,:]
#             cond[:,8:77,:] = w * cond[:,8:77,:] + (1-w) * cond_2[:,8:77,:]
#             #PCA_right = self.PCA_right(cond, 1)
#             #cond[:,:,9:10] += a * PCA_right
            
#             with torch.no_grad():
#                 img = img_clone.clone()
#                 total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
#                 iterator = time_range
#                 old_eps = []
#                 with autocast():
#                     for i, step in enumerate(tqdm(iterator)):
#                         index = total_steps - i - 1
#                         ts = torch.full((b,), step, device=device, dtype=torch.long)
#                         ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long,)
#                         outs = self.p_sample_plms_sampling(
#                                 img,
#                                 cond,
#                                 ts,
#                                 index=index,
#                                 use_original_steps=ddim_use_original_steps,
#                                 quantize_denoised=quantize_denoised,
#                                 temperature=temperature,
#                                 noise_dropout=noise_dropout,
#                                 score_corrector=score_corrector,
#                                 corrector_kwargs=corrector_kwargs,
#                                 unconditional_guidance_scale=unconditional_guidance_scale,
#                                 unconditional_conditioning=unconditional_conditioning,
#                                 old_eps=old_eps,
#                                 t_next=ts_next,
#                                 input_image=input_image,
#                                 optimizing_weight=torch.ones(50)[i],
#                                 #optimizing_weight=weighting_parameter[i],
#                                 noise_save_path=noise_save_path,
#                         )
#                         img, pred_x0, e_t = outs
#                         old_eps.append(e_t)
#                         if len(old_eps) >= 4:
#                             old_eps.pop(0)
#                     img_temp = self.model.decode_first_stage(img)
#                     del img
#                     img_temp_ddim = torch.clamp((img_temp + 1.0) / 2.0, min=0.0, max=1.0)
#                     img_temp_ddim = img_temp_ddim.cpu().permute(0, 2, 3, 1).permute(0, 3, 1, 2)
#                     # save image
#                     with torch.no_grad():
#                         x_sample = 255.0 * rearrange(img_temp_ddim[0].detach().cpu().numpy(), "c h w -> h w c")
#                         imgsave = Image.fromarray(x_sample.astype(np.uint8))
#                         imgsave.save(image_save_path + "{}.png".format(w))
#             torch.cuda.empty_cache()





        w1=1
        w2=5
        #cond1[:,int(w1):int(w2),:] = 0
        
        #cond1[:,15:77,:] = cond2[:,15:77,:]
        #cond1[:,2:4,:] = 1.1 * cond1[:,2:4,:]
        #cond1[:,4:5,:] = a * cond2[:,4:5,:] + (1-a) * cond1[:,4:5,:]
        #PCA_right = self.PCA_right(cond1, 1)
        #PCA_left = self.PCA_right(cond1, 1)
        #PCA_left = torch.repeat_interleave(PCA_left, 71, dim=1)
        #PCA_right = torch.repeat_interleave(PCA_right, 768, dim=2)
        #cond1[:,:,2:3] -= 0.3 * PCA_right

        with torch.no_grad():
            img = img_clone.clone()
            total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
            iterator = time_range
            old_eps = []
            with autocast():
                for i, step in enumerate(tqdm(iterator)):
                    index = total_steps - i - 1
                    ts = torch.full((b,), step, device=device, dtype=torch.long)
                    ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long,)
                    outs = self.p_sample_plms_sampling(
                            img,
                            cond1,
                            #cond2,
                            ts,
                            index=index,
                            use_original_steps=ddim_use_original_steps,
                            quantize_denoised=quantize_denoised,
                            temperature=temperature,
                            noise_dropout=noise_dropout,
                            score_corrector=score_corrector,
                            corrector_kwargs=corrector_kwargs,
                            unconditional_guidance_scale=unconditional_guidance_scale,
                            unconditional_conditioning=unconditional_conditioning,
                            old_eps=old_eps,
                            t_next=ts_next,
                            input_image=input_image,
                            #optimizing_weight=torch.ones(50)[i],
                            optimizing_weight=weighting_parameter[i],
                            noise_save_path=noise_save_path,
                    )
                    img, pred_x0, e_t = outs
                    old_eps.append(e_t)
                    if len(old_eps) >= 4:
                        old_eps.pop(0)
                img_temp = self.model.decode_first_stage(img)
                del img
                img_temp_ddim = torch.clamp((img_temp + 1.0) / 2.0, min=0.0, max=1.0)
                img_temp_ddim = img_temp_ddim.cpu().permute(0, 2, 3, 1).permute(0, 3, 1, 2)
                # save image
                with torch.no_grad():
                    x_sample = 255.0 * rearrange(img_temp_ddim[0].detach().cpu().numpy(), "c h w -> h w c")
                    imgsave = Image.fromarray(x_sample.astype(np.uint8))
                    #imgsave.save(image_save_path + "\{}.png".format(original_text))
                    imgsave.save(os.path.join(image_save_path, "{}.png".format(original_text)))
                    #readed_image = (torchvision.io.read_image(image_save_path + "original.png").float() / 255)
        torch.cuda.empty_cache()

        
        
        
        
#         condition = torch.ones_like(cond100)
#         condition = torch.ones_like(cond1)
#         scaler = GradScaler()
#         for epoch in tqdm(range(20)):
#             print(epoch)
#             img = img_clone.clone()
#             total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
#             iterator = time_range
#             old_eps = []

            
#             optimizer.zero_grad()
#             with autocast():
#                 for i in range(77):
#                     condition[:,i,:] = weighting_parameter[i] * cond1[:,i,:] + (1 - weighting_parameter[i]) * cond2[:,i,:]
#                     #condition[:,i,:] = weighting_parameter[i] * cond1[:,i,:]
                
#                 for i, step in enumerate(tqdm(iterator)):
#                     index = total_steps - i - 1
#                     ts = torch.full((b,), step, device=device, dtype=torch.long)
#                     ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long,)
#                     outs = self.p_sample_plms_sampling(
#                             img,
#                             condition,
#                             ts,
#                             index=index,
#                             use_original_steps=ddim_use_original_steps,
#                             quantize_denoised=quantize_denoised,
#                             temperature=temperature,
#                             noise_dropout=noise_dropout,
#                             score_corrector=score_corrector,
#                             corrector_kwargs=corrector_kwargs,
#                             unconditional_guidance_scale=unconditional_guidance_scale,
#                             unconditional_conditioning=unconditional_conditioning,
#                             old_eps=old_eps,
#                             t_next=ts_next,
#                             input_image=input_image,
#                             optimizing_weight=torch.ones(50)[i],
#                             #optimizing_weight=weighting_parameter[i],
#                             noise_save_path=noise_save_path,
#                     )
#                     img, pred_x0, e_t = outs
#                     old_eps.append(e_t)
#                     if len(old_eps) >= 4:
#                         old_eps.pop(0)
#                 img_temp = self.model.decode_first_stage(img)
#                 del img
#                 img_temp_ddim = torch.clamp((img_temp + 1.0) / 2.0, min=0.0, max=1.0)
#                 img_temp_ddim = img_temp_ddim.cpu().permute(0, 2, 3, 1).permute(0, 3, 1, 2)
#                 # save image
#                 if epoch % 1 == 0:
#                     with torch.no_grad():
#                         x_sample = 255. * rearrange(img_temp_ddim[0].detach().cpu().numpy(), 'c h w -> h w c')
#                         imgsave = Image.fromarray(x_sample.astype(np.uint8))
#                         imgsave.save(image_save_path + "/%d.png"%(epoch))
#                         #torch.save(weighting_parameter,  lambda_save_path+"/weightingParam%d.pt"%(epoch))
                
#                 loss1 = VGGPerceptualLoss()(img_temp_ddim[0], readed_image)
#                 loss2 = DCLIPLoss()(readed_image, img_temp_ddim[0].float().cuda(), otext, new_text)
#                 print(loss1)
#                 print(loss2)
#                 loss = 1 * loss1 + loss2  # 0.05 or 0.03. Adjust according to attributes on scenes or people.
                
#                 #loss = DCLIPLoss()(readed_image, img_temp_ddim[0].float().cuda(), otext, "curly hair")
            
#             loss.backward(retain_graph=True)
#             # loss.backward()
#             optimizer.step()

#             weighting_parameter.data.clamp_(0,1)
            
#             #weighting_parameter = torch.sigmoid(weighting_parameter)
#             #weighting_parameter = torch.clamp(weighting_parameter, 0, 1)
#             #weighting_parameter.requires_grad=True
#             # # Scales loss. 为了梯度放大.
#             # torch.nan_to_num(loss)
#             # scaler.scale(loss).backward()
#             # # scaler.step() 首先把梯度的值unscale回来. 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重, 否则，忽略step调用，从而保证权重不更新（不被破坏）
#             # scaler.step(optimizer)
#             # # 准备着，看是否要增大scaler
#             # scaler.update()
#             print(weighting_parameter)

#             torch.cuda.empty_cache()

        return None


################ Disentangle End #########################

