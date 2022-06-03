# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import argparse
import gc
import os

import cog
import numpy as np
import open_clip
import torch
from cog import BasePredictor
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import make_grid


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )

        def load_model_from_config(config, ckpt, verbose=False):
            print(f"Loading model from {ckpt}")
            pl_sd = torch.load(ckpt, map_location="cuda:0")
            sd = pl_sd["state_dict"]
            model = instantiate_from_config(config.model)
            m, u = model.load_state_dict(sd, strict=False)
            if len(m) > 0 and verbose:
                print("missing keys:")
                print(m)
            if len(u) > 0 and verbose:
                print("unexpected keys:")
                print(u)

            model = model.half().cuda()
            model.eval()
            return model

        config = OmegaConf.load(
            "/latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        )
        model = load_model_from_config(config, "/content/models/ldm-model.ckpt")

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = model.to(device)

    def predict(
        self,
        Prompt: str = cog.Input(description="Your text prompt.", default=""),
        Steps: int = cog.Input(
            description="Number of steps to run the model", default=100
        ),
        ETA: int = cog.Input(description="Can be 0 or 1", default=1),
        Samples_in_parallel: int = cog.Input(description="Batch size", default=4),
        Diversity_scale: float = cog.Input(
            description="As a rule of thumb, higher values of scale produce better samples at the cost of a reduced output diversity.",
            default=10.0,
        ),
        Width: int = cog.Input(description="Width", default=256),
        Height: int = cog.Input(description="Height", default=256),
    ) -> None:
        """Run a single prediction on the model"""
        Prompts = Prompt

        Iterations = 1
        output_path = "/outputs"
        PLMS_sampling = True

        os.system(f"rm -rf /content/steps")
        os.system(f"mkdir -p /content/steps")

        frames = []

        def save_img_callback(pred_x0, i):
            # print(pred_x0)
            frame_id = len(frames)
            x_samples_ddim = self.model.decode_first_stage(pred_x0)
            imgs = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            grid = imgs
            # grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            rows = len(imgs)
            # check if rows is quadratic and if yes take the square root
            height = int(rows**0.5)
            grid = make_grid(imgs, nrow=height)
            # to image
            grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
            step_out = os.path.join("/content/steps", f"aaa_{frame_id:04}.png")
            Image.fromarray(grid.astype(np.uint8)).save(step_out)

            if frame_id % 10 == 0:
                progress_out = os.path.join(output_path, "aaa_progress.png")
                Image.fromarray(grid.astype(np.uint8)).save(progress_out)
            frames.append(frame_id)

        def run(opt):
            torch.cuda.empty_cache()
            gc.collect()
            if opt.plms:
                opt.ddim_eta = 0
                sampler = PLMSSampler(self.model)
            else:
                sampler = DDIMSampler(self.model)

            os.makedirs(opt.outdir, exist_ok=True)
            outpath = opt.outdir

            sample_path = os.path.join(outpath, "samples")
            os.makedirs(sample_path, exist_ok=True)

            all_samples = list()
            samples_ddim, x_samples_ddim = None, None
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    with self.model.ema_scope():
                        uc = None
                        if opt.scale > 0:
                            uc = self.model.get_learned_conditioning(
                                opt.n_samples * [""]
                            )
                        for prompt in opt.prompts:
                            print(prompt)
                            for n in range(opt.n_iter):
                                c = self.model.get_learned_conditioning(
                                    opt.n_samples * [prompt]
                                )
                                shape = [4, opt.H // 8, opt.W // 8]
                                samples_ddim, _ = sampler.sample(
                                    S=opt.ddim_steps,
                                    conditioning=c,
                                    batch_size=opt.n_samples,
                                    shape=shape,
                                    verbose=False,
                                    img_callback=save_img_callback,
                                    unconditional_guidance_scale=opt.scale,
                                    unconditional_conditioning=uc,
                                    eta=opt.ddim_eta,
                                    x_T=samples_ddim,
                                )

                                x_samples_ddim = self.model.decode_first_stage(
                                    samples_ddim
                                )
                                x_samples_ddim = torch.clamp(
                                    (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                                )
                                all_samples.append(x_samples_ddim)
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, "n b c h w -> (n b) c h w")
                rows = opt.n_samples
                # check if rows is quadratic and if yes take the square root
                height = int(rows**0.5)
                grid = make_grid(grid, nrow=height)
                # to image
                grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'zzz_{prompt.replace(" ", "-")}.png'))
                # save individual images
                for n, x_sample in enumerate(all_samples[0]):
                    x_sample = x_sample.squeeze()
                    x_sample = 255.0 * rearrange(
                        x_sample.cpu().numpy(), "c h w -> h w c"
                    )
                    prompt_filename = prompt.replace(" ", "-")
                    Image.fromarray(x_sample.astype(np.uint8)).save(
                        os.path.join(
                            output_path, f"{output_path}/yyy_{prompt_filename}_{n}.png"
                        )
                    )

        os.system(f"rm {output_path}/aaa_*.png")

        args = argparse.Namespace(
            prompts=Prompts.split("->"),
            outdir=output_path,
            ddim_steps=Steps,
            ddim_eta=ETA,
            n_iter=Iterations,
            W=Width,
            H=Height,
            n_samples=Samples_in_parallel,
            scale=Diversity_scale,
            plms=PLMS_sampling,
        )
        run(args)

        # last_frame=!ls -w1 -t /content/steps/*.png | head -1
        # last_frame = last_frame[0]
        # !cp -v $last_frame /content/steps/aaa_0000.png
        # !cp -v $last_frame /content/steps/aaa_0001.png
        encoding_options = "-c:v libx264 -crf 20 -preset slow -vf format=yuv420p -c:a aac -movflags +faststart"
        os.system(
            f"ffmpeg -y -r 10 -i /content/steps/aaa_%04d.png {encoding_options} {output_path}/zzz_output.mp4"
        )
