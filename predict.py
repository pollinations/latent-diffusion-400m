import sys
from functools import lru_cache

import clip

sys.path.append("src/taming-transformers")
import tempfile
from typing import List

import numpy as np
import torch
from clip_retrieval.clip_back import load_index
from cog import BasePredictor, Input, Path
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from torch import nn

from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config


@lru_cache(maxsize=None)  # cache the model, so we don't have to load it every time
def load_clip(clip_model="ViT-L/14", use_jit=True, device="cpu"):
    clip_model, preprocess = clip.load(clip_model, device=device, jit=use_jit)
    return clip_model, preprocess


@torch.no_grad()
def encode_text_with_clip_model(
    text: str,
    clip_model: nn.Module,
    normalize: bool = True,
):
    assert text is not None and len(text) > 0, "must provide text"
    tokens = clip.tokenize(text, truncate=True).to("cpu")
    clip_text_embed = clip_model.encode_text(tokens).to("cpu")
    if normalize:
        clip_text_embed /= clip_text_embed.norm(dim=-1, keepdim=True)
    if clip_text_embed.ndim == 2:
        clip_text_embed = clip_text_embed[:, None, :]
    return clip_text_embed


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


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
    model.eval()
    return model


# TODO
image_index_path = "data/rdm/searchers/LAION_Aesthetic_index/image.index"


class Predictor(BasePredictor):
    def setup(self):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"Using device {self.device}")

        self.clip_model, _ = load_clip("ViT-L/14", use_jit=False, device="cpu")
        print(f"Loaded clip model ViT-L/14 to CPU")

        self.image_index = load_index(
            image_index_path, enable_faiss_memory_mapping=True
        )
        print(f"Loaded clip-retrieval faiss index at {self.image_index}")

        self.outdir = Path(tempfile.mkdtemp())
        config = OmegaConf.load(f"configs/retrieval-augmented-diffusion/768x768.yaml")
        model = load_model_from_config(config, f"models/rdm/rdm768x768/model.ckpt")
        self.model = model.to(self.device)
        print("Loaded latent-diffusion model.")

        self.sampler = PLMSSampler(self.model)
        print("Using PLMS sampler")

    def knn_search(self, query: torch.Tensor, num_results: int):
        query = query.squeeze(0)
        query = query.cpu().detach().numpy().astype("float32")
        distances, indices, embeddings = self.image_index.search_and_reconstruct(
            query, num_results
        )
        print(f"results shape: {indices.shape}")
        results = indices[0]  # first element is a list of indices

        nb_results = np.where(results == -1)[0]
        if len(nb_results) > 0:
            nb_results = nb_results[0]
        else:
            nb_results = len(results)
        result_embeddings = embeddings[0][:nb_results]
        result_embeddings = torch.from_numpy(result_embeddings.astype("float32"))
        result_embeddings /= result_embeddings.norm(dim=-1, keepdim=True)
        result_embeddings = result_embeddings.unsqueeze(0)
        return result_embeddings

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            default="",
            description="model will try to generate this text.",
        ),
        num_lookup_images: int = Input(
            default=1,
            description="The number of included neighbors in the knn search",
            ge=1,
            le=20,
        ),
        num_generations: int = Input(
            default=4,
            description="how many times to repeat the query",  # TODO
        ),
        height: int = Input(default=768, description="image height, in pixel space"),
        widht: int = Input(default=768, description="image width, in pixel space"),
        steps: int = Input(
            default=50,
            description="how many steps to run the model for",
        ),
        scale: float = Input(
            default=5.0,
            description="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        ),
    ) -> List[Path]:
        clip_text_features = encode_text_with_clip_model(
            text=prompt, clip_model=self.clip_model, normalize=True
        )
        result_embeddings = self.knn_search(clip_text_features, num_lookup_images)
        sample_conditioning = torch.cat(
            [
                clip_text_features.to(self.device),
                result_embeddings.to(self.device),
            ],
            dim=1,
        )
        if num_generations > 1:
            sample_conditioning = repeat(
                sample_conditioning, "1 k d -> b k d", b=num_generations
            )
        uncond_clip_embed = None
        if scale != 1.0:
            uncond_clip_embed = torch.zeros_like(sample_conditioning)
        with self.model.ema_scope():
            shape = [
                16,
                height // 16,
                widht // 16,
            ]  # note: currently hardcoded for f16 model
            samples_ddim, _ = self.sampler.sample(
                S=steps,
                conditioning=sample_conditioning,
                batch_size=sample_conditioning.shape[0],
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uncond_clip_embed,
                # eta=0.0,
            )
            decoded_generations = self.model.decode_first_stage(samples_ddim)
            decoded_generations = torch.clamp(
                (decoded_generations + 1.0) / 2.0, min=0.0, max=1.0
            )

            generation_paths = []
            for idx, generation in enumerate(decoded_generations):
                generation = 255.0 * rearrange(
                    generation.cpu().numpy(), "c h w -> h w c"
                )
                x_sample_target_path = self.outdir.joinpath(f"sample_{idx:03d}.png")
                x_sample_target_path = f"sample_{idx:03d}.png"  # TODO
                pil_image = Image.fromarray(generation.astype(np.uint8))
                pil_image.save(x_sample_target_path, "png")
                generation_paths.append(Path(x_sample_target_path))
        return generation_paths
