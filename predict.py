import sys
from functools import lru_cache

import clip

sys.path.append("src/taming-transformers")
import tempfile
from typing import List, Optional

import numpy as np
import torch
from clip_retrieval.clip_back import ParquetMetadataProvider, load_index, meta_to_dict
from cog import BaseModel, BasePredictor, Input, Path
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
    device: str = "cpu",
):
    assert text is not None and len(text) > 0, "must provide text"
    tokens = clip.tokenize(text, truncate=True).to(device)
    clip_text_embed = clip_model.encode_text(tokens).to(device)
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
    print(f"Loaded model from {ckpt}")
    return model.half()


def map_to_metadata(
    indices, distances, num_images, metadata_provider, columns_to_return=["url"]
):
    results = []
    metas = metadata_provider.get(indices[:num_images])
    for key, (dist, ind) in enumerate(zip(distances, indices)):
        output = {}
        meta = None if key + 1 > len(metas) else metas[key]
        # convert_metadata_to_base64(meta) # TODO
        if meta is not None:
            output.update(meta_to_dict(meta))
        output["id"] = ind.item()
        output["similarity"] = dist.item()
        print(output)
        results.append(output)
    print(len(results))
    return results


def build_searcher(database_name: str):
    image_index_path = Path(f"data/rdm/searchers/{database_name}/image.index")
    assert image_index_path.exists(), f"database at {image_index_path} does not exist"
    print(f"Loading semantic index from {image_index_path}")

    metadata_path = Path(f"data/rdm/searchers/{database_name}/metadata")
    return {
        "image_index": load_index(
            str(image_index_path), enable_faiss_memory_mapping=True
        ),
        "metadata_provider": ParquetMetadataProvider(str(metadata_path))
        if metadata_path.exists()
        else None,
    }


class Predictor(BasePredictor):
    def __init__(self):
        self.searchers = None

    @torch.inference_mode()
    def setup(self):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        config = OmegaConf.load(f"configs/retrieval-augmented-diffusion/768x768.yaml")
        model = load_model_from_config(config, f"/content/models/rdm/rdm768x768/model.ckpt")
        self.model = model.to(self.device)
        print(f"Loaded 1.4M param Retrieval Augmented Diffusion model to {self.device}")

        use_jit = self.device.type.startswith("cuda")
        self.clip_model, _ = load_clip("ViT-L/14", use_jit=use_jit, device=self.device)
        print(f"Loaded clip model ViT-L/14 to {self.device} with use_jit={use_jit}")

        self.sampler = PLMSSampler(self.model)
        print("Using PLMS sampler")

        self.database_names = (
            [  # TODO you have to copy this to the predict arg any time it is changed.
                "prompt-engineer",
                "cars",
                "openimages",
                "faces",
                "simulacra",
                "coco",
                "pixelart",
                "food",
                "country211",
                "laion-aesthetic",
                "vaporwave",
                "pets",
                "emotes",
                "pokemon",
            ]
        )

        self.searchers = {
            database_name: build_searcher(database_name)
            for database_name in self.database_names
        }

    @torch.no_grad()
    def knn_search(self, query: torch.Tensor, num_results: int, database_name: str):
        # TODO rewrite this method
        print(f"Running knn search with {database_name}")
        knn_index = self.searchers[database_name]["image_index"]
        query = query.squeeze(0)
        query = query.cpu().detach().numpy().astype("float32")
        distances, indices, embeddings = knn_index.search_and_reconstruct(
            query, num_results
        )
        results = indices[0]  # first element is a list of indices
        nb_results = np.where(results == -1)[0]
        if len(nb_results) > 0:
            nb_results = nb_results[0]
        else:
            nb_results = len(results)
        result_indices = results[:nb_results]
        result_distances = distances[0][:nb_results]
        result_embeddings = embeddings[0][:nb_results]
        result_embeddings = torch.from_numpy(result_embeddings.astype("float32"))
        result_embeddings /= result_embeddings.norm(dim=-1, keepdim=True)
        result_embeddings = result_embeddings.unsqueeze(0)
        return result_distances, result_indices, result_embeddings

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            default="",
            description="model will try to generate this text.",
        ),
        database_name: str = Input(
            default="laion-aesthetic",
            description="Which database to use for the semantic search. Different databases have different capabilities.",
            choices=[
                "none",
                "prompt-engineer",
                "cars",
                "openimages",
                "faces",
                "simulacra",
                "coco",
                "pixelart",
                "food",
                "country211",
                "laion-aesthetic",
                "vaporwave",
                "pets",
                "emotes",
                "pokemon",
            ],
        ),
        prompt_scale: float = Input(
            default=5.0,
            description="Determines influence of your prompt on generation.",
        ),
        num_database_results: int = Input(
            default=10,
            description="The number of search results to guide the generation with. Using more will 'broaden' capabilities of the model at the risk of causing mode collapse or artifacting.",
            ge=1,
            le=20,
        ),
        num_generations: int = Input(
            default=1,
            description="Number of images to generate. Using more will make generation take longer.",  # TODO
        ),
        height: int = Input(
            default=768, description="Desired height of generated images."
        ),
        width: int = Input(
            default=768, description="Desired width of generated images."
        ),
        steps: int = Input(
            default=50,
            description="How many steps to run the model for. Using more will make generation take longer. 50 tends to work well.",
        ),
    ) -> List[Path]:
        self.outdir = Path(tempfile.mkdtemp())

        prompt_embedding = encode_text_with_clip_model(
            text=prompt, clip_model=self.clip_model, normalize=True, device=self.device
        )
        if database_name != "none":
            knn_distances, knn_indices, knn_embeddings = self.knn_search(
                query=prompt_embedding,
                num_results=num_database_results,
                database_name=database_name,
            )
            if self.searchers[database_name]["metadata_provider"] is not None:
                search_results = map_to_metadata(
                    indices=knn_indices,
                    distances=knn_distances,
                    num_images=num_database_results,
                    metadata_provider=self.searchers[database_name]["metadata_provider"],
                )
                for search_result in search_results:
                    print("-----------------------------------------------------")
                    print(f"caption: {search_result['caption']}")
                    print(f"url: {search_result['url']}")
                    print("-----------------------------------------------------")
            sample_conditioning = torch.cat(
                [
                    prompt_embedding.to(self.device),
                    knn_embeddings.to(self.device),
                ],
                dim=1,
            )
        else:
            sample_conditioning = prompt_embedding.to(self.device)
            
        if num_generations > 1:
            sample_conditioning = repeat(
                sample_conditioning, "1 k d -> b k d", b=num_generations
            )
        uncond_clip_embed = None
        if prompt_scale != 1.0:
            uncond_clip_embed = torch.zeros_like(sample_conditioning)
        with self.model.ema_scope():
            shape = [
                16,
                height // 16,
                width // 16,
            ]  # note: currently hardcoded for f16 model
            samples_ddim, _ = self.sampler.sample(
                S=steps,
                conditioning=sample_conditioning,
                batch_size=sample_conditioning.shape[0],
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=prompt_scale,
                unconditional_conditioning=uncond_clip_embed,
                # eta=0.0,
            )
            decoded_generations = self.model.decode_first_stage(samples_ddim)
        decoded_generations = torch.clamp(
            (decoded_generations + 1.0) / 2.0, min=0.0, max=1.0
        )

        generation_paths = []
        for idx, generation in enumerate(decoded_generations):
            generation = 255.0 * rearrange(generation.cpu().numpy(), "c h w -> h w c")
            x_sample_target_path = self.outdir.joinpath(f"sample_{idx:03d}.png")
            pil_image = Image.fromarray(generation.astype(np.uint8))
            pil_image.save(x_sample_target_path, "png")
            pil_image.save(f"sample_{idx:03d}.png")
            generation_paths.append(Path(x_sample_target_path))
        return generation_paths
