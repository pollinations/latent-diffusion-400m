import sys

sys.path.append("/taming-transformers")
# sys.path.append("src/clip")
import glob
import os
import tempfile
import time
from itertools import islice
from multiprocessing import cpu_count
from typing import Iterator, List, Union

import numpy as np
import scann
import torch
import torch.nn as nn
from cog import BasePredictor, Input, Path
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import make_grid
from tqdm import tqdm, trange

from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.encoders.modules import FrozenClipImageEmbedder, FrozenCLIPTextEmbedder
from ldm.util import instantiate_from_config, parallel_data_prefetch


class Searcher(object):
    def __init__(self, database, retriever_version="ViT-L/14"):
        assert database in DATABASES
        # self.database = self.load_database(database)
        self.database_name = database
        self.searcher_savedir = f"data/rdm/searchers/{self.database_name}"
        self.database_path = f"data/rdm/retrieval_databases/{self.database_name}"
        self.retriever = self.load_retriever(version=retriever_version)
        self.database = {"embedding": [], "img_id": [], "patch_coords": []}
        self.load_database()
        self.load_searcher()

    def train_searcher(self, k, metric="dot_product", searcher_savedir=None):

        print("Start training searcher")
        searcher = scann.scann_ops_pybind.builder(
            self.database["embedding"]
            / np.linalg.norm(self.database["embedding"], axis=1)[:, np.newaxis],
            k,
            metric,
        )
        self.searcher = searcher.score_brute_force().build()
        print("Finish training searcher")

        if searcher_savedir is not None:
            print(f'Save trained searcher under "{searcher_savedir}"')
            os.makedirs(searcher_savedir, exist_ok=True)
            self.searcher.serialize(searcher_savedir)

    def load_single_file(self, saved_embeddings):
        compressed = np.load(saved_embeddings)
        self.database = {key: compressed[key] for key in compressed.files}
        print("Finished loading of clip embeddings.")

    def load_multi_files(self, data_archive):
        out_data = {key: [] for key in self.database}
        for d in tqdm(
            data_archive,
            desc=f"Loading datapool from {len(data_archive)} individual files.",
        ):
            for key in d.files:
                out_data[key].append(d[key])

        return out_data

    def load_database(self):
        print(f'Load saved patch embedding from "{self.database_path}"')
        file_content = glob.glob(os.path.join(self.database_path, "*.npz"))

        if len(file_content) == 1:
            self.load_single_file(file_content[0])
        elif len(file_content) > 1:
            data = [np.load(f) for f in file_content]
            prefetched_data = parallel_data_prefetch(
                self.load_multi_files,
                data,
                n_proc=min(len(data), cpu_count()),
                target_data_type="dict",
            )

            self.database = {
                key: np.concatenate([od[key] for od in prefetched_data], axis=1)[0]
                for key in self.database
            }
        else:
            raise ValueError(
                f'No npz-files in specified path "{self.database_path}" is this directory existing?'
            )

        print(
            f'Finished loading of retrieval database of length {self.database["embedding"].shape[0]}.'
        )

    def load_retriever(
        self,
        version="ViT-L/14",
    ):
        model = FrozenClipImageEmbedder(model=version)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        return model

    def load_searcher(self):
        print(
            f"load searcher for database {self.database_name} from {self.searcher_savedir}"
        )
        self.searcher = scann.scann_ops_pybind.load_searcher(self.searcher_savedir)
        print("Finished loading searcher.")

    def search(self, x, k):
        if self.searcher is None and self.database["embedding"].shape[0] < 2e4:
            self.train_searcher(
                k
            )  # quickly fit searcher on the fly for small databases
        assert self.searcher is not None, "Cannot search with uninitialized searcher"
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if len(x.shape) == 3:
            x = x[:, 0]
        query_embeddings = x / np.linalg.norm(x, axis=1)[:, np.newaxis]

        start = time.time()
        nns, distances = self.searcher.search_batched(
            query_embeddings, final_num_neighbors=k
        )
        end = time.time()

        out_embeddings = self.database["embedding"][nns]
        out_img_ids = self.database["img_id"][nns]
        out_pc = self.database["patch_coords"][nns]

        out = {
            "nn_embeddings": out_embeddings
            / np.linalg.norm(out_embeddings, axis=-1)[..., np.newaxis],
            "img_ids": out_img_ids,
            "patch_coords": out_pc,
            "queries": x,
            "exec_time": end - start,
            "nns": nns,
            "q_embeddings": query_embeddings,
        }

        return out

    def __call__(self, x, n):
        return self.search(x, n)


DATABASES = [
    "prompt_engineering",
]


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
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval().half()
    return model


class Predictor(BasePredictor):
    def setup(self):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        config = OmegaConf.load(f"configs/retrieval-augmented-diffusion/768x768.yaml")
        model = load_model_from_config(config, f"/content/models/rdm/rdm768x768/model.ckpt")
        self.model = model.to(self.device)
        self.clip_text_encoder = FrozenCLIPTextEmbedder("ViT-L/14", device="cpu")
        self.searcher = Searcher("prompt_engineering", retriever_version="ViT-L/14")
        self.sampler = PLMSSampler(self.model)
        self.outdir = Path(tempfile.mkdtemp())

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompts: str = Input(
            default="",
            description="model will try to generate this text. use newlines to generate multiple prompts",
        ),
        ddim_steps: int = Input(
            default=50, description="number of ddim sampling steps"
        ),
        ddim_eta: float = Input(
            default=0.0,
            description="ddim eta (eta=0.0 corresponds to deterministic sampling",
        ),
        H: int = Input(default=768, description="image height, in pixel space"),
        W: int = Input(default=768, description="image width, in pixel space"),
        n_samples: int = Input(
            default=3,
            description="how many samples to produce for each given prompt. A.k.a batch size",
        ),
        n_rows: int = Input(
            default=0, description="rows in the grid (default: n_samples)"
        ),
        scale: float = Input(
            default=5.0,
            description="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        ),
        knn: int = Input(
            default=10,
            description="The number of included neighbors, only applied when --use_neighbors=True",
            ge=1,
            le=20,
        ),
    ) -> List[Path]:
        assert len(prompts) > 0, "no prompts provided"

        # paths
        n_rows = n_rows if n_rows > 0 else n_samples
        print(f"sampling scale for cfg is {scale:.2f}")

        cond_clip_embed = self.clip_text_encoder.encode(prompts)
        cond_clip_embed = cond_clip_embed.to(self.model.device)

        uncond_clip_embed = None

        nn_dict = self.searcher(cond_clip_embed, knn)
        sample_conditioning = torch.cat(
            [
                cond_clip_embed.to(self.device),
                torch.from_numpy(nn_dict["nn_embeddings"]).to(self.device),
            ],
            dim=1,
        )
        if scale != 1.0:
            # uncond_clip_embed = torch.zeros_like(cond_clip_embed)
            uncond_clip_embed = torch.zeros_like(sample_conditioning)

        # TODO refactor to collect all samples in batches
        with self.model.ema_scope():
            shape = [
                16,
                H // 16,
                W // 16,
            ]  # note: currently hardcoded for f16 model
            samples_ddim, _ = self.sampler.sample(
                S=ddim_steps,
                conditioning=sample_conditioning,
                batch_size=cond_clip_embed.shape[0],
                shape=shape,
                verbose=True,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uncond_clip_embed,
                eta=ddim_eta,
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
                pil_image = Image.fromarray(generation.astype(np.uint8))
                pil_image.save(x_sample_target_path, "png")
                generation_paths.append(Path(x_sample_target_path))
        return generation_paths