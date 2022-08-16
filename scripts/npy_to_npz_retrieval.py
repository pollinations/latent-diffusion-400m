import numpy as np
import argparse


def convert_npy_to_npz(inverted_embeddings_path, target_path):
    inverted_embeddings = np.load(inverted_embeddings_path)
    number_of_embeds = inverted_embeddings.shape[0]
    patch_coords = np.repeat(np.array([[0, 0, 256, 256]]), number_of_embeds, axis=0)
    img_ids = np.arange(number_of_embeds)
    np.savez(target_path, embedding=inverted_embeddings, img_id=img_ids, patch_coords=patch_coords)
    print(f"Number of embeddings: {number_of_embeds}")
    print(f"Saved to {target_path}")
    print(f"Done!")
    return target_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", type=str)
    parser.add_argument("--npz_path", type=str, default="rdm_db.npz")
    args = parser.parse_args()
    assert args.npy_path.endswith(".npy"), "npy_path must end with .npy"
    assert args.npz_path.endswith(".npz"), "npz_path must end with .npz"

    print(f"Converting {args.npy_path} to {args.npz_path}")

    convert_npy_to_npz(args.npy_path, args.npz_path)