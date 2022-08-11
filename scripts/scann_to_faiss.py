from autofaiss import build_index
import numpy as np

embedding = None
for npz_i in range(1,5):
    npz_path = "rdm/retrieval_databases/openimages/2000000x768-part_" + str(npz_i) + ".npz"
    npz_i_obj = np.load(npz_path)
    if embedding is None:
        embedding = npz_i_obj['embedding']
    else:
        embedding = np.concatenate((embedding, npz_i_obj['embedding']), axis=0)

del npz_i_obj

build_index(embeddings=embedding, index_path="faiss_index/knn.index",
            index_infos_path="faiss_index/index_infos.json", max_index_memory_usage="30G",
            current_memory_available="40G")