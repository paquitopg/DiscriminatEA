import os
import argparse
import torch
from tqdm import tqdm
from transformers import AlbertTokenizer, AlbertModel, logging
logging.set_verbosity_warning()
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class TypeBertEmbedding:
    def __init__(self):
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
        self.tokenizer = tokenizer
        self.model = AlbertModel.from_pretrained("albert-base-v2")
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def embed_batch(self, sentences, batch_size=32):
        """Process sentences in batches for efficiency"""
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            if torch.cuda.is_available():
                for key in inputs.keys():
                    inputs[key] = inputs[key].cuda()
            with torch.no_grad():
                outputs = self.model(**inputs).last_hidden_state
                batch_emb = torch.mean(outputs, dim=1)
                embeddings.append(batch_emb.detach().cpu())
        return torch.cat(embeddings, dim=0)


def read_type_ids(data_dir):
    """Read type_ids mapping: type_id -> type_name"""
    type_ids_path = os.path.join(data_dir, "type_ids")
    type_id_to_name = {}
    with open(type_ids_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tid_str, tname = line.strip().split("\t", 1)
                type_id_to_name[int(tid_str)] = tname.replace("_", " ").replace(u"\xa0", "").strip()
    return type_id_to_name


def read_entity_types(file_path):
    """Read ent_types file: ent_id -> type_id"""
    mapping = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eid_str, tid_str = line.strip().split("\t")
                mapping[int(eid_str)] = int(tid_str)
    return mapping


def embed_type_ids(data_dir, type_id_to_name):
    """Embed each unique type_id once and save to type_ids_emb"""
    extractor = TypeBertEmbedding()
    type_ids = sorted(type_id_to_name.keys())
    type_names = [type_id_to_name[tid] for tid in type_ids]
    type_embeddings = extractor.embed_batch(type_names)

    out_path = os.path.join(data_dir, "type_ids_emb")
    with open(out_path, "w", encoding="utf-8") as fw:
        for i, tid in enumerate(type_ids):
            vec = type_embeddings[i].numpy()
            vec_str = ",".join([f"{v:.8f}" for v in vec.tolist()])
            fw.write(f"{tid}\t{vec_str}\n")


def compute_type_compatibility_matrix(data_dir, dim=64):
    """Compute type-to-type similarity matrix (T x T) from type_ids embeddings and save as type_type_sim.npy."""
    print("Computing type-to-type similarity matrix...")

    # Load type embeddings (type_id -> vector)
    type_ids_emb_path = os.path.join(data_dir, "type_ids_emb")
    if not os.path.exists(type_ids_emb_path):
        raise FileNotFoundError(f"Missing {type_ids_emb_path}. Run embedding first.")
    df_types = pd.read_table(type_ids_emb_path, sep="\t", header=None)
    type_vecs = df_types[1].str.split(",", expand=True).astype(float).values

    # Optional dimensionality reduction
    n_samples, n_features = type_vecs.shape
    if n_features > dim:
        from sklearn.decomposition import PCA
        n_components = min(dim, n_samples, n_features)
        if n_components >= 1:
            pca = PCA(n_components=n_components, svd_solver="full")
            type_vecs = pca.fit_transform(type_vecs)

    # Normalize and compute cosine similarity
    type_vecs = type_vecs / (np.linalg.norm(type_vecs, axis=1, keepdims=True) + 1e-12)
    type_type_sim = np.clip(type_vecs @ type_vecs.T, -1.0, 1.0)

    # Save T x T matrix
    out_path = os.path.join(data_dir, "type_type_sim.npy")
    np.save(out_path, type_type_sim)
    print(f"Type-to-type similarity matrix shape: {type_type_sim.shape}")

    return type_type_sim


def main(data: str):
    data_dir = os.path.join("data", data)

    required = ["type_ids", "ent_types_1", "ent_types_2"]
    for fn in required:
        if not os.path.exists(os.path.join(data_dir, fn)):
            raise FileNotFoundError(f"Missing required file: {os.path.join(data_dir, fn)}")

    # 1) Embed each unique type once
    type_id_to_name = read_type_ids(data_dir)
    embed_type_ids(data_dir, type_id_to_name)

    # 2) Compute entity-level compatibility matrix from type embeddings
    compute_type_compatibility_matrix(data_dir)

    print("Type embedding processing complete!")


def create_sample_type_files(data_dir):
    """Create minimal sample files (type_ids and ent_types_1/2) for testing."""
    type_ids = [
        "0\tPerson",
        "1\tCompany",
        "2\tLocation"
    ]
    ent_types_1 = [
        "0\t0",
        "1\t1",
        "2\t0",
        "3\t2"
    ]
    ent_types_2 = [
        "0\t0",
        "1\t1",
        "2\t0",
        "3\t1",
        "4\t2"
    ]
    with open(os.path.join(data_dir, "type_ids"), "w") as f:
        f.write("\n".join(type_ids))
    with open(os.path.join(data_dir, "ent_types_1"), "w") as f:
        f.write("\n".join(ent_types_1))
    with open(os.path.join(data_dir, "ent_types_2"), "w") as f:
        f.write("\n".join(ent_types_2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sample_dataset")
    args = parser.parse_args()
    main(args.dataset)
