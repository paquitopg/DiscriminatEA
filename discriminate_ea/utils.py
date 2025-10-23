import os
import random
import numpy as np
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


def fixed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_embeddings(data_path, add_noise, noise_ratio, use_structure=True, use_time=False, use_types=True):
    """Load embeddings and types for entity alignment.

    Returns:
        ent_name_emb (np.ndarray)
        ent_dw_emb (np.ndarray or None)
        ent_time_emb (np.ndarray or None)
        ent_types (torch.LongTensor or None): combined KG1+KG2 type IDs per entity
        type_compatibility_matrix (np.ndarray or None): (kg1_size x kg2_size)
    """
    ent_name_emb, ent_dw_emb, ent_time_emb, ent_types, type_compatibility_matrix = None, None, None, None, None
    
    ### load name embeddings
    kg1_name_emb = np.loadtxt(os.path.join(data_path, "ent_1_emb_64.txt"))
    kg2_name_emb = np.loadtxt(os.path.join(data_path, "ent_2_emb_64.txt"))
    ent_name_emb = np.array(kg1_name_emb.tolist() + kg2_name_emb.tolist())
    print(f"read entity name embedding shape: {ent_name_emb.shape}")
    
    if add_noise:
        # Add noise to embeddings
        sample_list = [i for i in range(ent_name_emb.shape[1])]
        mask_id = random.sample(sample_list, int(ent_name_emb.shape[1] * noise_ratio))
        ent_name_emb[:, mask_id] = 0
    
    ### load structure embeddings
    if use_structure:
        ent_dw_emb = np.loadtxt(os.path.join(data_path, "deep_emb.txt"))
        print(f"read entity deepwalk emb shape: {ent_dw_emb.shape}")
    
    ### load time embeddings
    if use_time:
        ent_time_emb = np.array(load_ent_time_matrix(data_path))
        print(f"read entity time embedding shape: {ent_time_emb.shape}")

    ### load entity types and type compatibility
    if use_types:
        ent_types, type_mapping = load_entity_types(data_path)
        type_compatibility_matrix = load_type_type_sim(data_path)
        if type_compatibility_matrix is not None:
            print(f"type-to-type similarity matrix shape: {type_compatibility_matrix.shape}")
    else:
        ent_types, type_mapping = None, None
        type_compatibility_matrix = None
    
    return ent_name_emb, ent_dw_emb, ent_time_emb, ent_types, type_compatibility_matrix

# Add this function to utils.py (around line 50, after load_embeddings function)

def load_entity_types(data_path):
    """Load entity type mappings for both KGs"""
    # Load type definitions
    type_mapping = {}
    type_ids_file = os.path.join(data_path, "type_ids")
    if os.path.exists(type_ids_file):
        with open(type_ids_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    type_id, type_name = line.strip().split('\t')
                    type_mapping[int(type_id)] = type_name
        
        print(f"Loaded {len(type_mapping)} entity types: {type_mapping}")
    else:
        print("Warning: type_ids file not found, using default types")
        type_mapping = {0: "Unknown"}
    
    # Load entity-type mappings for KG1
    ent_types_1 = {}
    ent_types_1_file = os.path.join(data_path, "ent_types_1")
    if os.path.exists(ent_types_1_file):
        with open(ent_types_1_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    ent_id, type_id = line.strip().split('\t')
                    ent_types_1[int(ent_id)] = int(type_id)
        print(f"Loaded {len(ent_types_1)} entity types for KG1")
    else:
        print("Warning: ent_types_1 file not found")
        ent_types_1 = {}
    
    # Load entity-type mappings for KG2
    ent_types_2 = {}
    ent_types_2_file = os.path.join(data_path, "ent_types_2")
    if os.path.exists(ent_types_2_file):
        with open(ent_types_2_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    ent_id, type_id = line.strip().split('\t')
                    ent_types_2[int(ent_id)] = int(type_id)
        print(f"Loaded {len(ent_types_2)} entity types for KG2")
    else:
        print("Warning: ent_types_2 file not found")
        ent_types_2 = {}
    
    # Determine KG sizes from ent_ids files if available, else from mappings
    ent_ids_1_path = os.path.join(data_path, "ent_ids_1")
    ent_ids_2_path = os.path.join(data_path, "ent_ids_2")
    if os.path.exists(ent_ids_1_path):
        with open(ent_ids_1_path, 'r', encoding='utf-8') as f:
            kg1_count = len([ln for ln in f if ln.strip()])
    else:
        kg1_count = max(ent_types_1.keys()) + 1 if ent_types_1 else 0
    if os.path.exists(ent_ids_2_path):
        with open(ent_ids_2_path, 'r', encoding='utf-8') as f:
            kg2_count = len([ln for ln in f if ln.strip()])
    else:
        kg2_count = max(ent_types_2.keys()) + 1 if ent_types_2 else 0
    
    combined_types = []
    for i in range(kg1_count):
        combined_types.append(ent_types_1.get(i, 0))  # Default to type 0 if not found
    for i in range(kg2_count):
        combined_types.append(ent_types_2.get(i, 0))  # Default to type 0 if not found
    
    # Convert to tensor for model use
    combined_types = torch.tensor(combined_types, dtype=torch.long)
    
    print(f"Combined entity types: {len(combined_types)} entities")
    print(f"Type distribution: {dict(Counter(combined_types.tolist()))}")
    
    return combined_types, type_mapping


def load_type_type_sim(data_path):
    """Load type-to-type similarity matrix (T x T)."""
    path = os.path.join(data_path, "type_type_sim.npy")
    if os.path.exists(path):
        return np.load(path)
    print("Warning: type_type_sim.npy not found; type-aware features disabled")
    return None


def type_aware_candidate_blocking_by_types(type_type_sim, ent_types, kg1_size, kg2_size, threshold=0.3, top_k=10):
    """Blocking using type-to-type similarity and per-entity type IDs.

    Args:
        typsse_type_sim (np.ndarray): T x T similarity
        ent_types (torch.LongTensor): length (kg1_size+kg2_size), type_id per entity
        kg1_size, kg2_size (int)
    """
    candidate_pairs = []
    ent_types_np = ent_types.cpu().numpy() if isinstance(ent_types, torch.Tensor) else ent_types

    for kg1_ent_id in range(kg1_size):
        t1 = ent_types_np[kg1_ent_id]
        type_sim_row = type_type_sim[t1]
        kg2_type_ids = ent_types_np[kg1_size: kg1_size + kg2_size]
        sims = type_sim_row[kg2_type_ids]

        above = np.where(sims >= threshold)[0]
        if len(above) > 0:
            order = np.argsort(sims[above])[::-1]
            top = above[order[:top_k]]
            for idx in top:
                candidate_pairs.append((kg1_ent_id, int(idx)))
        # If no entities meet the threshold, don't add any candidates for this entity

    print(f"Type-aware candidate blocking: {len(candidate_pairs)} pairs")
    return candidate_pairs


def compute_type_aware_loss_weights(alignment_pairs, type_sim, 
                                  kg1_size, kg2_size, base_weight=1.0, type_weight=0.5, ent_types=None):
    """
    Compute loss weights based on type compatibility for training
    
    Args:
        alignment_pairs: Training alignment pairs
        type_compatibility_matrix: Type similarity matrix
        kg1_size, kg2_size: Sizes of the two KGs
        base_weight: Base weight for all pairs
        type_weight: Additional weight for type-compatible pairs
    
    Returns:
        loss_weights: Array of weights for each alignment pair
    """
    loss_weights = np.ones(len(alignment_pairs)) * base_weight
    
    for i, (kg1_ent_id, kg2_ent_id) in enumerate(alignment_pairs):
        if kg1_ent_id < kg1_size and kg2_ent_id >= kg1_size:
            kg2_matrix_id = kg2_ent_id - kg1_size
            if kg2_matrix_id < kg2_size:
                if ent_types is not None and type_sim is not None and type_sim.ndim == 2:
                    # T x T matrix path
                    t1 = ent_types[kg1_ent_id].item() if isinstance(ent_types, torch.Tensor) else ent_types[kg1_ent_id]
                    t2 = ent_types[kg1_size + kg2_matrix_id].item() if isinstance(ent_types, torch.Tensor) else ent_types[kg1_size + kg2_matrix_id]
                    sim = type_sim[t1, t2]
                    loss_weights[i] += type_weight * sim
                elif type_sim is not None and type_sim.ndim == 2 and type_sim.shape[0] == kg1_size:
                    # Backward compatibility: entity-level matrix
                    loss_weights[i] += type_weight * type_sim[kg1_ent_id, kg2_matrix_id]
    
    return loss_weights


def compute_kg_similarity(data_path, alignment_seed, gamma=0.5, use_structure=True, use_text=True):
    """
    Compute similarity between two knowledge graphs using structure and text similarity.
    
    The similarity is computed as:
    Similarity(KG_s, KG_t) = γ · Structure_Similarity(KG_s, KG_t) + (1-γ) · Text_Similarity(KG_s, KG_t)
    
    Where:
    - Structure_Similarity = (1/|S|) Σ cos(A_KG_s[e_s], A_KG_t[e_t]) for (e_s,e_t) ∈ S
    - Text_Similarity = (1/|S|) Σ cos(T_KG_s[e_s], T_KG_t[e_t]) for (e_s,e_t) ∈ S
    - S is the alignment seed (known aligned entities)
    - A_KG[e] is the 1-hop neighbor vector of entity e
    - T_KG[e] is the name embedding vector of entity e
    
    Args:
        data_path (str): Path to the data directory containing KG files
        alignment_seed (list): List of aligned entity pairs [(e_s, e_t), ...] where e_s is from KG1 and e_t is from KG2
        gamma (float): Weight for structure similarity (1-gamma for text similarity)
        use_structure (bool): Whether to compute structure similarity
        use_text (bool): Whether to compute text similarity
    
    Returns:
        dict: Dictionary containing overall similarity, structure similarity, and text similarity scores
    """
    print("Computing KG similarity...")
    
    # Load name embeddings for text similarity
    text_embeddings = None
    if use_text:
        kg1_name_emb = np.loadtxt(os.path.join(data_path, "ent_1_emb_64.txt"))
        kg2_name_emb = np.loadtxt(os.path.join(data_path, "ent_2_emb_64.txt"))
        text_embeddings = np.array(kg1_name_emb.tolist() + kg2_name_emb.tolist())
        print(f"Loaded text embeddings shape: {text_embeddings.shape}")
    
    # Load structural embeddings for structure similarity
    structure_embeddings = None
    if use_structure:
        structure_embeddings = np.loadtxt(os.path.join(data_path, "deep_emb.txt"))
        print(f"Loaded structural embeddings shape: {structure_embeddings.shape}")
    
    # Get KG sizes from entity ID files
    with open(os.path.join(data_path, "ent_ids_1"), "r") as f:
        kg1_size = len(f.readlines())
    with open(os.path.join(data_path, "ent_ids_2"), "r") as f:
        kg2_size = len(f.readlines())
    
    print(f"KG1 size: {kg1_size}, KG2 size: {kg2_size}")
    
    # Compute similarities
    structure_similarity = 0.0
    text_similarity = 0.0
    
    if len(alignment_seed) == 0:
        print("Warning: No alignment seed provided, returning zero similarities")
        return {
            'overall_similarity': 0.0,
            'structure_similarity': 0.0,
            'text_similarity': 0.0,
            'num_pairs': 0
        }
    
    # Compute structure similarity using density difference and degree coefficients
    if use_structure:
        # Load triples to build adjacency matrices
        triples, node_size, rel_size = load_triples(data_path, reverse=True)
        
        # Build adjacency matrices for both KGs
        adj_matrix_kg1 = np.zeros((kg1_size, kg1_size))
        adj_matrix_kg2 = np.zeros((kg2_size, kg2_size))
        
        # Fill adjacency matrices
        for h, r, t in triples:
            if h < kg1_size and t < kg1_size:  # KG1 internal edge
                adj_matrix_kg1[h, t] = 1
                adj_matrix_kg1[t, h] = 1  # Make symmetric
            elif h >= kg1_size and t >= kg1_size:  # KG2 internal edge
                h_kg2 = h - kg1_size
                t_kg2 = t - kg1_size
                adj_matrix_kg2[h_kg2, t_kg2] = 1
                adj_matrix_kg2[t_kg2, h_kg2] = 1  # Make symmetric
        
        # Compute graph densities
        # Density = actual_edges / possible_edges
        edges_kg1 = np.sum(adj_matrix_kg1) // 2  # Divide by 2 because matrix is symmetric
        edges_kg2 = np.sum(adj_matrix_kg2) // 2  # Divide by 2 because matrix is symmetric
        possible_edges_kg1 = kg1_size * (kg1_size - 1) // 2
        possible_edges_kg2 = kg2_size * (kg2_size - 1) // 2
        
        density_kg1 = edges_kg1 / possible_edges_kg1 if possible_edges_kg1 > 0 else 0
        density_kg2 = edges_kg2 / possible_edges_kg2 if possible_edges_kg2 > 0 else 0
        
        # Compute density similarity coefficient (0 if different, 1 if identical)
        # Use 1 - |density1 - density2| / max(density1, density2) to normalize
        if max(density_kg1, density_kg2) > 0:
            density_coeff = 1.0 - abs(density_kg1 - density_kg2) / max(density_kg1, density_kg2)
        else:
            density_coeff = 1.0  # Both densities are 0, so they're identical
        
        # Compute max degrees for normalization
        max_degree_kg1 = np.max(adj_matrix_kg1.sum(axis=1)) if np.max(adj_matrix_kg1.sum(axis=1)) > 0 else 1
        max_degree_kg2 = np.max(adj_matrix_kg2.sum(axis=1)) if np.max(adj_matrix_kg2.sum(axis=1)) > 0 else 1
        
        # Compute mean degree coefficient across alignment seed
        degree_coeffs = []
        for e_s, e_t in alignment_seed:
            # e_s is from KG1, e_t is from KG2 (global ID)
            if e_s < kg1_size and e_t >= kg1_size:
                # Convert KG2 global entity ID to local ID
                e_t_local = e_t - kg1_size
                if e_t_local < kg2_size:
                    # Compute degree similarity coefficient
                    degree_s = np.sum(adj_matrix_kg1[e_s])  # Degree of entity e_s in KG1
                    degree_t = np.sum(adj_matrix_kg2[e_t_local])  # Degree of entity e_t_local in KG2
                    
                    # Normalize degrees
                    norm_degree_s = degree_s / max_degree_kg1
                    norm_degree_t = degree_t / max_degree_kg2
                    
                    # Degree similarity coefficient (1 if same normalized degree, 0 if maximum difference)
                    degree_coeff = 1.0 - abs(norm_degree_s - norm_degree_t)
                    degree_coeffs.append(degree_coeff)
        
        mean_degree_coeff = np.mean(degree_coeffs) if degree_coeffs else 0.0
        
        # Structure similarity = density_coefficient * mean_degree_coefficient
        structure_similarity = density_coeff * mean_degree_coeff
        
        print(f"Graph densities - KG1: {density_kg1:.4f}, KG2: {density_kg2:.4f}")
        print(f"Density coefficient: {density_coeff:.4f}")
        print(f"Mean degree coefficient: {mean_degree_coeff:.4f}")
        print(f"Structure similarity: {structure_similarity:.4f}")
    
    # Compute text similarity
    if use_text and text_embeddings is not None:
        text_cosines = []
        for e_s, e_t in alignment_seed:
            # e_s is from KG1, e_t is from KG2 (global ID)
            if e_s < kg1_size and e_t >= kg1_size:
                text_vec_s = text_embeddings[e_s]  # KG1 entity embedding
                text_vec_t = text_embeddings[e_t]  # KG2 entity embedding (e_t is already the global ID)
                
                # Compute cosine similarity
                if np.linalg.norm(text_vec_s) > 0 and np.linalg.norm(text_vec_t) > 0:
                    cos_sim = np.dot(text_vec_s, text_vec_t) / (np.linalg.norm(text_vec_s) * np.linalg.norm(text_vec_t))
                    text_cosines.append(cos_sim)
                else:
                    text_cosines.append(0.0)
        
        text_similarity = np.mean(text_cosines) if text_cosines else 0.0
        print(f"Text similarity: {text_similarity:.4f}")
    
    # Compute overall similarity
    overall_similarity = gamma * structure_similarity + (1 - gamma) * text_similarity
    
    print(f"Overall KG similarity (gamma={gamma}): {overall_similarity:.4f}")
    
    return {
        'overall_similarity': overall_similarity,
        'structure_similarity': structure_similarity,
        'text_similarity': text_similarity,
        'num_pairs': len(alignment_seed)
    }

def neigh_ent_dict_gene(rel_triples, max_length, pad_id=None):
    """
    get one hop neighbor of entity
    return a dict, key = entity, value = (padding) neighbors of entity
    """
    neigh_ent_dict = dict()
    for i in range(pad_id):
        neigh_ent_dict[i] = []

    for h, _, t, _, _ in rel_triples:
        if h == t:
            continue
        neigh_ent_dict[h].append(t)
        neigh_ent_dict[t].append(h)
    #In order to get the maximum number of neighbors randomly for each entity
    for e in neigh_ent_dict.keys():
        np.random.shuffle(neigh_ent_dict[e])
        np.random.shuffle(neigh_ent_dict[e])
        np.random.shuffle(neigh_ent_dict[e])
    for e in neigh_ent_dict.keys():
        neigh_ent_dict[e] = neigh_ent_dict[e][:max_length]
    if pad_id != None:
        for e in neigh_ent_dict.keys():
            pad_list = [pad_id] * (max_length - len(neigh_ent_dict[e]))
            neigh_ent_dict[e] = neigh_ent_dict[e] + pad_list
    return neigh_ent_dict


def ent2attributeValues_gene(entid_list, att_datas, max_length, pad_value=None):
    """
    get attribute Values of entity
    return a dict, key = entity ,value = (padding) attribute_values of entity
    """
    ent2attributevalues = dict()
    for e in entid_list:
        ent2attributevalues[e] = []
    for e, _, l, _ in att_datas:
        ent2attributevalues[e].append(l)
    # random choose attributeValue to maxlength.
    for e in ent2attributevalues.keys():
        np.random.shuffle(ent2attributevalues[e])
    for e in ent2attributevalues.keys():
        ent2attributevalues[e] = ent2attributevalues[e][:max_length]
    if pad_value != None:
        for e in ent2attributevalues.keys():
            pad_list = [pad_value] * (max_length - len(ent2attributevalues[e]))
            ent2attributevalues[e] = ent2attributevalues[e] + pad_list
    return ent2attributevalues


def cos_sim_mat_generate(emb1, emb2, bs=128, cuda_num=0):
    """
    return cosine similarity matrix of embedding1(emb1) and embedding2(emb2)
    """
    array_emb1 = F.normalize(torch.FloatTensor(emb1), p=2,dim=1)
    array_emb2 = F.normalize(torch.FloatTensor(emb2), p=2,dim=1)
    res_mat = batch_mat_mm(array_emb1,array_emb2.t(),cuda_num,bs=bs)
    return res_mat


def batch_mat_mm(mat1, mat2, cuda_num, bs=128):
    #be equal to matmul, Speed up computing with GPU
    res_mat = []
    axis_0 = mat1.shape[0]
    for i in range(0,axis_0,bs):
        temp_div_mat_1 = mat1[i:min(i+bs,axis_0)].cuda(cuda_num)
        res = temp_div_mat_1.mm(mat2.cuda(cuda_num))
        res_mat.append(res.cpu())
    res_mat = torch.cat(res_mat,0)
    return res_mat


def batch_topk(mat, bs=128, topn = 50, largest = False, cuda_num = 0):
    #be equal to topk, Speed up computing with GPU
    res_score = []
    res_index = []
    axis_0 = mat.shape[0]
    for i in range(0,axis_0,bs):
        temp_div_mat = mat[i:min(i+bs,axis_0)].cuda(cuda_num)
        score_mat,index_mat =temp_div_mat.topk(topn,largest=largest)
        res_score.append(score_mat.cpu())
        res_index.append(index_mat.cpu())
    res_score = torch.cat(res_score,0)
    res_index = torch.cat(res_index,0)
    return res_score,res_index


def test_topk_res(index_mat):
    ent1_num,ent2_num = index_mat.shape
    topk_list = [0 for _ in range(ent2_num)]
    MRR = 0
    for i in range(ent1_num):
        for j in range(ent2_num):
            if index_mat[i][j].item() == i:
                MRR += ( 1 / (j+1) )
                for h in range(j,ent2_num):
                    topk_list[h]+=1
                break
    topk_list = [round(x/ent1_num,5) for x in topk_list]
    print("hit @ 1: {:.5f}    hit @10 : {:.5f}    ".format(topk_list[1 - 1], topk_list[10 - 1]), end="")
    if ent2_num >= 25:
        print("hit @ 25: {:.5f}    ".format(topk_list[25 - 1]), end="")
    if ent2_num >= 50:
        print("hit @ 50: {:.5f}    ".format(topk_list[50 - 1]), end="")
    print("")
    MRR/=ent1_num
    print("MRR:{:.5f}".format(MRR))


### function for loading datas
def load_triples(file_path, reverse=True):
    def reverse_triples(triples, rs):
        reversed_triples = np.zeros_like(triples)
        for i in range(len(triples)):
            reversed_triples[i, 0] = triples[i, 2]
            reversed_triples[i, 2] = triples[i, 0]
            reversed_triples[i, 1] = triples[i, 1] + rs
        return reversed_triples

    with open(os.path.join(file_path, "triples_1")) as fr:
        triples1 = fr.readlines()

    with open(os.path.join(file_path, "triples_2")) as fr:
        triples2 = fr.readlines()

    triples = np.array([line.strip().split("\t") for line in tqdm(triples1 + triples2, desc="load triples")]).astype(np.int64)
    node_size = max([np.max(triples[:, 0]), np.max(triples[:, 2])]) + 1
    rel_size = np.max(triples[:, 1]) + 1

    all_triples = np.concatenate([triples, reverse_triples(triples, rel_size)], axis=0)
    all_triples = np.unique(all_triples, axis=0)

    return all_triples, node_size, rel_size * 2 if reverse else rel_size


def load_alignments(file_path):
    alignments = []
    with open(file_path, "r", encoding="utf-8") as fr:
        for line in tqdm(fr.readlines(), desc="load alignments"):
            if line:
                e1, e2 = [int(e) for e in line.strip().split("\t")]
                alignments.append([e1, e2])
    return np.array(alignments)


def load_aligned_pair(data_path, ratio=0.3):
    if "sup_ent_ids" not in os.listdir(data_path):
        with open(os.path.join(data_path, "ref_ent_ids")) as f:
            aligned = f.readlines()
    else:
        with open(os.path.join(data_path, "ref_ent_ids")) as f:
            ref = f.readlines()
        with open(os.path.join(data_path, "sup_ent_ids")) as f:
            sup = f.readlines()
        aligned = ref + sup

    aligned = np.array([line.replace("\n", "").split("\t") for line in aligned]).astype(np.int64)
    np.random.shuffle(aligned)
    return aligned[:int(len(aligned) * ratio)], aligned[int(len(aligned) * ratio):]


def load_ent_time_matrix(data_path):
    ### load entities
    ent_1_list, ent_2_list = [], []
    with open(os.path.join(data_path, "ent_ids_1"), "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            if line:
                line = line.strip().split("\t")
                ent_1_list.append(int(line[0]))
    with open(os.path.join(data_path, "ent_ids_2"), "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            if line:
                line = line.strip().split("\t")
                ent_2_list.append(int(line[0]))
    ent_1_num, ent_2_num = len(ent_1_list), len(ent_2_list)

    ### get id-time dictionary
    time_dict = {}
    with open(os.path.join(data_path, "time_id"), "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            if line:
                line = line.strip().split("\t")
                if line[1] == "" or line[1][0] == "-":
                    line[1] = "~"
                time_dict[int(line[0])] = line[1]
                if line[1] == "~":
                    continue
                time_y = int(line[1].split("-")[0])

    ### get time embeddings
    def rel_time_cal(time_year, time_month):
        return (time_year - 1995) * 13 + time_month + 1
    time_emb_size = 1 + 27*13
    ent_1_emb = np.zeros([ent_1_num, time_emb_size])
    ent_2_emb = np.zeros([ent_2_num, time_emb_size])
    with open(os.path.join(data_path, "triples_1"), "r", encoding="utf-8") as fr:
        for line in tqdm(fr.readlines()):
            h, _, _, ts, te = [int(e) for e in line.strip().split("\t")]
            for tau in [ts, te]:
                if time_dict[tau] != "~":
                    time_y, time_m = [int(t) for t in time_dict[tau].split("-")]
                    if time_y < 1995:
                        ent_1_emb[h, 0] += 1
                    else:
                        ent_1_emb[h, rel_time_cal(time_y, time_m)] += 1
    with open(os.path.join(data_path, "triples_2"), "r", encoding="utf-8") as fr:
        for line in tqdm(fr.readlines()):
            time_y_s, time_m_s = 0, 0
            time_y_e, time_m_e = 0, 0
            h, r, t, ts, te = [int(e) for e in line.strip().split("\t")]
            if time_dict[ts] != "~":
                time_y_s, time_m_s = [int(t) for t in time_dict[ts].split("-")]
                if time_y < 1995:
                    ent_2_emb[h-ent_1_num, 0] += 1
                    time_y_s, time_m_s = 1995, 0
            if time_dict[te] != "~" and time_dict[ts] != "~":
                time_y_e, time_m_e = [int(t) for t in time_dict[te].split("-")]
                if time_y_e >= 1995:
                    ent_2_emb[h-ent_1_num, rel_time_cal(time_y_s, time_m_s):rel_time_cal(time_y_e, time_m_e)] += 1

    return np.array(ent_1_emb.tolist() + ent_2_emb.tolist())


### function for model training
def get_n_params(model:nn.Module):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def get_train_set(train_alignments, batch_size, node_size):
    negative_ratio = batch_size // len(train_alignments) + 1 # ==2 because batch_size == len(train_alignments)
    train_set = np.reshape(np.repeat(np.expand_dims(train_alignments, axis=0), axis=0, repeats=negative_ratio), newshape=(-1, 2))
    np.random.shuffle(train_set)
    train_set = train_set[:batch_size]
    train_set = np.concatenate([train_set, np.random.randint(0, node_size, train_set.shape)], axis=-1) # add negative samples but completely randomly!
    return train_set