import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from discriminate_ea.model import DiscriminatEA
from discriminate_ea.utils import load_embeddings, load_alignments, get_train_set, type_aware_candidate_blocking_by_types
from discriminate_ea.CSLS_ import eval_alignment_by_sim_mat

def cosine_similarity(x, y):
    """Compute cosine similarity between two sets of vectors."""
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)
    return np.dot(x_norm, y_norm.T)

def get_top_k_candidates(scores, k=10):
    """Get top-k candidates for each entity."""
    top_k_indices = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    top_k_scores = np.take_along_axis(scores, top_k_indices, axis=1)
    return top_k_indices, top_k_scores

def get_type_aware_top_k_candidates(scores, type_compatible_pairs, k=10):
    """Get top-k candidates for each entity, considering only type-compatible pairs."""
    top_k_indices = []
    top_k_scores = []
    
    # Create a set of type-compatible pairs for faster lookup
    compatible_set = set(type_compatible_pairs)
    
    for i in range(scores.shape[0]):  # For each KG1 entity
        # Get scores for this entity
        entity_scores = scores[i]
        
        # Find type-compatible candidates for this entity
        compatible_candidates = []
        for j in range(scores.shape[1]):  # For each KG2 entity
            if (i, j) in compatible_set:
                compatible_candidates.append((j, entity_scores[j]))
        
        # Sort by score (descending) and take top-k
        compatible_candidates.sort(key=lambda x: x[1], reverse=True)
        compatible_candidates = compatible_candidates[:k]
        
        # Extract indices and scores
        if len(compatible_candidates) >= k:
            indices = [c[0] for c in compatible_candidates]
            scores_for_entity = [c[1] for c in compatible_candidates]
        else:
            # If not enough compatible candidates, pad with -1 indices and -inf scores
            indices = [c[0] for c in compatible_candidates] + [-1] * (k - len(compatible_candidates))
            scores_for_entity = [c[1] for c in compatible_candidates] + [-np.inf] * (k - len(compatible_candidates))
        
        top_k_indices.append(indices)
        top_k_scores.append(scores_for_entity)
    
    return np.array(top_k_indices), np.array(top_k_scores)

def add_noise_to_model_embeddings(model, noise_ratio, noise_type, device):
    """Add white noise to specific embedding types in the model."""
    
    print(f"Adding {noise_type} noise with ratio {noise_ratio} to model embeddings...")
    
    # Store original norms for comparison
    original_norms = {}
    if noise_type in ["name", "both"] and hasattr(model, 'ent_name_emb') and model.ent_name_emb is not None:
        original_norms['name'] = torch.norm(model.ent_name_emb).item()
    if noise_type in ["structure", "both"] and hasattr(model, 'ent_dw_emb') and model.ent_dw_emb is not None:
        original_norms['structure'] = torch.norm(model.ent_dw_emb).item()
    
    with torch.no_grad():
        if noise_type in ["name", "both"]:
            # Add noise to name embeddings
            if hasattr(model, 'ent_name_emb') and model.ent_name_emb is not None:
                noise = torch.randn_like(model.ent_name_emb) * noise_ratio
                model.ent_name_emb.data += noise.to(device)
                print(f"Added noise to name embeddings: shape {model.ent_name_emb.shape}")
        
        if noise_type in ["structure", "both"]:
            # Add noise to structural embeddings
            if hasattr(model, 'ent_dw_emb') and model.ent_dw_emb is not None:
                noise = torch.randn_like(model.ent_dw_emb) * noise_ratio
                model.ent_dw_emb.data += noise.to(device)
                print(f"Added noise to structural embeddings: shape {model.ent_dw_emb.shape}")
        
    
    # Print noise impact
    if original_norms:
        print("Noise impact on embedding norms:")
        if 'name' in original_norms:
            new_norm = torch.norm(model.ent_name_emb).item()
            change_pct = ((new_norm - original_norms['name']) / original_norms['name']) * 100
            print(f"  Name embeddings: {original_norms['name']:.4f} → {new_norm:.4f} ({change_pct:+.2f}%)")
        if 'structure' in original_norms:
            new_norm = torch.norm(model.ent_dw_emb).item()
            change_pct = ((new_norm - original_norms['structure']) / original_norms['structure']) * 100
            print(f"  Structure embeddings: {original_norms['structure']:.4f} → {new_norm:.4f} ({change_pct:+.2f}%)")

def evaluate_candidate_predictions(candidates_file, ground_truth, hit_k=[1, 5, 10]):
    """Evaluate predictions using the generated candidate file."""
    # Load ground truth pairs
    gt_dict = {}
    for kg1_id, kg2_id in ground_truth:
        gt_dict[int(kg1_id)] = int(kg2_id)
    
    # Load candidates
    candidates_dict = {}
    with open(candidates_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                kg1_id = int(parts[0])
                kg2_id = int(parts[1])
                if kg1_id not in candidates_dict:
                    candidates_dict[kg1_id] = []
                candidates_dict[kg1_id].append(kg2_id)
    
    # Evaluate
    hits = [0] * len(hit_k)
    mrr_sum = 0.0
    total_pairs = 0
    
    for kg1_id, true_kg2_id in gt_dict.items():
        if kg1_id in candidates_dict:
            candidates = candidates_dict[kg1_id]
            
            # Find rank of true candidate
            try:
                rank = candidates.index(true_kg2_id) + 1  # 1-based ranking
                mrr_sum += 1.0 / rank
                
                # Check hits@k
                for i, k in enumerate(hit_k):
                    if rank <= k:
                        hits[i] += 1
                        
            except ValueError:
                # True candidate not found in predictions
                pass
                
            total_pairs += 1
    
    # Compute final metrics
    acc = [hits[i] / total_pairs for i in range(len(hit_k))] if total_pairs > 0 else [0.0] * len(hit_k)
    mrr = mrr_sum / total_pairs if total_pairs > 0 else 0.0
    
    return acc, mrr

def main():
    # 1. Parse arguments
    parser = argparse.ArgumentParser(description="Generate alignment candidates using a trained DiscriminatEA model.")
    parser.add_argument("--data", type=str, required=True, help="Name of the dataset (folder under data/)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model .pt file")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--output", type=str, default="alignment_candidates.txt", help="Output file for candidates")
    parser.add_argument("--num-candidates", type=int, default=10, help="Number of candidates per entity (default: 10)")
    parser.add_argument("--scoring", type=str, default="cosine", choices=["cosine", "l2", "csls"], 
                       help="Scoring method (default: cosine)")
    parser.add_argument("--no_structure", action="store_true", default=False)
    parser.add_argument("--no_types", action="store_true", help="Disable type-aware candidate blocking")
    parser.add_argument("--type_threshold", type=float, default=0.0, help="Type similarity threshold for candidate blocking (0.0 = disabled)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate predictions if ground truth available")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing (default: 1000)")
    
    # Noise perturbation arguments
    parser.add_argument("--add_noise", action="store_true", help="Add white noise to embeddings at inference")
    parser.add_argument("--noise_ratio", type=float, default=0.1, help="Noise intensity (standard deviation of Gaussian noise)")
    parser.add_argument("--noise_type", type=str, choices=["name", "structure", "both"], default="both", 
                       help="Which embeddings to add noise to: name, structure, or both")
    parser.add_argument("--no_name", action="store_true", help="Model was trained without name embeddings")
    args = parser.parse_args()

    data = args.data
    use_structure = not args.no_structure
    use_name = not args.no_name
    use_types = not args.no_types

    # 2. Set device
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"

    # 3. Load dataset embeddings
    data_path = os.path.join("data", args.data)
    ent_name_emb, ent_dw_emb, ent_types, type_compatibility_matrix = load_embeddings(data_path, add_noise=False, noise_ratio=0.0, use_structure=True, use_types=use_types)

    # 4. Recreate model architecture
    model = DiscriminatEA(
        ent_name_emb=ent_name_emb,
        ent_dw_emb=ent_dw_emb,
        ent_types=ent_types,
        use_name=use_name,
        use_structure=use_structure,
        emb_size=64,
        structure_size=8,
        device=device
    )
    model = model.to(device)

    # 5. Load trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # 5.5. Add noise to embeddings if requested
    if args.add_noise:
        add_noise_to_model_embeddings(model, args.noise_ratio, args.noise_type, device)

    # 6. Get entity embeddings
    with torch.no_grad():
        # Get embeddings for all entities
        all_entities = torch.arange(ent_name_emb.shape[0]).to(device)
        features = model()  # This returns [num_entities, emb_size]
        
        # Get actual entity counts from entity ID files (much smaller than embedding files)
        with open(os.path.join(data_path, "ent_ids_1"), "r") as f:
            n_kg1 = len(f.readlines())
        with open(os.path.join(data_path, "ent_ids_2"), "r") as f:
            n_kg2 = len(f.readlines())
        print(f"KG1 entities: {n_kg1}, KG2 entities: {n_kg2}")
        
        # Split embeddings based on actual counts
        kg1_embeddings = features[:n_kg1].cpu().numpy()
        kg2_embeddings = features[n_kg1:n_kg1+n_kg2].cpu().numpy()

    # 7. Compute similarity matrix
    print("Computing similarity matrix...")
    if args.scoring == "cosine":
        similarity_matrix = cosine_similarity(kg1_embeddings, kg2_embeddings)
    elif args.scoring == "l2":
        # Convert L2 distance to similarity (higher is better)
        distances = np.linalg.norm(kg1_embeddings[:, None, :] - kg2_embeddings[None, :, :], axis=2)
        similarity_matrix = -distances  # Convert distance to similarity
    else:  # CSLS
        # Implement CSLS scoring here
        similarity_matrix = cosine_similarity(kg1_embeddings, kg2_embeddings)
    
    # 7.5. Get type-compatible candidate pairs if type blocking is enabled
    type_compatible_pairs = None
    if args.type_threshold > 0.0 and use_types and (ent_types is not None) and (type_compatibility_matrix is not None):
        print(f"Applying type-aware candidate blocking with threshold {args.type_threshold}")
        type_compatible_pairs = type_aware_candidate_blocking_by_types(
            type_compatibility_matrix, ent_types, n_kg1, n_kg2, 
            threshold=args.type_threshold, top_k=args.num_candidates
        )
        print(f"Type-aware candidate blocking found {len(type_compatible_pairs)} type-compatible pairs")
    elif args.type_threshold > 0.0:
        print("Type-aware candidate blocking requested but type information not available")

    # 8. Generate top-k candidates
    print(f"Generating top-{args.num_candidates} candidates...")
    if type_compatible_pairs is not None:
        top_k_indices, top_k_scores = get_type_aware_top_k_candidates(similarity_matrix, type_compatible_pairs, args.num_candidates)
    else:
        top_k_indices, top_k_scores = get_top_k_candidates(similarity_matrix, args.num_candidates)

    # 9. Save candidates
    print(f"Saving candidates to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"# Format: kg1_entity_id\tkg2_entity_id\tsimilarity_score\trank\n")
        # Only generate candidates for KG1 entities (first n_kg1 rows of similarity matrix)
        total_candidates = 0
        for i in range(n_kg1):
            for rank, (j, score) in enumerate(zip(top_k_indices[i], top_k_scores[i])):
                # Skip invalid candidates (j == -1)
                if j == -1:
                    continue
                # Adjust KG2 entity index: add n_kg1 offset to get correct entity ID
                kg2_entity_id = j + n_kg1
                f.write(f"{i}\t{kg2_entity_id}\t{score:.6f}\t{rank+1}\n")
                total_candidates += 1

    # 10. Evaluate if requested
    if args.evaluate:
        ref_pairs_path = os.path.join(data_path, "ref_pairs")
        if os.path.exists(ref_pairs_path):
            print("Evaluating predictions...")
            ground_truth = load_alignments(ref_pairs_path)
            
            # Evaluate using the generated candidate file
            hit_k = [1, 5, 10]
            acc, mrr = evaluate_candidate_predictions(args.output, ground_truth, hit_k)
            
            print(f"Evaluation Results (based on generated candidates):")
            print(f"Hits@{hit_k}: {[round(a, 3) for a in acc]}")
            print(f"MRR: {mrr:.3f}")
        else:
            print("No ground truth found for evaluation")

    print(f"Total candidates generated: {total_candidates}")
    print(f"Done! Candidates saved to {args.output}")

if __name__ == "__main__":
    main()

"""
# Basic candidate generation
python predict_alignment.py --data airelle --model-path trained_models/airelle_model.pt --num-candidates 20

# With evaluation
python predict_alignment.py --data airelle --model-path trained_models/airelle_model.pt --num-candidates 10 --evaluate

# Different scoring method
python predict_alignment.py --data airelle --model-path trained_models/airelle_model.pt --scoring l2 --num-candidates 15

# With type-aware candidate blocking
python predict_alignment.py --data airelle --model-path trained_models/airelle_model.pt --type_threshold 0.7 --num-candidates 10

# Disable types completely
python predict_alignment.py --data airelle --model-path trained_models/airelle_model.pt --no_types --num-candidates 15

# With noise perturbation
python predict_alignment.py --data airelle --model-path trained_models/airelle_model.pt --add_noise --noise_ratio 0.1 --noise_type both --num-candidates 10

# Add noise only to name embeddings
python predict_alignment.py --data airelle --model-path trained_models/airelle_model.pt --add_noise --noise_ratio 0.2 --noise_type name --num-candidates 10

# Add noise only to structural embeddings
python predict_alignment.py --data airelle --model-path trained_models/airelle_model.pt --add_noise --noise_ratio 0.15 --noise_type structure --num-candidates 10

# Use model trained without name embeddings
python predict_alignment.py --data airelle --model-path trained_models/airelle_model_no_name.pt --no_name --num-candidates 10
"""