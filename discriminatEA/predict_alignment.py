import os
import argparse
import torch
import numpy as np
from discriminatEA.model import Simple_HHEA
from discriminatEA.utils import load_embeddings, load_alignments, get_train_set
from discriminatEA.CSLS_ import eval_alignment_by_sim_mat

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

def evaluate_predictions(model, ground_truth, hit_k=[1, 5, 10], csls_k=10):
    """Evaluate predictions using the same logic as training evaluation."""
    model.eval()
    with torch.no_grad():
        # Get embeddings for ground truth pairs (same as training)
        feat = model()[ground_truth]
        print(f"Evaluation feat shape: {feat.shape}")
        
        # Split into left and right entities (same as training)
        Lvec, Rvec = feat[:, 0, :].detach().cpu().numpy(), feat[:, 1, :].detach().cpu().numpy()
        Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
        Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
        
        # Use the same evaluation function as training
        t_prec_set, acc, t_mrr = eval_alignment_by_sim_mat(Lvec, Rvec, hit_k, 16, csls_k, accurate=True)
        acc = [round(n, 3) for n in acc]
        
    return t_prec_set, acc, t_mrr

def main():
    # 1. Parse arguments
    parser = argparse.ArgumentParser(description="Generate alignment candidates using a trained Simple-HHEA model.")
    parser.add_argument("--data", type=str, required=True, help="Name of the dataset (folder under data/)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model .pt file")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--output", type=str, default="alignment_candidates.txt", help="Output file for candidates")
    parser.add_argument("--num-candidates", type=int, default=10, help="Number of candidates per entity (default: 10)")
    parser.add_argument("--scoring", type=str, default="cosine", choices=["cosine", "l2", "csls"], 
                       help="Scoring method (default: cosine)")
    parser.add_argument("--no_time", action="store_true")
    parser.add_argument("--no_structure", action="store_true", default=False)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate predictions if ground truth available")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing (default: 1000)")
    args = parser.parse_args()

    data = args.data
    use_time = ("icews_wiki" in data or "icews_yago" in data) and not args.no_time
    use_structure = not args.no_structure

    # 2. Set device
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"

    # 3. Load dataset embeddings
    data_path = os.path.join("data", args.data)
    ent_name_emb, ent_dw_emb, ent_time_emb, ent_types, type_compatibility_matrix = load_embeddings(data_path, add_noise=False, noise_ratio=0.0, use_structure=True, use_time=use_time)

    # 4. Recreate model architecture
    print("debugging in predict_alignment.py")
    print(use_time)
    model = Simple_HHEA(
        time_span=1+27*13,
        ent_name_emb=ent_name_emb,
        ent_time_emb=ent_time_emb,
        ent_dw_emb=ent_dw_emb,
        ent_types=ent_types,
        use_structure=use_structure,
        use_time=use_time,
        emb_size=64,
        structure_size=8,
        time_size=8,
        device=device
    )
    model = model.to(device)

    # 5. Load trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

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

    # 8. Generate top-k candidates
    print(f"Generating top-{args.num_candidates} candidates...")
    top_k_indices, top_k_scores = get_top_k_candidates(similarity_matrix, args.num_candidates)

    # 9. Save candidates
    print(f"Saving candidates to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"# Format: kg1_entity_id\tkg2_entity_id\tsimilarity_score\trank\n")
        # Only generate candidates for KG1 entities (first n_kg1 rows of similarity matrix)
        for i in range(n_kg1):
            for rank, (j, score) in enumerate(zip(top_k_indices[i], top_k_scores[i])):
                # Adjust KG2 entity index: add n_kg1 offset to get correct entity ID
                kg2_entity_id = j + n_kg1
                f.write(f"{i}\t{kg2_entity_id}\t{score:.6f}\t{rank+1}\n")

    # 10. Evaluate if requested
    if args.evaluate:
        ref_pairs_path = os.path.join(data_path, "ref_pairs")
        if os.path.exists(ref_pairs_path):
            print("Evaluating predictions...")
            ground_truth = load_alignments(ref_pairs_path)
            
            # No need to convert predictions - we'll use the same evaluation as training
            
            # Evaluate using the same logic as training
            hit_k = [1, 5, 10]
            t_prec_set, acc, t_mrr = evaluate_predictions(model, ground_truth, hit_k)
            
            print(f"Evaluation Results:")
            print(f"Hits@{hit_k}: {acc}")
            print(f"MRR: {t_mrr:.3f}")
        else:
            print("No ground truth found for evaluation")

    print(f"Total candidates generated: {n_kg1 * args.num_candidates}")
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
"""