import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch

from simple_hhea.CSLS_ import eval_alignment_by_sim_mat
from simple_hhea.utils import *
from simple_hhea.model import Simple_HHEA


### load embeddings


### training
def l1(ll, rr):
    return torch.sum(torch.abs(ll - rr), axis=-1)

def evaluate(model, dev_alignments, hit_k=[1, 5, 10], num_threads=16, csls_k=10):
    model.eval()
    with torch.no_grad():
        feat = model()[dev_alignments]
        Lvec, Rvec = feat[:, 0, :].detach().cpu().numpy(), feat[:, 1, :].detach().cpu().numpy()
        Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
        Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
        t_prec_set, acc, t_mrr = eval_alignment_by_sim_mat(Lvec, Rvec, hit_k, num_threads, csls_k, accurate=True)
        acc = [round(n, 3) for n in acc]
    return t_prec_set, acc, t_mrr

def train(model:nn.Module, alignment_pairs, dev_alignments, epochs=1500, learning_rate=0.01, weight_decay=0.001, gamma=1.0, hit_k=[1, 5, 10], csls_k=10, save_best_model_path=None, type_aware_training=False, loss_weights=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print(f"parameters: {get_n_params(model)}")

    losses = []
    t_prec = []
    accs = []
    t_mrrs = []
    best_acc = [0] * len(hit_k)
    best_mrr = 0
    best_hit5 = 0
    batch_size = len(alignment_pairs)
    for i in tqdm(range(epochs)):
        ### forwad
        model.train()
        optimizer.zero_grad()
        feat = model()[alignment_pairs]
        ### loss
        l, r, fl, fr = feat[:, 0, :], feat[:, 1, :], feat[:, 2, :], feat[:, 3, :]
        
        if type_aware_training and loss_weights is not None:
            # Apply type-aware loss weighting
            weights_tensor = torch.tensor(loss_weights[:batch_size], dtype=torch.float32, device=model.device)
            # Weighted triplet loss
            # l is the anchor, r is the positive, fl is the negative, fr is another negative, gamma the margin
            # Idea : separate both the anchor and the positiv from the negative samples

            loss = torch.sum(weights_tensor * (
                nn.ReLU()(gamma + l1(l, r) - l1(l, fr)) + 
                nn.ReLU()(gamma / 2 + l1(l, r) - l1(fl, r))
            )) / batch_size
        else:
            # Standard triplet loss
            loss = torch.sum(nn.ReLU()(gamma + l1(l, r) - l1(l, fr)) + nn.ReLU()(gamma + l1(l, r) - l1(fl, r))) / batch_size
        ### backward
        losses.append(loss.item())
        loss.backward(retain_graph=True)
        optimizer.step()
        ### evaluate
        if (i + 1) % 10 == 0:
            t_prec_set, acc, t_mrr = evaluate(model, dev_alignments, hit_k, csls_k)

            for j in range(len(hit_k)):
                if best_acc[j] < acc[j]:
                    best_acc[j] = acc[j]
            # Save if new best mrr, or if mrr is equal but hit@5 is better
            save_model = False
            if best_mrr < t_mrr:
                best_mrr = t_mrr
                best_hit5 = acc[1] if len(acc) > 1 else 0
                save_model = True
            elif best_mrr == t_mrr:
                if len(acc) > 1 and acc[1] > best_hit5:
                    best_hit5 = acc[1]
                    save_model = True
            if save_model and save_best_model_path is not None:
                torch.save(model.state_dict(), save_best_model_path)
            print(f"//best results: hits@{hit_k} = {best_acc}, mrr = {best_mrr:.3f}//")
            accs.append(acc)
            t_mrrs.append(t_mrr)
            t_prec.append(t_prec_set)
    return losses, t_prec, accs, t_mrrs, best_acc, best_mrr




if __name__ == "__main__":
    ### hyper parmeters
    parser = argparse.ArgumentParser(description="Simple-HHEA Experiment")
    parser.add_argument("--data", type=str, default="icews_wiki")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=12306)
    ###### ablation settings
    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument("--noise_ratio", type=float, default=0.3)
    parser.add_argument("--no_name", action="store_true", help="Disable name embeddings")
    parser.add_argument("--no_structure", action="store_true")
    parser.add_argument("--no_time", action="store_true")
    parser.add_argument("--no_types", action="store_true", help="Disable type-aware training")
    parser.add_argument("--type_threshold", type=float, default=0.7, help="Type similarity threshold for candidate blocking")
    parser.add_argument("--type_weight", type=float, default=0.5, help="Weight for type compatibility in loss")
    ###### training settings
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--hit_k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--csls_k", type=int, default=10) #number of nearest neighbors to use for CSLS
    parser.add_argument("--save-model", action="store_true", help="Save the trained model after training.")
    args = parser.parse_args()
    
    ### basic settings
    data = args.data
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    use_time = ("icews_wiki" in data or "icews_yago" in data) and not args.no_time
    use_structure = not args.no_structure
    use_name = not args.no_name
    use_types = not args.no_types
    print(f"start exp: noise_ratio={args.noise_ratio}, data=\"{args.data}\", use_name={use_name}, use_structure={use_structure}, use_time={use_time}, use_types={use_types} with type_weight={args.type_weight}")
    ### random settings
    fixed(args.random_seed)

    ### load datas
    data_path = os.path.join("data", data)
    all_triples, node_size, rel_size = load_triples(data_path, True)
    print(f"node_size={node_size} , rel_size={rel_size}")

    train_alignments = load_alignments(os.path.join(data_path, "sup_pairs"))
    dev_alignments = load_alignments(os.path.join(data_path, "ref_pairs"))
    print(f"Train/Val: {len(train_alignments)}/{len(dev_alignments)}")

    ### load embeddings (including types)
    ent_name_emb, ent_dw_emb, ent_time_emb, ent_types_tensor, type_compatibility_matrix = load_embeddings(
        data_path, args.add_noise, args.noise_ratio, use_structure, use_time, use_types
    )

    # ent_types_tensor is already a torch.LongTensor from utils.load_entity_types

    ### model
    # Conditionally use name embeddings based on use_name flag
    name_emb_for_model = ent_name_emb if use_name else None
    
    model = Simple_HHEA(
        time_span=1+27*13,
        ent_name_emb=name_emb_for_model,
        ent_time_emb=ent_time_emb,
        ent_dw_emb=ent_dw_emb,
        ent_types=ent_types_tensor,
        use_name=use_name,
        use_structure=use_structure,
        use_time=use_time,
        emb_size=64,
        structure_size=8,
        time_size=8,
        device=device
    )
    model = model.to(device)

    alignment_pairs = get_train_set(train_alignments, node_size, node_size)

    # Prepare model save path if needed
    trained_model_dir = "trained_models"
    if args.save_model:
        if not os.path.exists(trained_model_dir):
            os.makedirs(trained_model_dir)
        model_save_path = os.path.join(trained_model_dir, f"{data}_simple_hhea_model_name{use_name}_structure{use_structure}_time{use_time}_types{use_types}weight{args.type_weight}.pt")
    else:
        model_save_path = None

    # Type-aware training setup
    type_aware_training = use_types and (ent_types_tensor is not None) and (type_compatibility_matrix is not None)
    
    if type_aware_training:
        print("Using type-aware training with loss weighting and candidate blocking")
        # Get actual KG sizes from entity ID files
        kg1_size = 0
        kg2_size = 0
        if os.path.exists(os.path.join(data_path, "ent_ids_1")):
            with open(os.path.join(data_path, "ent_ids_1"), 'r') as f:
                kg1_size = len(f.readlines())
        if os.path.exists(os.path.join(data_path, "ent_ids_2")):
            with open(os.path.join(data_path, "ent_ids_2"), 'r') as f:
                kg2_size = len(f.readlines())
        
        # Compute type-aware loss weights for the actual training batch (positive pairs only)
        loss_weights = compute_type_aware_loss_weights(
            alignment_pairs[:, :2], type_compatibility_matrix, kg1_size, kg2_size,
            base_weight=1.0, type_weight=args.type_weight, ent_types=ent_types_tensor
        )
        
        # Optionally perform candidate blocking
        if args.type_threshold > 0:
            candidate_pairs = type_aware_candidate_blocking_by_types(
                type_compatibility_matrix, ent_types_tensor, kg1_size, kg2_size, threshold=args.type_threshold, top_k=10
            )
            print(f"Type-aware candidate blocking found {len(candidate_pairs)} pairs")
    else:
        loss_weights = None
        if not use_types:
            print("Type-aware training disabled via command line argument")
        else:
            print("Type information not available, using standard training")

    losses, t_prec, accs, t_mrrs, best_acc, best_mrr = train(
        model, alignment_pairs, dev_alignments, args.epochs, args.lr, args.wd, args.gamma, 
        hit_k=args.hit_k, csls_k=args.csls_k, save_best_model_path=model_save_path,
        type_aware_training=type_aware_training, loss_weights=loss_weights
    )

    ### save result
    result_dir = "results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(os.path.join(result_dir, f"{data}_result_file_mlp.txt"), "a+", encoding="utf-8") as fw:
        fw.write(f"settings: noise_ratio: {args.noise_ratio}, use_name: {use_name}, use_time: {use_time}, use_structure: {use_structure}, use_types: {use_types}, type_weight: {args.type_weight}, type_threshold: {args.type_threshold}\n\tbest results: hits@[1, 5, 10] = {best_acc}, mrr = {best_mrr:.3f}\n")

    if args.save_model:
        print(f"Best model saved to {model_save_path}")
    
    print("Model training completed")