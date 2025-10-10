"""
Main method to produce embeddings for a given dataset.
Combines the following steps:
- get name embeddings
- get deepwalk embeddings
- get longterm embeddings
- get type embeddings (optional)
- save embeddings

Parameters: 
--dataset: name of the dataset to produce embeddings for
--no_types: disable type embedding generation

Usage:
python produce_embedding.py --dataset <dataset_name> [--no_types]
"""

from argparse import ArgumentParser, Namespace
from discriminatEA.process_name_embedding import main as process_name_embedding
from discriminatEA.feature_perprocessing.preproccess import main as process_deepwalk_embedding
from discriminatEA.feature_perprocessing.longterm.main import main as process_longterm_embedding
from discriminatEA.feature_perprocessing.get_deep_emb import main as process_deep_emb
from discriminatEA.process_type_embedding import main as process_type_embedding

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sample_dataset")
    parser.add_argument("--no_types", action="store_true", help="Disable type embedding generation")
    args = parser.parse_args()
    dataset = args.dataset

    # 1. Name Embeddings
    print(f"Producing name embeddings for dataset {dataset}")
    process_name_embedding(dataset)
    print("--------------------------------")  
    print("--------------------------------")  
    # 2. Deepwalk Preprocessing
    print(f"Producing deepwalk embeddings (preprocessing) for dataset {dataset}")
    process_deepwalk_embedding(dataset)
    print("--------------------------------")   
    print("--------------------------------")  
    # 3. Longterm Embeddings (node2vec)
    print(f"Producing longterm embeddings for dataset {dataset}")
    # Construct Namespace to mimic CLI args for longterm/main.py
    longterm_args = Namespace(
        input=f"data/{dataset}/deepwalk.data",
        output=f"data/{dataset}/longterm.vec",
        dimensions=64,
        walk_length=80,
        num_walks=10,
        window_size=10,
        iter=5,
        workers=12,
        p=1e-100,
        q=0.7,  # match shell script
        weighted=False,
        unweighted=False,
        directed=False,
        undirected=False,
        node2rel=f"data/{dataset}/node2rel"
    )
    process_longterm_embedding(longterm_args)
    print("--------------------------------")
    print("--------------------------------")
    # 4. Deep Embeddings
    print(f"Producing deep embeddings for dataset {dataset}")
    process_deep_emb(f"data/{dataset}/")
    print("--------------------------------")
    print("--------------------------------")
    
    # 5. Type Embeddings (optional) using new format: type_ids + ent_types_1/2
    if not args.no_types:
        print(f"Producing type embeddings for dataset {dataset}")
        process_type_embedding(dataset)
        print("--------------------------------")
        print("--------------------------------")
    else:
        print("Skipping type embedding generation (--no_types flag set)")
        print("--------------------------------")
        print("--------------------------------")

    
    print(f"Produced embeddings for dataset {dataset}")



