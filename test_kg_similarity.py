#!/usr/bin/env python3
"""
Test script for the compute_kg_similarity function.
This script demonstrates how to use the KG similarity computation method.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'simple_hhea'))

from simple_hhea.utils import compute_kg_similarity, load_alignments

def test_kg_similarity():
    """Test the KG similarity computation function."""
    
    # Example usage - you would replace this with your actual data path
    data_path = "data/airelle"  # Adjust this to your actual data path
    
    # Check if data path exists
    if not os.path.exists(data_path):
        print(f"Data path {data_path} does not exist.")
        print("Please update the data_path variable to point to your KG data directory.")
        return
    
    try:
        # Load alignment seed (reference pairs)
        alignment_file = os.path.join(data_path, "ref_ent_ids")
        if os.path.exists(alignment_file):
            alignment_seed = load_alignments(alignment_file)
            print(f"Loaded {len(alignment_seed)} alignment pairs from {alignment_file}")
        else:
            print(f"Alignment file {alignment_file} not found.")
            return
        
        # Test different gamma values
        gamma_values = [0.0, 0.3, 0.5, 0.7, 1.0]
        
        print("\n" + "="*60)
        print("KG SIMILARITY COMPUTATION RESULTS")
        print("="*60)
        
        for gamma in gamma_values:
            print(f"\nTesting with gamma = {gamma}")
            print("-" * 30)
            
            # Compute KG similarity
            results = compute_kg_similarity(
                data_path=data_path,
                alignment_seed=alignment_seed,
                gamma=gamma,
                use_structure=True,
                use_text=True
            )
            
            print(f"Results:")
            print(f"  Overall similarity: {results['overall_similarity']:.4f}")
            print(f"  Structure similarity: {results['structure_similarity']:.4f}")
            print(f"  Text similarity: {results['text_similarity']:.4f}")
            print(f"  Number of pairs: {results['num_pairs']}")
        
        print("\n" + "="*60)
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_kg_similarity()
