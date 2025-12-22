import numpy as np
import pandas as pd
from scipy import stats
import argparse

def read_csv_to_distribution(file_path):
    """Read a CSV file and return its probability distribution."""
    try:
        # Read CSV file, assuming no header
        data = pd.read_csv(file_path, header=None).values.flatten()
        
        # Calculate probability distribution
        values, counts = np.unique(data, return_counts=True)
        probabilities = counts / counts.sum()
        
        return dict(zip(values, probabilities))
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        raise

def calculate_kl_divergence(p, q):
    """Calculate KL divergence between two probability distributions."""
    # Ensure both distributions have the same keys
    all_keys = set(p.keys()).union(set(q.keys()))
    
    # Align the distributions and add small epsilon to avoid log(0)
    eps = 1e-10
    p_array = np.array([p.get(k, eps) for k in all_keys])
    q_array = np.array([q.get(k, eps) for k in all_keys])
    
    # Normalize to ensure they are proper probability distributions
    p_array = p_array / p_array.sum()
    q_array = q_array / q_array.sum()
    
    # Calculate KL divergence
    return stats.entropy(p_array, q_array)

def main():
    originalPath = "dataset/clf_num/bank-marketing.csv"
    generatePath = "dataset/generatedData/CLLM_results/bank-marketing_synthetic.csv"
    # parser = argparse.ArgumentParser(description='Calculate KL divergence between two CSV files')
    # parser.add_argument('file1', type=str, help=originalPath)
    # parser.add_argument('file2', type=str, help=generatePath)
    
    try:
        # Read and process both files
        dist1 = read_csv_to_distribution(originalPath)
        dist2 = read_csv_to_distribution(generatePath)
        
        # Calculate KL divergence (note: KL is not symmetric)
        kl_1_2 = calculate_kl_divergence(dist1, dist2)
        kl_2_1 = calculate_kl_divergence(dist2, dist1)
        
        # Jensen-Shannon divergence (symmetric version of KL)
        js_divergence = 0.5 * (kl_1_2 + kl_2_1)
        
        print(f"KL Divergence ({originalPath} -> {generatePath}): {kl_1_2:.6f}")
        print(f"KL Divergence ({generatePath} -> {originalPath}): {kl_2_1:.6f}")
        print(f"Jensen-Shannon Divergence: {js_divergence:.6f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
