import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import sys
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap Spearman Correlation for GLOBAL Prophage & Defense Burden.")
    parser.add_argument('-p', '--prophage', required=True, help="Input: Strain and ALL prophage list (no header).")
    parser.add_argument('-d', '--defense', required=True, help="Input: ALL Defense systems and strains (header).")
    parser.add_argument('-m', '--mash', required=True, help="Input: Mash clusters (CSV).")
    parser.add_argument('--min_strains', type=int, default=3, help="Threshold: Min strains in a cluster (default: 3).")
    parser.add_argument('--iterations', type=int, default=1000, help="Number of bootstrap iterations (default: 1000).")
    parser.add_argument('--out_plot', default='bootstrap_global_distribution.pdf',
                        help="Output: PDF file for distribution plot.")
    return parser.parse_args()


def main():
    args = parse_args()

    strain_prophage_counts = {}
    with open(args.prophage, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if not parts: continue
            strain = parts[0]
            # 减去第一列的菌株名，剩下的长度就是该菌株携带的所有原噬菌体总数
            strain_prophage_counts[strain] = len(parts) - 1

    strain_defense_counts = {}
    with open(args.defense, 'r') as f:
        header = next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            # 无论什么防御系统，只要出现，对应的菌株计数就加 1
            strains = parts[1:]
            for strain in strains:
                strain_defense_counts[strain] = strain_defense_counts.get(strain, 0) + 1

    print("[Info] Parsed global prophage and defense system counts.")

    df_mash = pd.read_csv(args.mash)
    df_mash.rename(columns={'Sample': 'Strain'}, inplace=True)

    df_mash['Prophage_Count'] = df_mash['Strain'].map(strain_prophage_counts).fillna(0)
    df_mash['Defense_Count'] = df_mash['Strain'].map(strain_defense_counts).fillna(0)

    cluster_counts = df_mash['Cluster'].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= args.min_strains].index
    df_filtered = df_mash[df_mash['Cluster'].isin(valid_clusters)]

    num_valid_clusters = len(valid_clusters)
    print(f"[Info] Valid Mash clusters (N>={args.min_strains}): {num_valid_clusters} out of {len(cluster_counts)}")

    if num_valid_clusters < 3:
        sys.exit("Error: Not enough clusters to calculate correlation.")

    print(f"\n[Running] Performing {args.iterations} bootstrap iterations...")
    rho_values = []
    sig_count = 0

    grouped = df_filtered.groupby('Cluster')

    for _ in tqdm(range(args.iterations), desc="Bootstrapping"):
        sampled_df = grouped.sample(n=1)
        rho, p_val = spearmanr(sampled_df['Prophage_Count'], sampled_df['Defense_Count'])

        if not np.isnan(rho):
            rho_values.append(rho)
            if p_val < 0.05:
                sig_count += 1


    if len(rho_values) == 0:
        sys.exit(
            "Error: All bootstrap iterations resulted in NaN. Check your input data (e.g., all counts might be zero).")

    rho_array = np.array(rho_values)
    mean_rho = np.mean(rho_array)
    ci_lower = np.percentile(rho_array, 2.5)
    ci_upper = np.percentile(rho_array, 97.5)
    sig_ratio = (sig_count / len(rho_values)) * 100

    print("\n" + "=" * 45)
    print(f"Global Burden Bootstrap Results ({len(rho_values)} valid iterations):")
    print(f"Mean Spearman rho: {mean_rho:.4f}")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Proportion of iterations with P<0.05: {sig_ratio:.1f}%")
    print("=" * 45 + "\n")

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    ax = sns.histplot(rho_array, bins=30, kde=True, color='teal', edgecolor='black', alpha=0.6)

    plt.axvline(mean_rho, color='red', linestyle='-', linewidth=2, label=f'Mean rho ({mean_rho:.3f})')
    plt.axvline(ci_lower, color='darkorange', linestyle='--', linewidth=2, label=f'95% CI Lower ({ci_lower:.3f})')
    plt.axvline(ci_upper, color='darkorange', linestyle='--', linewidth=2, label=f'95% CI Upper ({ci_upper:.3f})')

    # plt.title(
    #     f'Global Burden: Prophage vs Defense System\nBootstrap $\\rho$ Distribution (1 Genome/Cluster, {args.iterations} Iterations)',
    #     pad=15)
    plt.xlabel('Spearman Correlation Coefficient ($\\rho$)', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.legend()
    plt.tight_layout()

    plt.savefig(args.out_plot, dpi=300, bbox_inches='tight')
    print(f"[Success] Distribution plot saved to: {args.out_plot}\n")


if __name__ == '__main__':
    main()