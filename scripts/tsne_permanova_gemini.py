import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from skbio.diversity import beta_diversity
from skbio.stats.distance import permanova


def run_permanova(df_matrix, metadata_path, factor_name):
    df_meta = pd.read_csv(metadata_path, sep='\t', header=None, names=['strain', 'val'])
    df_meta['val'] = df_meta['val'].fillna('Unknown').replace('', 'Unknown')

    common = np.intersect1d(df_matrix.index, df_meta['strain'])
    df_m = df_matrix.loc[common]
    df_t = df_meta.set_index('strain').loc[common]

    mask = (df_t['val'] != "Unknown") & (df_t.groupby('val')['val'].transform('count') >= 3)
    df_m = df_m[mask.values]
    df_t = df_t[mask.values]

    if len(df_t['val'].unique()) < 2:
        return None

    dm = beta_diversity('jaccard', df_m.values, ids=df_m.index)
    res = permanova(dm, df_t['val'].values, permutations=999)

    f_stat = res['test statistic']
    n = res['sample size']
    k = res['number of groups']

    # R2 = (F * (k-1)) / (F * (k-1) + (n-k))
    r2 = (f_stat * (k - 1)) / (f_stat * (k - 1) + (n - k))

    return {
        'Factor': factor_name,
        'R2': r2,
        'p-value': res['p-value'],
        'F-stat': f_stat
    }


def main():
    parser = argparse.ArgumentParser(description="Compare effect sizes (R2) of different factors.")
    parser.add_argument("--matrix", required=True, help="Presence/Absence CSV")
    parser.add_argument("--meta_list", nargs='+', required=True,
                        help="Format: FactorName:Path (e.g., ST:st.tsv Country:loc.tsv)")
    
    # 修改1: 默认输出文件名后缀改为 .pdf
    parser.add_argument("--output", default="permanova_comparison.pdf", help="Output plot path (PDF format)")
    args = parser.parse_args()

    df_matrix = pd.read_csv(args.matrix, index_col=0)
    results = []

    for item in args.meta_list:
        try:
            name, path = item.split(':')
            print(f"Processing {name}...")
            res = run_permanova(df_matrix, path, name)
            if res:
                results.append(res)
        except Exception as e:
            print(f"❌ Error processing {item}: {e}")

    if not results:
        print("No valid results to plot.")
        return

    df_res = pd.DataFrame(results).sort_values('R2', ascending=True)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_res['Factor'], df_res['R2'], color='#5DADE2', edgecolor='#2E86C1', height=0.6)

    for i, bar in enumerate(bars):
        row = df_res.iloc[i]
        p_val = row['p-value']

        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "n.s."

        plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f" R²={row['R2']:.3f}, F={row['F-stat']:.1f} ({sig})",
                 va='center', fontsize=10, fontweight='bold')

    plt.xlabel('Effect Size (R² / Explained Variance)', fontsize=12, labelpad=10)
    plt.title('Drivers of Prophage Community Composition (PERMANOVA)', fontsize=14, pad=20)
    plt.xlim(0, df_res['R2'].max() * 1.4)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()

    plt.savefig(args.output, format='pdf')
    plt.close()
    print(f"\n✅ Comparison plot saved to {args.output}")


if __name__ == "__main__":
    main()