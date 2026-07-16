import pandas as pd
import argparse
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from collections import Counter
import os


def run_analysis_pipeline(args):
    print("Loading matrix and calculating t-SNE coordinates...")
    df_matrix = pd.read_csv(args.output_csv, index_col=0)

    jaccard_dist = pairwise_distances(df_matrix.values, metric='jaccard')

    tsne = TSNE(n_components=2, perplexity=30, metric='precomputed',
                init='random', random_state=42, n_iter=1000)
    embeddings = tsne.fit_transform(jaccard_dist)

    df_base = pd.DataFrame(embeddings, index=df_matrix.index, columns=['TSNE1', 'TSNE2'])

    tasks = [
        ("ST Type", args.st_mapping, args.tsne_out_prefix + "_ST.pdf"),
        ("Country", args.country_mapping, args.tsne_out_prefix + "_Country.pdf"),
        ("Date", args.date_mapping, args.tsne_out_prefix + "_Date.pdf")
    ]

    for feat_name, map_file, output_path in tasks:
        if not map_file or not os.path.exists(map_file):
            print(f"Skipping {feat_name} because mapping file is missing.")
            continue

        print(f"Generating plot for: {feat_name}...")
        plot_tsne(df_base.copy(), map_file, feat_name, output_path)


def plot_tsne(df_tsne, map_file, label_name, output_path):
    mapping_dict = {}
    counter = Counter()
    with open(map_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                strain, val = parts
                mapping_dict[strain] = val
                counter[val] += 1

    top_items = [item for item, count in counter.most_common(20) if item != "Unknown"]
    df_tsne['Category'] = df_tsne.index.map(lambda x: mapping_dict.get(x, "others"))
    df_tsne.loc[~df_tsne['Category'].isin(top_items), 'Category'] = 'others'

    distinct_colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#42d4f4'
    ]

    fig, ax = plt.subplots(figsize=(12, 9))  # 固定画布比例
    grouped = df_tsne.groupby('Category')

    if 'others' in grouped.groups:
        others = grouped.get_group('others')
        ax.scatter(others['TSNE1'], others['TSNE2'], c='#d3d3d3', alpha=0.2, s=6, label='others', zorder=1)

    color_idx = 0
    for name, group in grouped:
        if name == 'others': continue
        ax.scatter(group['TSNE1'], group['TSNE2'],
                   c=distinct_colors[color_idx % len(distinct_colors)],
                   alpha=0.9, s=20, label=name, zorder=2, edgecolors='none')
        color_idx += 1

    ax.set_title(f't-SNE of Prophage Profiles\n(Colored by {label_name})', fontsize=14, pad=15)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')

    plt.subplots_adjust(right=0.75, left=0.1, top=0.9, bottom=0.1)

    ax.legend(title=label_name, bbox_to_anchor=(1.02, 0.5), loc='center left',
              fontsize=8, title_fontsize=10, frameon=False, markerscale=1.2)

    plt.savefig(output_path, format='pdf')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate unified t-SNE plots for ST, Country, and Date.")

    # 输入矩阵
    parser.add_argument('--output_csv', required=True, help='Path to presence/absence matrix CSV')

    # 三个不同的映射文件 (修改了 source_mapping 为 country_mapping)
    parser.add_argument('--st_mapping', help='Strain to ST mapping (tab-separated)')
    parser.add_argument('--country_mapping', help='Strain to Country mapping (tab-separated)')
    parser.add_argument('--date_mapping', help='Strain to Date mapping (tab-separated)')

    # 输出前缀 (修改了 tsne_png_prefix 为 tsne_out_prefix)
    parser.add_argument('--tsne_out_prefix', required=True, help='Prefix for output images (e.g., results/my_plot)')

    args = parser.parse_args()
    run_analysis_pipeline(args)


if __name__ == "__main__":
    main()