import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, fisher_exact
from statsmodels.stats.multitest import multipletests
import gseapy as gp
from gseapy import dotplot
import argparse
import matplotlib.pyplot as plt
import os
import sys
import re
import warnings
import random
from collections import Counter

warnings.filterwarnings('ignore')


#### parse_phenotypes这个函数的主要作用是，读取mash群文件，菌株与prophage/ds关系文件，判断出哪些mash群属于正相关群/负相关群
def parse_phenotypes(file_mash, file_prophage, file_ds, min_cluster_size, ds_target):
    print(f">>> 阶段 1/4: 提取群体并进行表型划分  目标ds为{ds_target} <<<")
    cluster_to_strains = {}
    strain_universe = set()
    df_mash = pd.read_csv(file_mash, sep=',|\t', engine='python', header=0)
    df_mash.columns = ['Sample', 'Cluster']
    for _, row in df_mash.iterrows():
        strain = str(row['Sample']).strip()
        cluster = str(row['Cluster']).strip()
        cluster_to_strains.setdefault(cluster, []).append(strain)
        strain_universe.add(strain)

    strain_to_prophages = {s: [] for s in strain_universe}
    with open(file_prophage, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1 and parts[0] in strain_to_prophages:
                strain_to_prophages[parts[0]].extend(parts[1:])

    strain_to_ds = {s: set() for s in strain_universe}
    with open(file_ds, 'r') as f:
        header_skipped = False
        for line in f:
            if not header_skipped:
                header_skipped = True
                continue
            parts = line.strip().split()
            if len(parts) > 1:
                ds_name = parts[0]
                for strain in parts[1:]:
                    if strain in strain_to_ds:
                        strain_to_ds[strain].add(ds_name)

    clusters_pos_dict = {}
    clusters_neg_dict = {}

    for cluster, strains in cluster_to_strains.items():
        if len(strains) < min_cluster_size:
            continue

        p_counts_with = [len(strain_to_prophages[s]) for s in strains if ds_target in strain_to_ds[s]]
        p_counts_without = [len(strain_to_prophages[s]) for s in strains if ds_target not in strain_to_ds[s]]

        if len(p_counts_with) == 0 or len(p_counts_without) == 0:
            continue

        _, p_val = mannwhitneyu(p_counts_with, p_counts_without, alternative='two-sided')
        if p_val < 0.05:
            if np.mean(p_counts_with) > np.mean(p_counts_without):
                clusters_pos_dict[cluster] = strains
            else:
                clusters_neg_dict[cluster] = strains

    total_pos_strains = sum(len(s) for s in clusters_pos_dict.values())
    total_neg_strains = sum(len(s) for s in clusters_neg_dict.values())

    print(f"保留的正相关群: {len(clusters_pos_dict)} 个 (涉及 {total_pos_strains} 株菌), 分别是{clusters_pos_dict.keys()}")
    print(f"保留的负相关群: {len(clusters_neg_dict)} 个 (涉及 {total_neg_strains} 株菌), 分别是{clusters_neg_dict.keys()}\n")
    return clusters_pos_dict, clusters_neg_dict


#### 该函数的作用是，读取上一个函数中生成的正/负相关群，执行bootstrap抽样，以及进行fisher精确检验
def run_bootstrap_gwas(file_cdhit, pos_dict, neg_dict, sample_size, bootstrap_n, upper_threshold, outdir):
    print(f">>> 阶段 2/4: Bootstrap 泛基因组关联分析 (B={bootstrap_n}, 阈值={upper_threshold}) <<<")

    all_pool_strains = set()
    for strains in pos_dict.values(): all_pool_strains.update(strains)
    for strains in neg_dict.values(): all_pool_strains.update(strains)

    print("  -> 正在预加载候选菌株的分布矩阵 (优化版)...")
    cluster_presence = {}
    with open(file_cdhit, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            cid = parts[0]
            strains_here = set()
            for p in parts[1:]:
                match = re.search(r'(GCA_\d+\.\d+)', p)
                if match and match.group(1) in all_pool_strains:
                    strains_here.add(match.group(1))
            if strains_here:
                cluster_presence[cid] = strains_here

    print("  -> 正在执行 Bootstrap 抽样检验 (这可能需要一小会儿)...")
    pos_sig_counts = Counter()
    neg_sig_counts = Counter()

    # 【新增】：用于记录每个 cluster 在多少次抽样中真正参与了检验（非全无/全有）
    tested_counts = Counter()

    for i in range(bootstrap_n):
        samp_pos = set()
        for c, strains in pos_dict.items():
            samp_pos.update(random.sample(strains, min(len(strains), sample_size)))

        samp_neg = set()
        for c, strains in neg_dict.items():
            samp_neg.update(random.sample(strains, min(len(strains), sample_size)))

        tot_p = len(samp_pos)
        tot_n = len(samp_neg)

        cids, pvals, or_vals = [], [], []

        for cid, strains_here in cluster_presence.items():
            pos_w = len(strains_here & samp_pos)
            neg_w = len(strains_here & samp_neg)

            # 如果在本次抽出的菌株中全有或全无，则跳过
            if (pos_w == 0 and neg_w == 0) or (pos_w == tot_p and neg_w == tot_n):
                continue

            tested_counts[cid] += 1

            table = [[pos_w, tot_p - pos_w], [neg_w, tot_n - neg_w]]
            or_val, p_val = fisher_exact(table, alternative='two-sided')

            # ==========================================================
            # 【新增探头】：专门监控 Cluster_38684 的每一次抽样列联表
            if cid == "Cluster_38684":
                # 为了防止控制台被1000行日志淹没，也可以加个限制，比如只看前20次
                # if i < 20:
                print(f"[Bootstrap {i + 1:03d}] {cid}:")
                print(f"  正相关群 (携带 vs 缺失): [{pos_w:2d}, {tot_p - pos_w:2d}]")
                print(f"  负相关群 (携带 vs 缺失): [{neg_w:2d}, {tot_n - neg_w:2d}]")
                print(f"  -> OR: {or_val:.2f} | P-value: {p_val:.2e}\n")
            # ==========================================================

            cids.append(cid)
            pvals.append(p_val)
            or_vals.append(or_val)

        if pvals:
            _, fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
            for j in range(len(cids)):
                if fdr[j] < 0.05:
                    if or_vals[j] > 1:
                        pos_sig_counts[cids[j]] += 1
                    elif or_vals[j] < 1:
                        neg_sig_counts[cids[j]] += 1

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    res_list = []
    for cid, count in pos_sig_counts.items():
        res_list.append({'Cluster': cid, 'Direction': 'Positive', 'Bootstrap_Score': count})
    for cid, count in neg_sig_counts.items():
        res_list.append({'Cluster': cid, 'Direction': 'Negative', 'Bootstrap_Score': count})

    df_boot = pd.DataFrame(res_list)
    if not df_boot.empty:
        df_boot.sort_values(by=['Direction', 'Bootstrap_Score'], ascending=[True, False], inplace=True)
        boot_out_path = os.path.join(outdir, "bootstrap_fisher_results.tsv")
        df_boot.to_csv(boot_out_path, sep='\t', index=False)
        print(f"  -> Bootstrap 报告已保存至: {boot_out_path}")

    # 提取最终稳健的群
    robust_pos = {cid for cid, count in pos_sig_counts.items() if count >= upper_threshold}
    robust_neg = {cid for cid, count in neg_sig_counts.items() if count >= upper_threshold}

    # 【新增】：提取有效的背景群（只有被成功检验过 >= upper 阈值次的，才有资格作为背景）
    effective_bg_clusters = {cid for cid, count in tested_counts.items() if count >= upper_threshold}

    print(
        f"  -> 检验完成。支持率 >= {upper_threshold} 的稳健簇: 正相关 {len(robust_pos)} 个 | 负相关 {len(robust_neg)} 个")
    print(f"  -> 有效背景簇数量: {len(effective_bg_clusters)} 个 (已剔除未充分暴露的罕见簇)\n")

    return robust_pos, robust_neg, effective_bg_clusters, all_pool_strains


#### 这段函数的目的是，将上一步检验出来的cluster，对回到基因上
def extract_genes_and_background(file_cdhit, robust_pos, robust_neg, effective_bg_clusters, all_pool_strains):
    print(">>> 阶段 3/4: 提取目标基因及重构有效背景集 <<<")
    genes_pos = []
    genes_neg = []
    bg_genes = set()

    with open(file_cdhit, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            cid = parts[0]

            # 【核心逻辑】：只有属于有效背景宇宙的 Cluster，我们才提取其中的基因
            if cid in effective_bg_clusters:
                valid_proteins = []
                for p in parts[1:]:
                    match = re.search(r'(GCA_\d+\.\d+)', p)
                    if match and match.group(1) in all_pool_strains:
                        valid_proteins.append(p)

                if valid_proteins:
                    bg_genes.update(valid_proteins)  # 纳入正式的富集分析背景集

                    if cid in robust_pos:
                        genes_pos.extend(valid_proteins)
                    elif cid in robust_neg:
                        genes_neg.extend(valid_proteins)

    print(f"  -> 提取完毕: 正相关基因 {len(genes_pos)} 个, 负相关基因 {len(genes_neg)} 个")
    print(f"  -> 最终有效背景宇宙大小: {len(bg_genes)} 个非冗余基因\n")
    return genes_pos, genes_neg, bg_genes


def run_go_enrichment_memory_optimized(emapper_file, target_genes, background_genes, output_dir, prefix, cutoff=0.05):
    print(f">>> 阶段 4/4: 运行 GO 富集分析 ({prefix} 组) <<<")

    if not target_genes:
        print(f"[{prefix}] 无达标的差异基因，跳过富集。\n")
        return

    go_gene_sets = {}

    # 解析 EggNOG 注释，只保留在背景集(background_genes)中的基因
    print("  -> 正在映射 GO 词条...")
    with open(emapper_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip('\n').split('\t')

            if len(parts) > 9:
                gene_id = parts[0].strip()

                # 【核心拦截】只允许存在于抽样池宇宙中的基因参与计算
                if gene_id not in background_genes:
                    continue

                go_terms = parts[9].strip()
                if go_terms and go_terms != '-' and go_terms != 'nan':
                    for go in go_terms.split(','):
                        go = go.strip()
                        if go not in go_gene_sets:
                            go_gene_sets[go] = []
                        go_gene_sets[go].append(gene_id)

    # 取交集（为了防止名称微小差异，规范化匹配）
    overlap = set(target_genes) & background_genes
    print(f"[{prefix}] 目标基因: {len(target_genes)} | 背景基因: {len(background_genes)} | 在背景库中匹配成功: {len(overlap)}")

    if len(overlap) < 5:
        print(f"[警告] {prefix} 匹配的基因过少，跳过富集。\n")
        return

    try:
        enr_res = gp.enrich(
            gene_list=list(overlap),
            gene_sets=go_gene_sets,
            background=list(background_genes),  # 传入我们严格定义的背景
            outdir=None,
            cutoff=cutoff,
            verbose=False
        )
    except Exception as e:
        print(f"运行富集出错: {e}\n")
        return

    if enr_res is None or enr_res.results.empty:
        print(f"[{prefix}] 未筛选到 FDR < {cutoff} 的显著 GO Terms。\n")
        return

    results = enr_res.results
    output_table = os.path.join(output_dir, f"{prefix}_GO_enrichment.tsv")
    results.to_csv(output_table, sep="\t", index=False)

    try:
        sig_results = results[results['Adjusted P-value'] < cutoff]
        plot_data = sig_results if len(sig_results) > 0 else results.head(20)

        if len(plot_data) > 0:
            ax = dotplot(plot_data, title=f'GO Enrichment ({prefix})', cmap='viridis', top_term=20)
            plt.savefig(os.path.join(output_dir, f"{prefix}_GO_dotplot.pdf"), bbox_inches='tight')
            print(f"[{prefix}] 结果和气泡图已保存。\n")
    except Exception as e:
        print(f"[{prefix}] 绘图失败: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Pangenome GWAS with Bootstrapping Rarefaction")
    parser.add_argument('--mash', required=True)
    parser.add_argument('--prophage', required=True)
    parser.add_argument('--ds', required=True)
    parser.add_argument('--cdhit', required=True)
    parser.add_argument('--eggnog', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--ds_target', type=str, default='CRISPR-Cas')
    parser.add_argument('--min_cluster_size', type=int, default=50)
    parser.add_argument('--cutoff', type=float, default=0.05)

    # 新增的参数
    parser.add_argument('--sample_size', type=int, default=50, help='每个Mash群的抽样菌株数')
    parser.add_argument('--bootstrap', type=int, default=100, help='Bootstrap重抽样次数')
    parser.add_argument('--upper', type=int, default=80, help='进入富集分析的Bootstrap最低支持阈值')

    args = parser.parse_args()

    # 1. 解析表型并获取 Mash 分组
    pos_dict, neg_dict = parse_phenotypes(
        args.mash, args.prophage, args.ds, args.min_cluster_size, args.ds_target
    )

    # 2. 执行 Bootstrap 检验并获取稳健的 Cluster
    robust_pos, robust_neg, effective_bg_clusters, all_pool_strains = run_bootstrap_gwas(
        args.cdhit, pos_dict, neg_dict, args.sample_size, args.bootstrap, args.upper, args.outdir
    )

    # 3. 提取基因和有效背景集 (Universe)
    genes_pos, genes_neg, bg_genes = extract_genes_and_background(
        args.cdhit, robust_pos, robust_neg, effective_bg_clusters, all_pool_strains
    )

    # 4. 富集分析
    run_go_enrichment_memory_optimized(
        args.eggnog, genes_pos, bg_genes, args.outdir, "Enriched_in_Positive", args.cutoff
    )
    run_go_enrichment_memory_optimized(
        args.eggnog, genes_neg, bg_genes, args.outdir, "Enriched_in_Negative", args.cutoff
    )

    print(">>> 任务顺利结束 <<<")


if __name__ == "__main__":
    main()