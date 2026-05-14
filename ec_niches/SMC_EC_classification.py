#!/usr/bin/env python3

"""
script to annotate EC and SMC subtypes
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import io
import math
import os
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

SMCs = {
    "arterial_arteriolar_SMCs": [
        "Acta2", "Tagln", "Myh11", "Pln", "Cnn1", "Adamts1",
        "Arl4d", "Atf3", "Cd93", "H2afj", "Nanos1"
    ],
    "venous_SMCs": [
        "Abca1", "AI593442", "Car4", "Col6a2"
    ],
}

EC_SUBTYPES = {
    "aECs": [
        "Bmx", "Efnb2", "Vegfc", "Mgp", "Cytl1", "Sema3g",
        "Gkn3", "Fbln2", "Hey1", "Egfl8", "Jag1", "Igf2", "Clu"
    ],
    "capECs": [
        "Slc7a5", "Mfsd2a", "Tfrc", "Slc16a1", "Meox1", "Col4a3",
        "Angpt2", "Rgcc", "Cxcl12", "Ecscr", "Apln", "Car4"
    ],
    "vECs": [
        "Nr2f2", "Slc38a5", "Flrt2", "Ier3", "Ackr1",
        "Lcn2", "Vcam1", "Ly6c1", "Ly6a", "Ctsc"
    ],
}


def prepare_anndata(adata, counts, log_norm, celltype):
    adata.raw = ad.AnnData(
        X=adata.layers[counts].copy(),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )

    adata.X = adata.layers[log_norm].copy()
    adata = adata[adata.obs[celltype] != "Undefined"].copy()

    return adata


def score_subtypes(adata_sub, marker_dict, prefix, use_raw=False):
    marker_dict_filtered = {
        subtype: [gene for gene in genes if gene in adata_sub.var_names]
        for subtype, genes in marker_dict.items()
    }

    print(f"\nMarkers found for {prefix}:")
    for subtype, genes in marker_dict_filtered.items():
        print(f"{subtype}: {genes}")

    score_cols = []

    for subtype, genes in marker_dict_filtered.items():
        score_name = f"{subtype}_score"
        score_cols.append(score_name)

        if genes:
            sc.tl.score_genes(
                adata_sub,
                gene_list=genes,
                score_name=score_name,
                use_raw=use_raw,
                gene_pool=list(adata_sub.var_names),
            )
        else:
            adata_sub.obs[score_name] = np.nan

    adata_sub.obs[f"{prefix}_max_score"] = adata_sub.obs[score_cols].max(axis=1)
    adata_sub.obs[f"{prefix}_best_subtype"] = (
        adata_sub.obs[score_cols]
        .idxmax(axis=1)
        .str.replace("_score", "", regex=False)
    )

    return adata_sub, marker_dict_filtered, score_cols


def add_threshold_labels(adata_sub, prefix, thresholds):
    for threshold in thresholds:
        col = f"{prefix}_subtype_thr_{str(threshold).replace('.', '_')}"
        adata_sub.obs[col] = np.where(
            adata_sub.obs[f"{prefix}_max_score"] > threshold,
            adata_sub.obs[f"{prefix}_best_subtype"],
            f"{prefix}_unassigned",
        )

    return adata_sub


def save_umap(adata_sub, color_cols, output_path, title=None):
    sc.pp.neighbors(adata_sub)
    sc.tl.umap(adata_sub)

    fig = sc.pl.umap(
        adata_sub,
        color=color_cols,
        ncols=3,
        wspace=0.4,
        title=title,
        show=False,
        return_fig=True,
    )

    for ax in fig.axes:
        for coll in ax.collections:
            coll.set_rasterized(True)

    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_dotplots(adata_sub, marker_dict, groupby_cols, output_path):
    ncols = 3
    nrows = math.ceil(len(groupby_cols) / ncols)
    panel_images = []

    for groupby_col in groupby_cols:
        dp = sc.pl.dotplot(
            adata_sub,
            var_names=marker_dict,
            groupby=groupby_col,
            use_raw=False,
            show=False,
            return_fig=True,
        )

        dp.make_figure()
        fig = dp.fig
        fig.suptitle("")

        for ax in fig.axes:
            ax.set_title("")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        panel_images.append((groupby_col, plt.imread(buf)))

        buf.close()
        plt.close(fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (label, img) in zip(axes, panel_images):
        ax.imshow(img)
        ax.set_title(label, fontsize=12)
        ax.axis("off")

    for ax in axes[len(panel_images):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Score and visualise EC and SMC subtype thresholds."
    )

    parser.add_argument("--anndata_file", required=True, help="Path to AnnData .h5ad file")
    parser.add_argument("--out", required=True, help="Output folder")
    parser.add_argument("--counts", default="counts", help="Layer containing raw counts")
    parser.add_argument("--log_norm",default="librarysize_log1p_norm",help="Layer containing log1p-normalised counts")
    parser.add_argument("--cell_types",default="cell_type_incl_low_quality_revised",help="obs column containing cell types")
    parser.add_argument("--exclude_sample",default="aging_s1_r0",help="Sample to exclude. Use 'none' to keep all samples.")
    return parser.parse_args()

def main():
    args = parse_args()

    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    sc.settings.figdir = fig_dir

    print("Loading AnnData...")
    adata = sc.read_h5ad(args.anndata_file)

    print("Preparing AnnData...")
    adata = prepare_anndata(
        adata,
        counts=args.counts,
        log_norm=args.log_norm,
        celltype=args.cell_types,
    )

    if args.exclude_sample.lower() != "none":
        adata = adata[adata.obs["sample"] != args.exclude_sample].copy()

    thresholds = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]

    print("\nProcessing SMCs...")
    adata_smc = adata[adata.obs[args.cell_types] == "SMCs"].copy()

    adata_smc, smc_markers_filtered, smc_score_cols = score_subtypes(
        adata_smc,
        marker_dict=SMCs,
        prefix="SMC",
        use_raw=False,
    )

    adata_smc = add_threshold_labels(adata_smc, "SMC", thresholds)

    smc_thr_cols = [
        f"SMC_subtype_thr_{str(t).replace('.', '_')}"
        for t in thresholds
    ]

    for col in smc_thr_cols:
        print(f"\n{col}")
        print(adata_smc.obs[col].value_counts())

    save_umap(
        adata_smc,
        smc_thr_cols,
        os.path.join(fig_dir, "smc_thresholds_umap.pdf"),
    )

    save_dotplots(
        adata_smc,
        smc_markers_filtered,
        smc_thr_cols,
        os.path.join(fig_dir, "SMC_dotplots_all_thresholds_one_page.pdf"),
    )

    print("\nProcessing ECs...")
    adata_ec = adata[adata.obs[args.cell_types] == "ECs"].copy()

    adata_ec, ec_markers_filtered, ec_score_cols = score_subtypes(
        adata_ec,
        marker_dict=EC_SUBTYPES,
        prefix="EC",
        use_raw=False,
    )

    adata_ec = add_threshold_labels(adata_ec, "EC", thresholds)

    ec_thr_cols = [
        f"EC_subtype_thr_{str(t).replace('.', '_')}"
        for t in thresholds
    ]

    for col in ec_thr_cols:
        print(f"\n{col}")
        print(adata_ec.obs[col].value_counts())

    save_umap(
        adata_ec,
        ec_thr_cols,
        os.path.join(fig_dir, "ec_thresholds_umap.pdf"),
    )

    save_dotplots(
        adata_ec,
        ec_markers_filtered,
        ec_thr_cols,
        os.path.join(fig_dir, "EC_dotplots_all_thresholds_one_page.pdf"),
    )

    ec_col_0 = "EC_subtype_thr_0_0"
    adata_ec_0 = adata_ec[adata_ec.obs[ec_col_0] != "EC_unassigned"].copy()

    save_umap(
        adata_ec_0,
        ec_col_0,
        os.path.join(fig_dir, "ec_0_0_umap.pdf"),
        title="EC subtypes at threshold = 0.0",
    )

    smc_col_0 = "SMC_subtype_thr_0_0"

    save_umap(
        adata_smc,
        smc_col_0,
        os.path.join(fig_dir, "smc_threshold_0_0_umap.pdf"),
        title="SMC subtypes at threshold = 0.0",
    )

    dp = sc.pl.dotplot(
        adata_smc,
        var_names=smc_markers_filtered,
        groupby=smc_col_0,
        use_raw=False,
        show=False,
        return_fig=True,
    )

    dp.make_figure()
    fig = dp.fig
    fig.suptitle("")

    for ax in fig.axes:
        ax.set_title("")

    fig.savefig(
        os.path.join(fig_dir, "SMC_dotplot_threshold_0_0.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)

    print(f"\nDone. Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()