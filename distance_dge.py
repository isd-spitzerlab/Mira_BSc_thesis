#!/usr/bin/env python3

import argparse
import warnings
from pathlib import Path
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from adjustText import adjust_text
from matplotlib.lines import Line2D
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SELECTED_EC_MARKER_GENES = {
    "aEC": [
        "Bmx", "Efnb2", "Vegfc", "Mgp", "Cytl1", "Sema3g", "Gkn3",
        "Fbln2", "Hey1", "Egfl8", "Jag1", "Igf2", "Notch3", "Clu", "Gjb2",
    ],
    "capEC": [
        "Slc7a5", "Mfsd2a", "Tfrc", "Slc16a1", "Meox1", "Col4a3",
        "Angpt2", "Rgcc", "Cxcl12", "Ecscr", "Apln", "Car4",
    ],
    "vEC": [
        "Nr2f2", "Slc38a5", "Flrt2", "Ier3", "Ackr1", "Lcn2",
        "Vcam1", "Ly6c1", "Ly6a", "Ctsc",
    ],
}

EC_MARKER_GENES_FOR_VOLCANO = {
    **SELECTED_EC_MARKER_GENES,
    "EC": ["Kdr", "Mfsd2a", "Slc40a1", "Nostrin", "Itm2a", "Pde3a", "Rad54b"],
}

CONTAMINATION_GENES = {
    "Mural": ["Myh11", "Acta2", "Pdgfrb"],
    "Glia": ["Foxf1", "Cemip", "Aqp4", "Gfap", "Apod", "Htra1", "Sox9", "C4b", "Trf", "Cldn11", "Klk6"],
    "Neurons": ["Arpp21", "Gad2", "Dnm1", "Madd", "Slc4a10", "Cux2", "Lamp5"],
    "Immune": ["Mrc1"],
}


def safe_subset(adata, mask):
    return adata[np.asarray(mask), :].copy()


def normalize_log1p(pb_in):
    pb_out = pb_in.copy()
    sc.pp.normalize_total(pb_out, target_sum=1e6)
    sc.pp.log1p(pb_out)
    return pb_out


def build_pseudobulks(
    adata_in,
    subtype_col,
    sample_col,
    count_layer,
    keep_subtypes,
    min_cells_per_pb):
    adata = adata_in.copy()

    if keep_subtypes:
        mask = adata.obs[subtype_col].isin(keep_subtypes).to_numpy()
        adata = safe_subset(adata, mask)

    if count_layer not in adata.layers:
        raise KeyError(f"Layer '{count_layer}' not found. Available layers: {list(adata.layers.keys())}")
    if subtype_col not in adata.obs:
        raise KeyError(f"obs column '{subtype_col}' not found.")
    if sample_col not in adata.obs:
        raise KeyError(f"obs column '{sample_col}' not found.")

    X = adata.layers[count_layer]
    adata.obs["pb_id"] = (
        adata.obs[sample_col].astype(str) + "__" + adata.obs[subtype_col].astype(str)
    )

    pb_rows = []
    pb_meta = []

    for pb_id in adata.obs["pb_id"].unique():
        mask = (adata.obs["pb_id"] == pb_id).to_numpy()
        xi = X[mask]

        summed = xi.sum(axis=0)
        summed = np.asarray(summed).ravel()

        sub = adata.obs.loc[mask, :]
        pb_rows.append(summed)
        pb_meta.append(
            {
                "pb_id": pb_id,
                sample_col: sub[sample_col].iloc[0],
                subtype_col: sub[subtype_col].iloc[0],
                "sample": sub[sample_col].iloc[0],
                "ec_subtype": sub[subtype_col].iloc[0],
                "n_cells": int(mask.sum()),
            }
        )

    if not pb_rows:
        raise ValueError("No pseudo-bulks were created. Check subtype labels and filters.")

    pb = ad.AnnData(
        X=np.vstack(pb_rows),
        obs=pd.DataFrame(pb_meta).set_index("pb_id"),
        var=adata.var.copy(),
    )

    pb = safe_subset(pb, (pb.obs["n_cells"] >= min_cells_per_pb).to_numpy())
    return pb


def filter_genes_by_pseudobulk_counts(pb, min_total_counts,min_nonzero_pb):
    count_df = pd.DataFrame(pb.X, index=pb.obs_names, columns=pb.var_names)

    keep_genes = (
        (count_df.sum(axis=0) >= min_total_counts)
        & ((count_df > 0).sum(axis=0) >= min_nonzero_pb)
    )

    pb_filt = pb[:, keep_genes.to_numpy()].copy()
    count_df_filt = pd.DataFrame(
        pb_filt.X,
        index=pb_filt.obs_names,
        columns=pb_filt.var_names,
    ).astype(int)

    return pb_filt, count_df_filt


def get_gene_expr_df(pb_obj, genes):
    rows = []
    genes = [g for g in genes if g in pb_obj.var_names]

    for gene in genes:
        x = pb_obj[:, gene].X
        x = x.toarray().ravel() if sp.issparse(x) else np.asarray(x).ravel()

        tmp = pb_obj.obs[["sample", "ec_subtype", "n_cells"]].copy()
        tmp["gene"] = gene
        tmp["expr"] = x
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["sample", "ec_subtype", "n_cells", "gene", "expr"])

    return pd.concat(rows, axis=0)


def plot_marker_panel(expr_df, genes, title, outpath, subtype_order=None, ncols=4):
    genes = [g for g in genes if g in expr_df["gene"].unique()]
    if len(genes) == 0:
        print(f"No genes to plot for {title}")
        return

    nrows = int(np.ceil(len(genes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax, gene in zip(axes, genes):
        d = expr_df[expr_df["gene"] == gene].copy()
        cats = subtype_order if subtype_order is not None else list(d["ec_subtype"].unique())
        cats = [c for c in cats if c in d["ec_subtype"].unique()]
        xpos = {cat: i for i, cat in enumerate(cats)}

        for sample, ds in d.groupby("sample"):
            ds = ds[ds["ec_subtype"].isin(cats)].copy()
            ds["x"] = ds["ec_subtype"].map(xpos)
            ds = ds.sort_values("x")
            ax.plot(ds["x"], ds["expr"], marker="o", label=sample)
            for _, row in ds.iterrows():
                ax.text(row["x"] + 0.03, row["expr"], sample, fontsize=7)

        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_title(gene)
        ax.set_ylabel("log-normalized expression")

    for ax in axes[len(genes):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    fig.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_program_scores(pb_obj, marker_dict):
    score_df = pb_obj.obs[["sample", "ec_subtype", "n_cells"]].copy()

    for program, genes in marker_dict.items():
        genes = [g for g in genes if g in pb_obj.var_names]
        if len(genes) == 0:
            score_df[f"{program}_score"] = np.nan
            continue

        x = pb_obj[:, genes].X
        if sp.issparse(x):
            x = x.toarray()

        score_df[f"{program}_score"] = np.asarray(x).mean(axis=1)

    return score_df


def plot_program_scores(score_df, outpath, subtype_order=None):
    programs = [c for c in score_df.columns if c.endswith("_score")]
    if not programs:
        return

    fig, axes = plt.subplots(1, len(programs), figsize=(4 * len(programs), 4), squeeze=False)

    for ax, score_col in zip(axes[0], programs):
        d = score_df.copy()
        cats = subtype_order if subtype_order is not None else list(d["ec_subtype"].unique())
        cats = [c for c in cats if c in d["ec_subtype"].unique()]
        xpos = {cat: i for i, cat in enumerate(cats)}

        for sample, ds in d.groupby("sample"):
            ds = ds[ds["ec_subtype"].isin(cats)].copy()
            ds["x"] = ds["ec_subtype"].map(xpos)
            ds = ds.sort_values("x")
            ax.plot(ds["x"], ds[score_col], marker="o", label=sample)
            for _, row in ds.iterrows():
                ax.text(row["x"] + 0.03, row[score_col], sample, fontsize=7)

        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_title(score_col.replace("_score", ""))
        ax.set_ylabel("mean log-normalized marker expression")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_scanpy_de(pb_in, group, reference):
    tmp = normalize_log1p(pb_in)
    mask = tmp.obs["ec_subtype"].isin([group, reference]).to_numpy()
    tmp = safe_subset(tmp, mask)

    sc.tl.rank_genes_groups(
        tmp,
        groupby="ec_subtype",
        groups=[group],
        reference=reference,
        method="wilcoxon",
    )

    df = sc.get.rank_genes_groups_df(tmp, group=group)
    df["contrast"] = f"{group}_vs_{reference}"
    df["method"] = "scanpy_wilcoxon"
    return df


def run_deseq2(count_df, meta_df, group, reference):
    keep = meta_df["ec_subtype"].isin([group, reference])
    counts_sub = count_df.loc[keep].copy()
    meta_sub = meta_df.loc[keep, ["ec_subtype", "sample", "n_cells"]].copy()

    meta_sub["ec_subtype"] = pd.Categorical(
        meta_sub["ec_subtype"],
        categories=[reference, group],
    )

    dds = DeseqDataSet(
        counts=counts_sub,
        metadata=meta_sub,
        design="~ ec_subtype",
        refit_cooks=True,
    )
    dds.deseq2()

    stats = DeseqStats(dds, contrast=["ec_subtype", group, reference])
    stats.summary()

    res = stats.results_df.reset_index().rename(columns={"index": "gene"})
    if "gene" not in res.columns:
        res = res.rename(columns={res.columns[0]: "gene"})
    res["contrast"] = f"{group}_vs_{reference}"
    res["method"] = "pydeseq2"
    return res


def volcano_plot_labelled_colours(
    res,
    title="",
    gene_col="gene",
    lfc_col="log2FoldChange",
    padj_col="padj",
    lfc_thr=1.0,
    padj_thr=0.1,
    n_label_each_side=20,
    ec_markers=None,
    contamination=None,
    outpath=None,
):
    res = res.copy()
    res = res.replace([np.inf, -np.inf], np.nan).dropna(subset=[lfc_col, padj_col])
    if res.empty:
        print(f"No finite DESeq2 results for {title}; skipping volcano.")
        return

    res["log10padj"] = -np.log10(res[padj_col].clip(lower=1e-300))

    gene_to_group = {}
    if ec_markers is not None:
        for group, genes in ec_markers.items():
            for gene in genes:
                gene_to_group[gene] = group
    if contamination is not None:
        for group, genes in contamination.items():
            for gene in genes:
                gene_to_group[gene] = group

    res["marker_group"] = res[gene_col].map(gene_to_group)
    res["is_marker"] = res["marker_group"].notna()

    group_order = ["aEC", "capEC", "vEC", "EC", "Mural", "Glia", "Neurons", "Immune"]
    color_map = {
        "aEC": "red",
        "capEC": "blue",
        "vEC": "green",
        "EC": "navy",
        "Mural": "orange",
        "Glia": "purple",
        "Neurons": "brown",
        "Immune": "olive",
    }

    sig = res[(res[padj_col] < padj_thr) & (res[lfc_col].abs() > lfc_thr)].copy()
    up = sig[sig[lfc_col] > 0].sort_values([padj_col, lfc_col], ascending=[True, False]).head(n_label_each_side)
    down = sig[sig[lfc_col] < 0].sort_values([padj_col, lfc_col], ascending=[True, True]).head(n_label_each_side)
    label_df = pd.concat([up, down], axis=0)
    label_df = label_df[label_df["is_marker"]].copy()
    label_df["label_color"] = label_df["marker_group"].map(color_map).fillna("black")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(res[lfc_col], res["log10padj"], s=18, alpha=0.5, color="lightgrey", zorder=1)

    sig_all = res[(res[lfc_col].abs() > lfc_thr) & (res["log10padj"] > -np.log10(padj_thr))]
    if not sig_all.empty:
        ax.scatter(sig_all[lfc_col], sig_all["log10padj"], s=18, alpha=0.7, color="dimgray", zorder=2)

    ax.axvline(lfc_thr, color="blue", linestyle="--", lw=0.8)
    ax.axvline(-lfc_thr, color="blue", linestyle="--", lw=0.8)
    ax.axhline(-np.log10(padj_thr), color="blue", linestyle="--", lw=0.8)

    texts = [
        ax.text(
            row[lfc_col],
            row["log10padj"],
            str(row[gene_col]),
            color=row["label_color"],
            fontsize=8,
            fontweight="bold",
            zorder=4,
        )
        for _, row in label_df.iterrows()
    ]

    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.6, color="grey"))

    ax.set_xlabel("log2 fold-change")
    ax.set_ylabel("-log10(Padj)")
    ax.set_title(title)

    groups_present = label_df["marker_group"].dropna().unique().tolist()
    ordered_groups_present = [g for g in group_order if g in groups_present]
    if ordered_groups_present:
        handles = [
            Line2D([0], [0], marker="o", linestyle="", color=color_map[g], label=g, markersize=6)
            for g in ordered_groups_present
        ]
        ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    if outpath is not None:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_top_genes_for_raw_plot(deseq_df, n_genes, lfc_col = "log2FoldChange", padj_col = "padj",gene_col = "gene"):
    d = deseq_df.copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[lfc_col, padj_col])
    d = d.sort_values([padj_col, lfc_col], ascending=[True, False])

    up = d[d[lfc_col] > 0].head(n_genes // 2)
    down = d[d[lfc_col] < 0].sort_values([padj_col, lfc_col], ascending=[True, True]).head(n_genes // 2)

    genes = pd.Index(up[gene_col].tolist() + down[gene_col].tolist()).unique().tolist()
    return genes[:n_genes]


def plot_raw_pseudobulk_counts(pb_in, genes, group, reference, outpath):
    genes = [g for g in genes if g in pb_in.var_names]
    if len(genes) == 0:
        print(f"No genes available to plot for {group} vs {reference}")
        return

    mask = pb_in.obs["ec_subtype"].isin([group, reference]).to_numpy()
    pb_sub = safe_subset(pb_in, mask)
    subtype_order = [group, reference]
    xpos = {k: i for i, k in enumerate(subtype_order)}

    fig, axes = plt.subplots(1, len(genes), figsize=(4 * len(genes), 4), squeeze=False)
    axes = axes[0]

    for ax, gene in zip(axes, genes):
        x = pb_sub[:, gene].X
        vals = x.toarray().ravel() if sp.issparse(x) else np.asarray(x).ravel()

        d = pb_sub.obs[["sample", "ec_subtype", "n_cells"]].copy()
        d["raw_count"] = vals

        for sample, ds in d.groupby("sample"):
            ds = ds.copy()
            ds["x"] = ds["ec_subtype"].map(xpos)
            ds = ds.sort_values("x")

            ax.plot(ds["x"], ds["raw_count"], marker="o", label=sample)
            for _, row in ds.iterrows():
                ax.text(row["x"] + 0.03, row["raw_count"], sample, fontsize=7)

        ax.set_xticks(range(len(subtype_order)))
        ax.set_xticklabels(subtype_order, rotation=45, ha="right")
        ax.set_title(gene)
        ax.set_ylabel("raw pseudo-bulk count")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    fig.suptitle(f"Raw pseudo-bulk counts: {group} vs {reference}", y=1.04)
    plt.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pca(pb_plot, outpath):
    sc.pp.pca(pb_plot)
    fig = sc.pl.pca(
        pb_plot,
        color=["ec_subtype", "sample", "n_cells"],
        wspace=0.35,
        show=False,
        return_fig=True,
    )
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Pseudo-bulk EC subtype DGE CLI.")
    parser.add_argument("--anndata_file", required=True, help="Input AnnData .h5ad file")
    parser.add_argument("--out", required=True, help="Output folder")
    parser.add_argument("--count_layer", default="counts", help="Layer containing raw counts")
    parser.add_argument("--sample_col", default="sample", help="obs column with sample IDs")
    parser.add_argument("--subtype_col", default="ec_subtype", help="obs column with EC subtype labels")
    parser.add_argument("--subtypes", default="aEC,capEC,vEC", help="Comma-separated EC subtypes to keep")
    parser.add_argument("--non_ec_labels", default="non_EC,non_ECs", help="Comma-separated labels to exclude")
    parser.add_argument("--min_cells_per_pb", type=int, default=20)
    parser.add_argument("--min_total_counts", type=int, default=20)
    parser.add_argument("--min_nonzero_pb", type=int, default=2)
    parser.add_argument("--n_top_table", type=int, default=20)
    parser.add_argument("--n_top_raw_plots", type=int, default=8)
    parser.add_argument("--skip_deseq2", action="store_true", help="Run Scanpy DE and plots, but skip PyDESeq2")
    return parser.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.out)
    fig_dir = outdir / "figures"
    table_dir = outdir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = str(fig_dir)

    subtypes = [s.strip() for s in args.subtypes.split(",") if s.strip()]
    non_ec_labels = [s.strip() for s in args.non_ec_labels.split(",") if s.strip()]
    contrasts = [("aEC", "capEC"), ("aEC", "vEC"), ("capEC", "vEC")]

    print("Loading AnnData...")
    adata = sc.read_h5ad(args.anndata_file)

    if args.subtype_col not in adata.obs:
        raise KeyError(f"obs column '{args.subtype_col}' not found. Available columns include: {list(adata.obs.columns[:20])}")

    # Filter out non-EC cells and keep requested subtypes.
    mask = ~adata.obs[args.subtype_col].isin(non_ec_labels)
    mask &= adata.obs[args.subtype_col].isin(subtypes)
    adata = safe_subset(adata, mask.to_numpy())

    print(f"Cells after EC subtype filtering: {adata.n_obs:,}")

    print("Building pseudo-bulks...")
    pb = build_pseudobulks(
        adata_in=adata,
        subtype_col=args.subtype_col,
        sample_col=args.sample_col,
        count_layer=args.count_layer,
        keep_subtypes=subtypes,
        min_cells_per_pb=args.min_cells_per_pb,
    )

    pb.obs.sort_values(["ec_subtype", "sample"]).to_csv(table_dir / "pseudobulk_metadata.csv")
    pb.obs.groupby("ec_subtype")["sample"].nunique().to_csv(table_dir / "replicates_per_subtype.csv")
    pb.obs.groupby("ec_subtype")["n_cells"].describe().to_csv(table_dir / "cells_per_subtype_summary.csv")

    ec_numbers = pb.obs.pivot_table(
        index="sample",
        columns="ec_subtype",
        values="n_cells",
        aggfunc="sum",
    )
    ec_numbers.to_csv(table_dir / "ec_numbers.csv")

    print("Filtering genes...")
    pb_filt, count_df = filter_genes_by_pseudobulk_counts(
        pb,
        min_total_counts=args.min_total_counts,
        min_nonzero_pb=args.min_nonzero_pb,
    )

    print(f"Before gene filtering: {pb.shape[1]} genes")
    print(f"After gene filtering:  {pb_filt.shape[1]} genes")

    pb_plot = normalize_log1p(pb_filt)
    save_pca(pb_plot, str(fig_dir / "pseudobulk_pca.pdf"))

    all_marker_genes = sorted(set(g for genes in SELECTED_EC_MARKER_GENES.values() for g in genes))
    expr_df = get_gene_expr_df(pb_plot, all_marker_genes)
    expr_df = expr_df[expr_df["ec_subtype"] != "unassigned_EC"].copy()
    expr_df.to_csv(table_dir / "marker_gene_expression_long.csv", index=False)

    subtype_order = ["aEC", "capEC", "vEC"]
    for subtype, genes in SELECTED_EC_MARKER_GENES.items():
        plot_marker_panel(
            expr_df,
            genes,
            title=f"{subtype} marker genes across pseudo-bulks",
            outpath=str(fig_dir / f"{subtype}_marker_panel.pdf"),
            subtype_order=subtype_order,
            ncols=4,
        )

    score_df = compute_program_scores(pb_plot, SELECTED_EC_MARKER_GENES)
    score_df.to_csv(table_dir / "program_scores.csv")
    plot_program_scores(score_df, str(fig_dir / "program_scores.pdf"), subtype_order=subtype_order)

    meta_df = pb_filt.obs.copy()
    all_scanpy = []
    all_deseq = []

    for group, reference in contrasts:
        if group not in pb_filt.obs["ec_subtype"].unique() or reference not in pb_filt.obs["ec_subtype"].unique():
            print(f"Skipping {group}_vs_{reference}: one or both groups are absent.")
            continue

        contrast_name = f"{group}_vs_{reference}"
        print(f"\n=== {contrast_name} ===")

        scanpy_df = run_scanpy_de(pb_filt, group, reference)
        scanpy_df.to_csv(table_dir / f"{contrast_name}_scanpy.csv", index=False)
        all_scanpy.append(scanpy_df)

        print("Top Scanpy hits:")
        print(scanpy_df.head(args.n_top_table).to_string(index=False))

        if not args.skip_deseq2:
            deseq_df = run_deseq2(count_df, meta_df, group, reference)
            deseq_df.to_csv(table_dir / f"{contrast_name}_pydeseq2.csv", index=False)
            all_deseq.append(deseq_df)

            print("Top DESeq2 hits:")
            print(deseq_df.sort_values("padj").head(args.n_top_table).to_string(index=False))

            volcano_plot_labelled_colours(
                deseq_df,
                title=contrast_name,
                gene_col="gene",
                lfc_col="log2FoldChange",
                padj_col="padj",
                lfc_thr=1.0,
                padj_thr=0.1,
                n_label_each_side=20,
                ec_markers=EC_MARKER_GENES_FOR_VOLCANO,
                contamination=CONTAMINATION_GENES,
                outpath=str(fig_dir / f"{contrast_name}_labelled_volcano.pdf"),
            )

            top_raw_genes = select_top_genes_for_raw_plot(
                deseq_df,
                n_genes=args.n_top_raw_plots,
            )
            print("Genes chosen for raw-count inspection:", top_raw_genes)

            plot_raw_pseudobulk_counts(
                pb_filt,
                genes=top_raw_genes,
                group=group,
                reference=reference,
                outpath=str(fig_dir / f"{contrast_name}_raw_counts.pdf"),
            )

    if all_scanpy:
        pd.concat(all_scanpy, ignore_index=True).to_csv(table_dir / "all_scanpy_pairwise.csv", index=False)

    if all_deseq:
        pd.concat(all_deseq, ignore_index=True).to_csv(table_dir / "all_pydeseq2_pairwise.csv", index=False)

    print(f"\nDone. Results saved to: {outdir}")


if __name__ == "__main__":
    main()
