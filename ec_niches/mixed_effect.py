#!/usr/bin/env python3

import dask
dask.config.set({"dataframe.query-planning": True})

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import itertools
import os
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
from scipy.spatial import cKDTree
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

EC_SUBTYPES = {
    "aECs": [
        "Bmx", "Efnb2", "Vegfc", "Mgp", "Cytl1", "Sema3g", "Gkn3",
        "Fbln2", "Hey1", "Egfl8", "Jag1", "Igf2", "Notch3", "Mgp", "Clu",
    ],
    "capECs": [
        "Slc7a5", "Mfsd2a", "Tfrc", "Slc16a1", "Meox1", "Col4a3",
        "Angpt2", "Rgcc", "Cxcl12", "Ecscr", "Apln", "Car4",
    ],
    "vECs": [
        "Nr2f2", "Slc38a5", "Flrt2", "Ier3", "Ackr1",
        "Lcn2", "Vcam1", "Ly6c1", "Ly6a", "Ctsc",
    ],
}

SUBTYPE_ORDER = ["aECs", "capECs", "vECs"]
DISTANCE_METRICS = ["dist_to_nearest_smc", "dist_to_nearest_pericyte"]
TITLE_MAP = {
    "dist_to_nearest_smc": "Distance to nearest SMC",
    "dist_to_nearest_pericyte": "Distance to nearest Pericyte",
}


def parse_comma_list(value):
    if value is None or str(value).strip().lower() in {"", "none"}:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_radii(value):
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def safe_name(value):
    return str(value).replace("/", "_").replace(" ", "_")


def flatten_columns(df):
    df = df.copy()
    df.columns = [
        "_".join(map(str, c)).rstrip("_") if isinstance(c, tuple) else c
        for c in df.columns
    ]
    return df


def prepare_anndata(adata, counts, log_norm, celltype):
    adata.raw = ad.AnnData(
        X=adata.layers[counts].copy(),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    adata.X = adata.layers[log_norm].copy()
    mask = (adata.obs[celltype] != "Undefined").to_numpy()
    adata = adata[mask, :].copy()
    return adata


def annotate_brain_areas(adata, brain_areas_csv):
    brain_areas_csv = brain_areas_csv.rename(columns={"Unnamed: 0": "cell_id"})
    brain_areas_csv = brain_areas_csv.set_index("cell_id")
    brain_areas_csv = brain_areas_csv.reindex(index=adata.obs.index)

    adata.obs["brain_area"] = brain_areas_csv["label"]
    adata.obs["brain_area"] = (
        adata.obs["brain_area"].astype("string").str.replace("/", "_", regex=False)
    )
    mask = adata.obs["brain_area"].notna().to_numpy()
    adata = adata[mask, :].copy()
    return adata


def compute_nneighbours_per_celltype(adata, key_added, celltype_col):
    connectivities = adata.obsp[f"{key_added}_connectivities"].tocsr()
    celltypes_series = adata.obs[celltype_col].astype("category")
    celltype_categories = list(celltypes_series.cat.categories)

    composition_matrix = np.zeros((adata.n_obs, len(celltype_categories)))

    for i, ct in enumerate(celltype_categories):
        mask_filter = (celltypes_series == ct).values.astype(float)
        composition_matrix[:, i] = connectivities.dot(mask_filter)

    for i, ct in enumerate(celltype_categories):
        adata.obs[f"nhood_{ct}"] = composition_matrix[:, i]
        adata.obsm[f"nhood_{ct}"] = composition_matrix[:, i]

    adata.obsm["celltype_nhoods"] = composition_matrix
    adata.uns["celltype_nhood_categories"] = celltype_categories
    return adata


def annotate_ec_subtypes(ec_adata):
    markers_filtered = {
        subtype: [gene for gene in genes if gene in ec_adata.var_names]
        for subtype, genes in EC_SUBTYPES.items()
    }

    for subtype, markers in markers_filtered.items():
        if markers:
            sc.tl.score_genes(
                ec_adata,
                gene_list=markers,
                score_name=subtype,
                use_raw=False,
                gene_pool=list(ec_adata.var_names),
            )
        else:
            ec_adata.obs[subtype] = np.nan

    score_cols = list(EC_SUBTYPES.keys())
    max_scores = ec_adata.obs[score_cols].max(axis=1)
    best_subtypes = ec_adata.obs[score_cols].idxmax(axis=1)

    ec_adata.obs["ec_subtype"] = best_subtypes
    ec_adata.obs.loc[max_scores <= 0, "ec_subtype"] = "unassigned"
    mask = (ec_adata.obs["ec_subtype"] != "unassigned").to_numpy()
    ec_adata = ec_adata[mask, :].copy()
    return ec_adata


def calculate_ec_niches(
    adata,
    ec_adata,
    celltype_col,
    celltype_nhoods,
    ec_subtype_col,
    age_col,
    *,
    age=None,
    sample_col="sample",
    brain_area_col="brain_area",
):
    celltypes = list(
        adata.uns.get(
            "celltype_nhood_categories",
            adata.obs[celltype_col].astype("category").cat.categories,
        )
    )
    x = ec_adata.obsm[celltype_nhoods]

    if x.shape[1] != len(celltypes):
        raise ValueError(
            f"{celltype_nhoods} has {x.shape[1]} columns, but {len(celltypes)} cell types were found."
        )

    ec_niches = pd.DataFrame(x, columns=celltypes, index=ec_adata.obs.index)
    ec_niches["EC_subtypes"] = ec_adata.obs[ec_subtype_col]
    ec_niches["age_months"] = pd.to_numeric(ec_adata.obs[age_col], errors="coerce")
    ec_niches["sample"] = ec_adata.obs[sample_col]
    ec_niches["EC_cell_ID"] = ec_adata.obs_names
    ec_niches["brain_area"] = ec_adata.obs[brain_area_col]

    ec_niches = ec_niches.drop(columns=["Undefined"], errors="ignore")
    celltype_cols = [c for c in celltypes if c in ec_niches.columns and c != "Undefined"]
    ec_niches["row_sum"] = ec_niches[celltype_cols].sum(axis=1)
    ec_niches = ec_niches.loc[ec_niches["row_sum"] != 0].copy()

    if age is not None:
        ec_niches = ec_niches[ec_niches["age_months"] == float(age)].copy()

    return ec_niches


def build_niches_long(adata, radii, celltype_col, age_col, age):
    areas = adata.obs["brain_area"].dropna().unique()
    all_niches = []

    for radius in radii:
        print(f"\n=== Processing radius {radius} µm ===")
        radius_key = f"spatial_multi_{radius}"

        sq.gr.spatial_neighbors(
            adata,
            spatial_key="spatial_microns",
            coord_type="generic",
            radius=radius,
            delaunay=False,
            key_added=radius_key,
            library_key="sample",
        )

        adata = compute_nneighbours_per_celltype(adata, radius_key, celltype_col)

        ec_mask = (adata.obs[celltype_col] == "ECs").to_numpy()
        ec_adata = adata[ec_mask, :].copy()
        ec_adata = annotate_ec_subtypes(ec_adata)

        for area in areas:
            area_mask = (ec_adata.obs["brain_area"] == area).to_numpy()
            if area_mask.sum() == 0:
                print(f"Skipping {area}: no EC cells")
                continue

            temp_adata = ec_adata[area_mask, :].copy()
            if temp_adata.n_obs == 0:
                print(f"Skipping {area}: no EC cells")
                continue

            ec_niches = calculate_ec_niches(
                adata,
                temp_adata,
                celltype_col,
                "celltype_nhoods",
                "ec_subtype",
                age_col,
                age=age,
            )
            ec_niches["radius"] = radius
            all_niches.append(ec_niches)

    if not all_niches:
        raise RuntimeError(
            "No EC niches were created. Check cell type labels, brain areas, age filter, and radii."
        )

    final_niches_df = pd.concat(all_niches, axis=0, ignore_index=True)

    cell_types = list(adata.obs[celltype_col].astype("category").cat.categories)
    final_niches_df = final_niches_df.rename(
        columns=lambda x: x.replace("-", "_") if x in cell_types else x
    )

    value_vars = [
        c.replace("-", "_")
        for c in cell_types
        if c.replace("-", "_") in final_niches_df.columns
    ]
    final_niches_df_long = final_niches_df.melt(
        id_vars=["EC_subtypes", "sample", "EC_cell_ID", "brain_area", "radius"],
        value_vars=value_vars,
        var_name="cell_type",
        value_name="cell_count",
    )

    final_niches_df_long["EC_cell_ID"] = (
        final_niches_df_long["EC_cell_ID"].astype(str).str.replace("-", "_", regex=False)
    )
    return final_niches_df, final_niches_df_long


def compute_ec_mural_distances(adata, celltype_col):
    coords = adata.obsm["spatial_microns"]

    ec_mask = adata.obs["ec_subtype"].notna()
    smc_mask = adata.obs[celltype_col] == "SMCs"
    peri_mask = adata.obs[celltype_col] == "Pericytes"

    print("ECs with subtype:", int(ec_mask.sum()))
    print("SMCs:", int(smc_mask.sum()))
    print("Pericytes:", int(peri_mask.sum()))

    if ec_mask.sum() == 0 or smc_mask.sum() == 0 or peri_mask.sum() == 0:
        raise RuntimeError("Cannot compute mural distances: ECs, SMCs, or Pericytes are missing.")

    ec_coords = coords[ec_mask.to_numpy(), :]
    smc_coords = coords[smc_mask.to_numpy(), :]
    peri_coords = coords[peri_mask.to_numpy(), :]

    smc_tree = cKDTree(smc_coords)
    peri_tree = cKDTree(peri_coords)

    dist_smc, _ = smc_tree.query(ec_coords, k=1)
    dist_peri, _ = peri_tree.query(ec_coords, k=1)

    ec_df = adata.obs.loc[ec_mask, ["brain_area", "sample", "ec_subtype"]].copy()
    ec_df["dist_to_nearest_smc"] = dist_smc
    ec_df["dist_to_nearest_pericyte"] = dist_peri
    ec_df["peri_minus_smc"] = dist_peri - dist_smc

    return ec_df


def plot_ec_distance_histograms(
    ec_df,
    output_dir,
    brain_area=None,
    subtypes=("aECs", "capECs", "vECs"),
    column="peri_minus_smc",
    bin_width=None,
    bins=40,
    tick_interval=100,
    density=False,
):
    df = ec_df.copy()
    if brain_area is not None:
        df = df[df["brain_area"] == brain_area].copy()
    if df.empty:
        return

    xmin = tick_interval * np.floor(df[column].min() / tick_interval)
    xmax = tick_interval * np.ceil(df[column].max() / tick_interval)
    xticks = np.arange(xmin, xmax + tick_interval, tick_interval)
    bin_edges = np.arange(xmin, xmax + bin_width, bin_width) if bin_width is not None else bins

    fig, axes = plt.subplots(
        1,
        len(subtypes),
        figsize=(5 * len(subtypes), 4),
        sharex=True,
        sharey=True,
    )
    if len(subtypes) == 1:
        axes = [axes]

    for ax, subtype in zip(axes, subtypes):
        subset = df.loc[df["ec_subtype"] == subtype, column]
        ax.hist(subset, bins=bin_edges, density=density, alpha=0.45)
        ax.axvline(0, linestyle="--")
        ax.set_title(subtype)
        ax.set_xticks(xticks)
        ax.set_xlim(xmin, xmax)

    ylabel = "Density" if density else "Count"
    fig.supxlabel("Distance to nearest Pericyte - distance to nearest SMC (µm)")
    fig.supylabel(ylabel)
    title_area = brain_area if brain_area is not None else "All brain areas"
    fig.suptitle(f"{title_area} ECs: relative proximity to Pericytes vs SMCs", y=1.02)
    plt.tight_layout()

    fig.savefig(
        os.path.join(output_dir, f"{safe_name(title_area)}_rel_dist_mural.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def compute_sample_distance_stats(
    ec_df,
    sample_col="sample",
    subtype_col="ec_subtype",
    brain_area_col="brain_area",
    metrics=tuple(DISTANCE_METRICS),
):
    sample_stats = (
        ec_df.groupby([brain_area_col, sample_col, subtype_col], observed=True)[list(metrics)]
        .agg(["mean", "std"])
        .reset_index()
    )
    return flatten_columns(sample_stats)


def distance_stats_to_means(sample_distance_stats, metrics=tuple(DISTANCE_METRICS)):
    sample_means = sample_distance_stats[
        ["brain_area", "sample", "ec_subtype"] + [f"{m}_mean" for m in metrics]
    ].copy()
    sample_means = sample_means.rename(columns={f"{m}_mean": m for m in metrics})
    return sample_means


def plot_ec_distance_errorbars(
    sample_distance_stats,
    output_dir,
    brain_area=None,
    subtypes=("aECs", "capECs", "vECs"),
    metrics=("dist_to_nearest_smc", "dist_to_nearest_pericyte"),
    sample_col="sample",
    subtype_col="ec_subtype",
    brain_area_col="brain_area",
    jitter=0.08,
    point_size=40,
    alpha=0.9,
    random_seed=0,
    filename_suffix="mean_sd_dist_mural",
    add_title_suffix="mean ± SD distances",
):
    df = sample_distance_stats.loc[
        sample_distance_stats[brain_area_col] == brain_area
    ].copy()
    df = df[df[subtype_col].isin(subtypes)].copy()

    if df.empty:
        return

    samples = sorted(df[sample_col].dropna().unique())
    sample_colors = {sample: f"C{i % 10}" for i, sample in enumerate(samples)}
    rng = np.random.default_rng(random_seed)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        for i, subtype in enumerate(subtypes, start=1):
            subtype_df = df.loc[df[subtype_col] == subtype]

            for _, row in subtype_df.iterrows():
                x = i + rng.uniform(-jitter, jitter)
                color = sample_colors[row[sample_col]]

                ax.errorbar(
                    x,
                    row[mean_col],
                    yerr=row[std_col],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    elinewidth=1.2,
                    capsize=3,
                    markersize=np.sqrt(point_size),
                    alpha=alpha,
                    zorder=3,
                )

        ax.set_xticks(range(1, len(subtypes) + 1))
        ax.set_xticklabels(subtypes)
        ax.set_title(TITLE_MAP.get(metric, metric))
        ax.set_ylabel("Distance (µm)")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=sample_colors[sample],
            label=sample,
            markersize=6,
        )
        for sample in samples
    ]

    axes[-1].legend(handles=handles, title="Sample", loc="best")
    fig.suptitle(f"{brain_area}: EC sample-level {add_title_suffix} by subtype", y=1.02)
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"{safe_name(brain_area)}_{filename_suffix}.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def summarize_cell_counts(ec_df, adata, celltype_col):
    ec_counts = ec_df.groupby("sample", observed=True).size().rename("n_EC")
    ec_subtype_counts = (
        ec_df.groupby(["sample", "ec_subtype"], observed=True)
        .size()
        .unstack(fill_value=0)
    )

    obs = adata.obs
    pericytes = (
        obs.loc[obs[celltype_col] == "Pericytes"]
        .groupby("sample", observed=True)
        .size()
        .rename("n_pericytes")
    )
    smcs = (
        obs.loc[obs[celltype_col] == "SMCs"]
        .groupby("sample", observed=True)
        .size()
        .rename("n_SMCs")
    )

    summary = pd.concat([ec_counts, ec_subtype_counts, pericytes, smcs], axis=1)
    summary = summary.fillna(0).astype(int)
    summary["n_total_cells"] = summary["n_EC"] + summary["n_pericytes"] + summary["n_SMCs"]
    return summary.sort_index()


def get_facet_axis(facetgrid, row_index, col_index):
    if len(facetgrid.row_names) > 1:
        return facetgrid.axes[row_index, col_index]
    return facetgrid.axes[col_index]


def add_sample_errorbar_points(
    facetgrid,
    stats_df,
    subtype_col,
    sample_col,
    radius_order,
    sample_order,
    sample_palette,
):
    x_lookup = {radius: index for index, radius in enumerate(radius_order)}
    jitter_values = np.linspace(-0.12, 0.12, max(len(sample_order), 1))
    jitter_map = dict(zip(sample_order, jitter_values))

    for row_index, row_name in enumerate(facetgrid.row_names):
        for col_index, col_name in enumerate(facetgrid.col_names):
            ax = get_facet_axis(facetgrid, row_index, col_index)

            sub = stats_df[
                (stats_df[subtype_col] == row_name)
                & (stats_df["brain_area"] == col_name)
            ].copy()

            if sub.empty:
                continue

            sub["xpos"] = sub["radius"].map(x_lookup)
            sub["xpos_jitter"] = sub["xpos"] + sub[sample_col].map(jitter_map)

            for sample in sample_order:
                sample_df = sub[sub[sample_col] == sample]

                if sample_df.empty:
                    continue

                ax.errorbar(
                    sample_df["xpos_jitter"],
                    sample_df["mean"],
                    yerr=sample_df["std"],
                    fmt="o",
                    color=sample_palette[sample],
                    ecolor=sample_palette[sample],
                    elinewidth=1.1,
                    capsize=3,
                    markersize=5,
                    alpha=0.95,
                    zorder=10,
                    label=str(sample),
                )


def add_sample_legend(fig, sample_order, sample_palette):
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=sample_palette[sample],
            markeredgecolor=sample_palette[sample],
            label=str(sample),
        )
        for sample in sample_order
    ]

    fig.legend(
        handles=handles,
        title="Sample",
        loc="upper right",
        frameon=False,
    )


def plot_mural_counts_by_radius(final_niches_df_long, output_dir):
    subtype_col = "EC_subtypes"
    sample_col = "sample"

    ec_order = ["aECs", "capECs", "vECs"]

    mural_df = final_niches_df_long[
        final_niches_df_long["cell_type"].isin(["SMCs", "Pericytes"])
    ].copy()

    if mural_df.empty:
        print("Skipping mural count plots: no SMCs or Pericytes in long niche table.")
        return

    radius_order = sorted(mural_df["radius"].dropna().unique())
    area_order = sorted(mural_df["brain_area"].dropna().unique())
    sample_order = sorted(mural_df[sample_col].dropna().unique())

    sample_colors = [
        "blue",
        "orange",
        "green",
        "purple",
        "red",
        "brown",
        "pink",
        "gray",
    ]
    sample_palette = dict(zip(sample_order, sample_colors[: len(sample_order)]))

    sample_stats = (
        mural_df.groupby(
            ["brain_area", "radius", subtype_col, "cell_type", sample_col],
            as_index=False,
            observed=True,
        )["cell_count"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plot_configs = [
        ("SMCs", "SMC count per EC niche, sample mean ± SD", "smc_counts_per_ec_subtype.pdf"),
        ("Pericytes", "Pericyte count per EC niche, sample mean ± SD", "peri_counts_per_ec_subtype.pdf"),
    ]

    for cell_type, ylabel, filename in plot_configs:
        plot_stats = sample_stats[sample_stats["cell_type"] == cell_type].copy()

        if plot_stats.empty:
            continue

        scaffold = (
            mural_df[mural_df["cell_type"] == cell_type]
            [["brain_area", "radius", subtype_col]]
            .drop_duplicates()
            .copy()
        )
        scaffold["cell_count"] = np.nan

        facetgrid = sns.FacetGrid(
            scaffold,
            col="brain_area",
            row=subtype_col,
            col_order=area_order,
            row_order=ec_order,
            height=3.5,
            aspect=1.2,
            sharey=False,
            margin_titles=True,
        )

        for row_index, _ in enumerate(facetgrid.row_names):
            for col_index, _ in enumerate(facetgrid.col_names):
                ax = get_facet_axis(facetgrid, row_index, col_index)
                ax.set_xticks(range(len(radius_order)))
                ax.set_xticklabels(radius_order, rotation=45)

        add_sample_errorbar_points(
            facetgrid=facetgrid,
            stats_df=plot_stats,
            subtype_col=subtype_col,
            sample_col=sample_col,
            radius_order=radius_order,
            sample_order=sample_order,
            sample_palette=sample_palette,
        )

        facetgrid.set_axis_labels("Radius (µm)", ylabel)
        facetgrid.set_titles(row_template="{row_name}", col_template="{col_name}")

        add_sample_legend(facetgrid.fig, sample_order, sample_palette)

        facetgrid.fig.tight_layout()
        facetgrid.fig.savefig(
            os.path.join(output_dir, filename),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(facetgrid.fig)


def p_to_stars(p):
    if pd.isna(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_sig_bracket(ax, x1, x2, y, h, text, fontsize=11):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="black")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=fontsize)


def run_friedman_and_pairwise(
    sample_means,
    subtype_col="ec_subtype",
    sample_col="sample",
    brain_area_col="brain_area",
    metrics=tuple(DISTANCE_METRICS),
    subtype_order=tuple(SUBTYPE_ORDER),
):
    overall_results = []
    pairwise_results = []

    for metric in metrics:
        for area, sub in sample_means.groupby(brain_area_col, sort=False):
            sub = sub[sub[subtype_col].isin(subtype_order)].copy()
            wide = (
                sub.pivot_table(
                    index=sample_col,
                    columns=subtype_col,
                    values=metric,
                    aggfunc="first",
                    observed=True,
                )
                .reindex(columns=subtype_order)
                .dropna()
            )
            n_samples = wide.shape[0]

            if n_samples < 2:
                overall_results.append({
                    "brain_area": area,
                    "metric": metric,
                    "n_samples": n_samples,
                    "friedman_stat": np.nan,
                    "friedman_p": np.nan,
                })
                continue

            stat, p = friedmanchisquare(*[wide[col].values for col in subtype_order])
            overall_results.append({
                "brain_area": area,
                "metric": metric,
                "n_samples": n_samples,
                "friedman_stat": stat,
                "friedman_p": p,
            })

            for g1, g2 in itertools.combinations(subtype_order, 2):
                x = wide[g1].values
                y = wide[g2].values
                try:
                    w_stat, p_raw = wilcoxon(
                        x,
                        y,
                        zero_method="wilcox",
                        alternative="two-sided",
                    )
                except ValueError:
                    w_stat, p_raw = np.nan, 1.0

                pairwise_results.append({
                    "brain_area": area,
                    "metric": metric,
                    "group1": g1,
                    "group2": g2,
                    "n_samples": n_samples,
                    "wilcoxon_stat": w_stat,
                    "p_raw": p_raw,
                })

    overall_df = pd.DataFrame(overall_results)
    pairwise_df = pd.DataFrame(pairwise_results)

    overall_df["friedman_p_bh"] = np.nan
    for metric in metrics:
        mask = (overall_df["metric"] == metric) & overall_df["friedman_p"].notna()
        if mask.any():
            overall_df.loc[mask, "friedman_p_bh"] = multipletests(
                overall_df.loc[mask, "friedman_p"].values,
                method="fdr_bh",
            )[1]

    pairwise_df["p_adj"] = np.nan
    for metric in metrics:
        mask = (pairwise_df["metric"] == metric) & pairwise_df["p_raw"].notna()
        if mask.any():
            pairwise_df.loc[mask, "p_adj"] = multipletests(
                pairwise_df.loc[mask, "p_raw"].values,
                method="fdr_bh",
            )[1]

    return overall_df, pairwise_df


def plot_ec_distance_errorbars_with_stats(
    sample_distance_stats,
    overall_df,
    pairwise_df,
    output_dir,
    brain_area=None,
    subtypes=("aECs", "capECs", "vECs"),
    metrics=("dist_to_nearest_smc", "dist_to_nearest_pericyte"),
    sample_col="sample",
    subtype_col="ec_subtype",
    brain_area_col="brain_area",
    jitter=0.08,
    point_size=40,
    alpha=0.9,
    random_seed=0,
    show_only_significant=True,
    require_significant_omnibus=False,
):
    df = sample_distance_stats.loc[
        sample_distance_stats[brain_area_col] == brain_area
    ].copy()
    df = df[df[subtype_col].isin(subtypes)].copy()

    if df.empty:
        return

    samples = sorted(df[sample_col].dropna().unique())
    sample_colors = {sample: f"C{i % 10}" for i, sample in enumerate(samples)}
    rng = np.random.default_rng(random_seed)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    pair_positions = {
        ("aECs", "capECs"): (1, 2),
        ("aECs", "vECs"): (1, 3),
        ("capECs", "vECs"): (2, 3),
    }

    for ax, metric in zip(axes, metrics):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        plotted_values = []

        for i, subtype in enumerate(subtypes, start=1):
            subtype_df = df.loc[df[subtype_col] == subtype]

            for _, row in subtype_df.iterrows():
                x = i + rng.uniform(-jitter, jitter)
                color = sample_colors[row[sample_col]]

                y = row[mean_col]
                yerr = row[std_col]

                plotted_values.append(y)
                if pd.notna(yerr):
                    plotted_values.extend([y - yerr, y + yerr])

                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt="o",
                    color=color,
                    ecolor=color,
                    elinewidth=1.2,
                    capsize=3,
                    markersize=np.sqrt(point_size),
                    alpha=alpha,
                    zorder=3,
                )

        ax.set_xticks(range(1, len(subtypes) + 1))
        ax.set_xticklabels(subtypes)
        ax.set_title(TITLE_MAP.get(metric, metric))
        ax.set_ylabel("Distance (µm)")

        annotate_pairs = True
        if require_significant_omnibus:
            omnibus = overall_df[
                (overall_df["brain_area"] == brain_area)
                & (overall_df["metric"] == metric)
            ]
            if (
                omnibus.empty
                or pd.isna(omnibus["friedman_p_bh"].iloc[0])
                or omnibus["friedman_p_bh"].iloc[0] >= 0.05
            ):
                annotate_pairs = False

        if annotate_pairs:
            area_pairs = pairwise_df[
                (pairwise_df["brain_area"] == brain_area)
                & (pairwise_df["metric"] == metric)
            ].copy()

            plotted_values = [v for v in plotted_values if pd.notna(v)]
            if not area_pairs.empty and plotted_values:
                ymax = max(plotted_values)
                ymin = min(plotted_values)
                yrange = ymax - ymin if ymax > ymin else max(abs(ymax), 1.0)
                base_y = ymax + 0.08 * yrange
                step = 0.10 * yrange
                bracket_h = 0.03 * yrange
                level = 0

                for _, row in area_pairs.iterrows():
                    stars = p_to_stars(row["p_adj"])
                    if show_only_significant and stars == "n.s.":
                        continue

                    pair_key = (row["group1"], row["group2"])
                    if pair_key not in pair_positions:
                        continue

                    x1, x2 = pair_positions[pair_key]
                    y = base_y + level * step
                    add_sig_bracket(ax, x1, x2, y, bracket_h, stars)
                    level += 1

                if level > 0:
                    ax.set_ylim(top=base_y + level * step + 0.12 * yrange)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=sample_colors[sample],
            label=sample,
            markersize=6,
        )
        for sample in samples
    ]
    axes[-1].legend(handles=handles, title="Sample", loc="best")
    fig.suptitle(f"{brain_area}: EC sample-level mean ± SD distances by subtype", y=1.02)
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"{safe_name(brain_area)}_mean_sd_dist_errorbar_stats.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mixed-model/niche and mural-distance analysis from the notebook."
    )
    parser.add_argument("--anndata_file", required=True, help="Path to AnnData .h5ad/.h5ad.gz file")
    parser.add_argument("--brain_areas", required=True, help="Path to brain area annotation CSV")
    parser.add_argument("--out", required=True, help="Output folder")
    parser.add_argument("--counts", default="counts", help="Layer containing raw counts")
    parser.add_argument("--log_norm", default="librarysize_log1p_norm", help="Layer containing log1p-normalised counts")
    parser.add_argument("--cell_types", default="cell_type_incl_low_quality_revised", help="obs column containing cell types")
    parser.add_argument("--age_col", default="age_months", help="obs column containing age")
    parser.add_argument("--age", default="3", help="Age value to filter for niche and distance analysis. Use 'none' to disable.")
    parser.add_argument("--radii", default="20,30,40,50,60,70,80,90,100", help="Comma-separated radii in microns")
    parser.add_argument("--exclude_brain_areas", default="BS_STR,STR_CTX,CAsp,Meninges,DG-sg", help="Comma-separated brain areas to exclude. Use 'none' to keep all.")
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = args.out.rstrip(",")
    csv_dir = os.path.join(out_dir, "csvs")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    sc.settings.figdir = fig_dir

    anndata_file = args.anndata_file.rstrip(",")
    brain_areas_file = args.brain_areas.rstrip(",")
    radii = parse_radii(args.radii)
    age = None if str(args.age).lower() == "none" else float(args.age)
    exclude_brain_areas = parse_comma_list(args.exclude_brain_areas)

    print("Loading AnnData...")
    adata_integrated = sc.read_h5ad(anndata_file)

    print("Loading brain area annotations...")
    brain_areas = pd.read_csv(brain_areas_file)

    print("Preparing AnnData...")
    adata_integrated = prepare_anndata(
        adata_integrated,
        counts=args.counts,
        log_norm=args.log_norm,
        celltype=args.cell_types,
    )
    adata_integrated = annotate_brain_areas(adata_integrated, brain_areas)

    if exclude_brain_areas:
        adata_integrated = adata_integrated[
            ~adata_integrated.obs["brain_area"].isin(exclude_brain_areas)
        ].copy()

    print("Brain areas:")
    print(adata_integrated.obs["brain_area"].value_counts())

    print("\nBuilding EC niche table...")
    final_niches_df, final_niches_df_long = build_niches_long(
        adata_integrated,
        radii=radii,
        celltype_col=args.cell_types,
        age_col=args.age_col,
        age=age,
    )

    final_niches_df.to_csv(os.path.join(csv_dir, "fixed_niches_wide_glmm.csv"), index=False)
    final_niches_df_long.to_csv(os.path.join(csv_dir, "fixed_niches_long_glmm.csv"), index=False)
    print("Final long niche dataframe shape:", final_niches_df_long.shape)

    print("\nPreparing EC subtype labels for distance analysis...")
    adata_distance = adata_integrated.copy()
    if age is not None:
        age_numeric = pd.to_numeric(adata_distance.obs[args.age_col], errors="coerce")
        age_mask = (age_numeric == age).to_numpy()
        adata_distance = adata_distance[age_mask, :].copy()

    ec_mask = (adata_distance.obs[args.cell_types] == "ECs").to_numpy()
    ec_adata = adata_distance[ec_mask, :].copy()
    ec_adata = annotate_ec_subtypes(ec_adata)

    adata_distance.obs["ec_subtype"] = None
    common = adata_distance.obs_names.intersection(ec_adata.obs_names)
    adata_distance.obs.loc[common, "ec_subtype"] = ec_adata.obs.loc[common, "ec_subtype"]

    ec_df = compute_ec_mural_distances(adata_distance, args.cell_types)
    ec_df.to_csv(os.path.join(csv_dir, "ec_mural_distances.csv"), index=True)

    print("\nComputing sample-level mean and SD distances...")
    sample_distance_stats = compute_sample_distance_stats(ec_df)
    sample_distance_stats.to_csv(
        os.path.join(csv_dir, "mean_sd_distance_murals.csv"),
        index=False,
    )

    sample_means = distance_stats_to_means(sample_distance_stats)
    sample_means.to_csv(
        os.path.join(csv_dir, "mean_distance_murals.csv"),
        index=False,
    )

    for area in ec_df["brain_area"].dropna().unique():
        plot_ec_distance_histograms(ec_df, fig_dir, brain_area=area)

    summary = summarize_cell_counts(ec_df, adata_distance, args.cell_types)
    summary.to_csv(os.path.join(csv_dir, "summary_counts_celltypes.csv"), index=True)

    print("\nPlotting mural cell counts by EC subtype and radius...")
    plot_mural_counts_by_radius(final_niches_df_long, fig_dir)

    print("\nRunning distance statistics...")
    overall_df, pairwise_df = run_friedman_and_pairwise(sample_means)
    overall_df.to_csv(os.path.join(csv_dir, "friedman_overall_distance_stats.csv"), index=False)
    pairwise_df.to_csv(os.path.join(csv_dir, "wilcoxon_pairwise_distance_stats.csv"), index=False)

    for area in ec_df["brain_area"].dropna().unique():
        plot_ec_distance_errorbars_with_stats(
            sample_distance_stats=sample_distance_stats,
            overall_df=overall_df,
            pairwise_df=pairwise_df,
            output_dir=fig_dir,
            brain_area=area,
            show_only_significant=True,
            require_significant_omnibus=False,
        )

    group_summary = (
        sample_means
        .groupby(["brain_area", "ec_subtype"], observed=True)[DISTANCE_METRICS]
        .agg(["mean", "std", "count"])
        .round(2)
    )
    group_summary.to_csv(
        os.path.join(csv_dir, "summary_stats_nearest_mural.csv"),
        index=True,
    )

    print(f"\nDone. CSVs saved to: {csv_dir}")
    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()