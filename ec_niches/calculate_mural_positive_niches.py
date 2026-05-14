#!/usr/bin/env python3

"""
Calculates and plots the percentage of EC niches that are positive for SMCs and pericytes, as well as total number of EC niches
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xarray_schema")
warnings.filterwarnings("ignore", category=FutureWarning, module="squidpy")
import dask
dask.config.set({"dataframe.query-planning": True})

import anndata as ad
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

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


def parse_input_lists(value, dtype=str):
    if value is None or str(value).strip().lower() in {"", "none"}:
        return []
    return [dtype(x.strip()) for x in str(value).split(",") if x.strip()]


def edit_file_name_for_saving(value):
    return str(value).replace("/", "_").replace(" ", "_").replace("-", "_")


def prepare_anndata(adata, counts, log1norm, celltype):
    adata.raw = ad.AnnData(
        X=adata.layers[counts].copy(),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    adata.X = adata.layers[log1norm].copy()
    adata = adata[adata.obs[celltype] != "Undefined"].copy()
    return adata


def annotate_brain_areas(adata, brain_areas_csv):
    brain_areas_csv = brain_areas_csv.rename(columns={"Unnamed: 0": "cell_id"})
    brain_areas_csv = brain_areas_csv.set_index("cell_id")
    brain_areas_csv = brain_areas_csv.reindex(index=adata.obs.index)

    adata.obs["brain_area"] = brain_areas_csv["label"]
    adata = adata[adata.obs["brain_area"].notna()].copy()
    adata.obs["brain_area"] = adata.obs["brain_area"].str.replace("/", "_", regex=False)
    return adata


def compute_nneighbours_per_celltype(adata, key_added, celltypes):
    connectivities = adata.obsp[f"{key_added}_connectivities"].tocsr()
    print(connectivities)

    different_cell_types = adata.obs[celltypes].astype("category")
    categories = list(different_cell_types.cat.categories)

    composition_matrix = np.zeros((adata.n_obs, len(categories)))

    for i, ct in enumerate(categories):
        mask_filter = (different_cell_types == ct).values.astype(float)
        composition_matrix[:, i] = connectivities.dot(mask_filter)

    cell_niches = []
    for i, ct in enumerate(categories):
        adata.obs[f"nhood_{ct}"] = composition_matrix[:, i]
        nhood = adata.obsm[f"nhood_{ct}"] = composition_matrix[:, i]
        cell_niches.append(nhood)

    niches_transp = np.transpose(np.array(cell_niches))
    neighbors_df = pd.DataFrame(niches_transp)

    adata.obsm["celltype_nhoods"] = np.array(neighbors_df)
    adata.obsm["cell_based_niche_categories"] = np.array(different_cell_types.values)

    return adata


def annotate_ec_subtypes(ec_adata):
    for subtype, markers in EC_SUBTYPES.items():
        sc.tl.score_genes(ec_adata, gene_list=markers, score_name=subtype)

    score_cols = list(EC_SUBTYPES.keys())
    max_scores = ec_adata.obs[score_cols].max(axis=1)
    best_subtypes = ec_adata.obs[score_cols].idxmax(axis=1)

    ec_adata.obs["ec_subtype"] = best_subtypes
    ec_adata.obs.loc[max_scores <= 0, "ec_subtype"] = "unassigned"

    ec_adata = ec_adata[ec_adata.obs.ec_subtype != "unassigned"].copy()
    return ec_adata


def calculate_ec_niches(
    adata,
    ec_adata,
    celltype_col,
    celltype_nhoods,
    ec_subtype,
    age_col,
    *,
    age=None,
    sample_col="sample",
    brain_area_col="brain_area"):

    different_cell_types = adata.obs[celltype_col].astype("category")
    celltypes = list(different_cell_types.astype(str).unique())

    x = ec_adata.obsm[celltype_nhoods]
    assert x.shape[1] == len(celltypes), (
        f"{celltype_nhoods} has {x.shape[1]} columns, "
        f"but celltypes has {len(celltypes)} entries"
    )

    ec_niches = pd.DataFrame(x, columns=celltypes, index=ec_adata.obs.index)
    ec_niches["EC_subtypes"] = ec_adata.obs[ec_subtype]
    ec_niches["age_months"] = pd.to_numeric(ec_adata.obs[age_col], errors="coerce")
    ec_niches["sample"] = ec_adata.obs[sample_col] if sample_col in ec_adata.obs.columns else "unknown"
    ec_niches["EC_cell_ID"] = ec_adata.obs_names

    if brain_area_col in ec_adata.obs.columns:
        ec_niches["brain_area"] = ec_adata.obs[brain_area_col]
    else:
        ec_niches["brain_area"] = "unknown"

    ec_niches = ec_niches.drop(columns=["Undefined"], errors="ignore")

    celltype_cols = [c for c in celltypes if c in ec_niches.columns and c != "Undefined"]
    ec_niches["row_sum"] = ec_niches[celltype_cols].sum(axis=1)

    print("Rows before row_sum filter:", ec_niches.shape[0])
    print("Row_sum == 0:", (ec_niches["row_sum"] == 0).sum())

    ec_niches = ec_niches.loc[ec_niches["row_sum"] != 0].copy()

    if age is not None:
        print("Unique ages:", ec_niches["age_months"].unique())
        print("Requested age:", age)
        ec_niches = ec_niches[ec_niches["age_months"] == age].copy()

    print("Final ec_niches shape:", ec_niches.shape)
    return ec_niches


def build_niches_df_long(adata, radii, celltype_col, age_col, age):
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
        print(f"Spatial neighbourhood graph computed for radius {radius}")

        adata = compute_nneighbours_per_celltype(adata, radius_key, celltype_col)

        ec_mask = (adata.obs[celltype_col] == "ECs").to_numpy()
        ec_adata = adata[ec_mask, :].copy()

        ec_adata = annotate_ec_subtypes(ec_adata)
        print(f"EC niches annotated for radius {radius}")

        for area in areas:
            temp_adata = ec_adata[ec_adata.obs["brain_area"] == area].copy()
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

    final_niches_df = pd.concat(all_niches, axis=0, ignore_index=True)
    print("Final dataframe shape:", final_niches_df.shape)

    cell_types = list(adata.obs[celltype_col].unique())

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


def summarize_mural_cell_abundance(
    df_in,
    target_label,
    group_cols=("EC_subtypes", "brain_area", "radius"),
    label_prefix="target"):
    target_df = df_in[df_in["cell_type"] == target_label].copy()

    zero_df = target_df[target_df["cell_count"] == 0]
    ge1_df = target_df[target_df["cell_count"] >= 1]

    zero_counts = (
        zero_df
        .groupby(list(group_cols))
        .size()
        .reset_index(name=f"{label_prefix}_0")
    )

    ge1_counts = (
        ge1_df
        .groupby(list(group_cols))
        .size()
        .reset_index(name=f"{label_prefix}_ge1")
    )

    summary = (
        pd.merge(
            ge1_counts,
            zero_counts,
            on=list(group_cols),
            how="outer",
        )
        .fillna(0)
    )

    summary[f"{label_prefix}_0"] = summary[f"{label_prefix}_0"].astype(int)
    summary[f"{label_prefix}_ge1"] = summary[f"{label_prefix}_ge1"].astype(int)

    summary[f"{label_prefix}_total"] = (
        summary[f"{label_prefix}_0"] + summary[f"{label_prefix}_ge1"]
    )

    summary[f"{label_prefix}_pct_0"] = np.where(
        summary[f"{label_prefix}_total"] > 0,
        100 * summary[f"{label_prefix}_0"] / summary[f"{label_prefix}_total"],
        0,
    )

    summary[f"{label_prefix}_pct_ge1"] = np.where(
        summary[f"{label_prefix}_total"] > 0,
        100 * summary[f"{label_prefix}_ge1"] / summary[f"{label_prefix}_total"],
        0,
    )

    return summary.sort_values(list(group_cols)).reset_index(drop=True)


def plot_niches(summary_df, label_prefix, zero_label, ge1_label, out_file):
    if summary_df.empty:
        print(f"No data available to plot for {label_prefix}.")
        return

    ec_subtypes = list(summary_df["EC_subtypes"].dropna().sort_values().unique())
    brain_areas = list(summary_df["brain_area"].dropna().sort_values().unique())

    if len(ec_subtypes) == 0 or len(brain_areas) == 0:
        print(f"No valid EC_subtypes or brain_area values to plot for {label_prefix}.")
        return

    nrows = len(ec_subtypes)
    ncols = len(brain_areas)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.2 * ncols, 2.8 * nrows),
        sharex=False,
        sharey=True,
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    dot_handles = []

    for i, ec in enumerate(ec_subtypes):
        for j, area in enumerate(brain_areas):
            ax = axes[i, j]

            sub = summary_df[
                (summary_df["EC_subtypes"] == ec)
                & (summary_df["brain_area"] == area)
            ].copy().sort_values("radius")

            if sub.empty:
                ax.set_visible(False)
                continue

            x = np.arange(len(sub))
            xlabels = sub["radius"].astype(str).tolist()

            y0 = sub[f"{label_prefix}_pct_0"].values
            y1 = sub[f"{label_prefix}_pct_ge1"].values
            totals = sub[f"{label_prefix}_total"].values

            ax.bar(x, y0, label=zero_label, width=0.9, color="#ADD8E6")
            ax.bar(x, y1, bottom=y0, label=ge1_label, width=0.9, color="coral")

            ax.set_ylim(0, 100)
            ax.set_ylabel(f"{ec}\n\nPercentage of niches" if j == 0 else "")

            ax2 = ax.twinx()
            dot = ax2.scatter(
                x,
                totals,
                color="black",
                s=18,
                zorder=5,
                label="total niches",
            )
            dot_handles.append(dot)

            ax2.set_ylim(0, max(totals) * 1.15 if max(totals) > 0 else 1)

            if j == ncols - 1:
                ax2.set_ylabel("Total niches")
            else:
                ax2.set_ylabel("")
                ax2.set_yticklabels([])

            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, rotation=90)
            ax.set_xlabel("Radius")

            if i == 0:
                ax.set_title(f"{label_prefix.upper()} presence in EC niches ({area})")

            ax.spines["top"].set_visible(False)
            ax2.spines["top"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if dot_handles:
        handles.append(dot_handles[0])
        labels.append("total niches")

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close(fig)


def run_mural_niche_analysis(
    final_niches_df_long,
    mural_cell_types,
    plot_brain_areas,
    csv_dir,
    fig_dir):
    for mural_cell_type in mural_cell_types:
        target_df = final_niches_df_long[
            final_niches_df_long["cell_type"].isin([mural_cell_type])
        ].copy()

        if plot_brain_areas:
            target_df = target_df[target_df["brain_area"].isin(plot_brain_areas)].copy()

        if target_df.empty:
            print(f"No data found for mural cell type {mural_cell_type} in requested brain area(s).")
            continue

        label_prefix = edit_file_name_for_saving(mural_cell_type).lower()
        if mural_cell_type.lower() == "pericytes":
            label_prefix = "pericyte"
        elif mural_cell_type.lower() == "smcs":
            label_prefix = "smc"

        summary = summarize_mural_cell_abundance(
            target_df,
            target_label=mural_cell_type,
            label_prefix=label_prefix,
        )

        summary_file = os.path.join(csv_dir, f"{label_prefix}_niche_summary.csv")
        figure_file = os.path.join(fig_dir, f"{label_prefix}_percentage_with_totals.pdf")

        summary.to_csv(summary_file, index=False)

        plot_niches(
            summary,
            label_prefix=label_prefix,
            zero_label="zero",
            ge1_label="non-zero",
            out_file=figure_file,
        )

        print(f"\n{mural_cell_type} summary:")
        print(summary.head())
        print(summary.shape)
        print(f"Saved summary: {summary_file}")
        print(f"Saved figure: {figure_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute percentage of EC niches positive for selected mural cell types."
    )

    parser.add_argument("--anndata_file", required=True, help="Path to AnnData .h5ad/.h5ad.gz file")
    parser.add_argument("--brain_areas", required=True, help="Path to brain area annotation CSV")
    parser.add_argument("--out", required=True, help="Output folder")

    parser.add_argument("--counts", default="counts", help="Layer containing raw counts")
    parser.add_argument(
        "--log_norm",
        default="librarysize_log1p_norm",
        help="Layer containing log1p-normalized counts",
    )
    parser.add_argument(
        "--cell_types",
        default="cell_type_incl_low_quality_revised",
        help="obs column containing cell types",
    )
    parser.add_argument("--age_col", default="age_months", help="obs column containing age")
    parser.add_argument("--age", default="3", help="Age value to filter. Use 'none' to disable.")

    parser.add_argument(
        "--radii",
        default="20,30,40,50,60,70,80,90,100,200,250,300",
        help="Comma-separated radii in microns. These are both computed and plotted.",
    )
    parser.add_argument(
        "--plot_brain_areas",
        "--plot_areas",
        default="HIP",
        help="Comma-separated brain areas to plot. Use 'none' for all retained areas.",
    )
    parser.add_argument(
        "--mural_cell_types",
        default="SMCs,Pericytes",
        help="Comma-separated mural cell types to summarize and plot.",
    )
    parser.add_argument(
        "--exclude_brain_areas",
        default="BS_STR,STR_CTX,CAsp,Meninges,DG-sg",
        help="Comma-separated brain areas to exclude before analysis. Use 'none' to keep all.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = args.out.rstrip(",")
    csv_dir = os.path.join(out_dir, "csvs")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    sc.settings.figdir = fig_dir

    radii = parse_input_lists(args.radii, dtype=int)
    plot_brain_areas = parse_input_lists(args.plot_brain_areas, dtype=str)
    mural_cell_types = parse_input_lists(args.mural_cell_types, dtype=str)
    exclude_brain_areas = parse_input_lists(args.exclude_brain_areas, dtype=str)
    age = None if str(args.age).strip().lower() == "none" else float(args.age)

    print("Loading AnnData...")
    adata_integrated = sc.read_h5ad(args.anndata_file.rstrip(","))

    print("Loading brain area annotations...")
    brain_areas = pd.read_csv(args.brain_areas.rstrip(","))

    print("Preparing AnnData...")
    adata_integrated = prepare_anndata(
        adata_integrated,
        counts=args.counts,
        log1norm=args.log_norm,
        celltype=args.cell_types,
    )
    adata_integrated = annotate_brain_areas(adata_integrated, brain_areas)

    if exclude_brain_areas:
        adata_integrated = adata_integrated[
            ~adata_integrated.obs["brain_area"].isin(exclude_brain_areas)
        ].copy()

    print("Brain areas retained:")
    print(adata_integrated.obs["brain_area"].value_counts())

    print("\nBuilding EC niche table...")
    final_niches_df, final_niches_df_long = build_niches_df_long(
        adata_integrated,
        radii=radii,
        celltype_col=args.cell_types,
        age_col=args.age_col,
        age=age,
    )

    wide_file = os.path.join(csv_dir, "niches_wide_glmm.csv")
    long_file = os.path.join(csv_dir, "niches_long_glmm.csv")

    final_niches_df.to_csv(wide_file, index=False)
    final_niches_df_long.to_csv(long_file, index=False)

    print("Final wide niche dataframe shape:", final_niches_df.shape)
    print("Final long niche dataframe shape:", final_niches_df_long.shape)
    print(f"Saved wide table: {wide_file}")
    print(f"Saved long table: {long_file}")

    run_mural_niche_analysis(
        final_niches_df_long=final_niches_df_long,
        mural_cell_types=mural_cell_types,
        plot_brain_areas=plot_brain_areas,
        csv_dir=csv_dir,
        fig_dir=fig_dir,
    )

    print(f"\nDone. CSVs saved to: {csv_dir}")
    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
