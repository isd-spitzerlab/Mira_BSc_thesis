#!/usr/bin/env python3

"""

"""

import dask
dask.config.set({"dataframe.query-planning": True})

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import gc
import os
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

DEFAULT_OBS_TO_KEEP = [
    "cell_type_incl_low_quality_revised",
    "sample",
    "age_months",
]

DEFAULT_LAYERS_TO_REMOVE = [
    "volume_log1p_norm",
    "volume_norm",
    "zscore",
]

DEFAULT_AREAS_TO_KEEP = [
    "BS",
    "CTX",
    "HIP",
    "STR",
    "fiber_tracts",
    "VS",
]

DEFAULT_RADII = [20, 50, 80, 100, 200, 250, 300]


def parse_csv_list(value):
    if value is None or value == "":
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_list(value):
    if value is None or value == "":
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def minimize_anndata(adata, obs_to_keep, coord_column, layers_to_remove):
    missing_obs = [c for c in obs_to_keep if c not in adata.obs.columns]
    if missing_obs:
        raise KeyError(f"Missing obs columns in AnnData: {missing_obs}")

    if coord_column not in adata.obsm:
        raise KeyError(f"Missing coordinate column in adata.obsm: {coord_column}")

    adata.obs = adata.obs[obs_to_keep].copy()
    coords = adata.obsm[coord_column].copy()

    adata.obsm.clear()
    adata.obsm[coord_column] = coords

    adata.uns.clear()
    adata.obsp.clear()
    adata.varm.clear()

    for layer in layers_to_remove:
        adata.layers.pop(layer, None)

    gc.collect()
    return adata


def prepare_anndata(adata, counts, log_norm, celltype):
    if counts not in adata.layers:
        raise KeyError(f"Counts layer not found: {counts}")

    if log_norm not in adata.layers:
        raise KeyError(f"Log-normalized layer not found: {log_norm}")

    if celltype not in adata.obs.columns:
        raise KeyError(f"Cell type column not found in adata.obs: {celltype}")

    adata.raw = ad.AnnData(
        X=adata.layers[counts].copy(),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )

    adata.X = adata.layers[log_norm].copy()

    mask = (adata.obs[celltype] != "Undefined").to_numpy()
    adata = adata[mask, :].copy()

    return adata


def annotate_brain_areas(adata, brain_areas_df):
    if "Unnamed: 0" in brain_areas_df.columns:
        brain_areas_df = brain_areas_df.rename(columns={"Unnamed: 0": "cell_id"})

    if "cell_id" not in brain_areas_df.columns:
        raise KeyError(
            "Brain area CSV must contain either a 'cell_id' column or an 'Unnamed: 0' column."
        )

    if "label" not in brain_areas_df.columns:
        raise KeyError("Brain area CSV must contain a 'label' column.")

    brain_areas_df = brain_areas_df.set_index("cell_id")
    brain_areas_df = brain_areas_df.reindex(index=adata.obs.index)

    adata.obs["brain_area"] = brain_areas_df["label"]

    adata.obs["brain_area"] = (
        adata.obs["brain_area"]
        .astype("string")
        .str.replace("/", "_", regex=False)
    )

    mask = adata.obs["brain_area"].notna().to_numpy()
    adata = adata[mask, :].copy()

    return adata


def filter_brain_areas(adata, brain_area_column, areas_to_keep):
    if brain_area_column not in adata.obs.columns:
        raise KeyError(f"Brain area column not found in adata.obs: {brain_area_column}")

    mask = adata.obs[brain_area_column].isin(areas_to_keep).to_numpy()
    return adata[mask, :].copy()


def compute_nneighbours_per_celltype(adata, key_added, celltype_col):
    connectivities_key = f"{key_added}_connectivities"

    if connectivities_key not in adata.obsp:
        raise KeyError(f"Missing neighborhood graph in adata.obsp: {connectivities_key}")

    connectivities = adata.obsp[connectivities_key].tocsr()
    different_cell_types = adata.obs[celltype_col].astype("category")

    categories = list(different_cell_types.cat.categories)
    nr_cells = adata.n_obs

    composition_matrix = np.zeros((nr_cells, len(categories)), dtype=np.float32)

    for i, celltype in enumerate(categories):
        mask_filter = (different_cell_types == celltype).to_numpy().astype(float)
        composition_matrix[:, i] = connectivities.dot(mask_filter)

    for i, celltype in enumerate(categories):
        safe_celltype = str(celltype).replace("/", "_")
        adata.obs[f"nhood_{safe_celltype}"] = composition_matrix[:, i]

    adata.obsm["celltype_nhoods"] = composition_matrix
    adata.uns["celltype_nhood_categories"] = categories

    return adata


def annotate_ec_subtypes(ec_adata):
    selected_ec_subtypes = {
        "aECs": [
            "Bmx", "Efnb2", "Vegfc", "Mgp", "Cytl1", "Sema3g",
            "Gkn3", "Fbln2", "Hey1", "Egfl8", "Jag1", "Igf2",
            "Notch3", "Mgp", "Clu",
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

    for subtype, markers in selected_ec_subtypes.items():
        markers_present = [gene for gene in markers if gene in ec_adata.var_names]

        if markers_present:
            sc.tl.score_genes(
                ec_adata,
                gene_list=markers_present,
                score_name=subtype,
                use_raw=False,
            )
        else:
            ec_adata.obs[subtype] = np.nan

    selected_subtype_scores = list(selected_ec_subtypes.keys())

    max_scores = ec_adata.obs[selected_subtype_scores].max(axis=1)
    best_subtypes = ec_adata.obs[selected_subtype_scores].idxmax(axis=1)

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
    ec_subtype,
    age_col,
    *,
    age=None,
    sample_col="sample",
    brain_area_col="brain_area",
):
    if "celltype_nhood_categories" in adata.uns:
        celltypes = list(adata.uns["celltype_nhood_categories"])
    else:
        celltypes = list(adata.obs[celltype_col].astype("category").cat.categories)

    x = ec_adata.obsm[celltype_nhoods]

    if x.shape[1] != len(celltypes):
        raise ValueError(
            f"{celltype_nhoods} has {x.shape[1]} columns, "
            f"but there are {len(celltypes)} cell type categories."
        )

    ec_niches = pd.DataFrame(x, columns=celltypes, index=ec_adata.obs.index)

    ec_niches["EC_subtypes"] = ec_adata.obs[ec_subtype].to_numpy()
    ec_niches["age_months"] = pd.to_numeric(ec_adata.obs[age_col], errors="coerce").to_numpy()
    ec_niches["sample"] = ec_adata.obs[sample_col].to_numpy()
    ec_niches["EC_cell_ID"] = ec_adata.obs_names

    if brain_area_col in ec_adata.obs.columns:
        ec_niches["brain_area"] = ec_adata.obs[brain_area_col].to_numpy()

    ec_niches = ec_niches.drop(columns=["Undefined"], errors="ignore")

    celltype_cols = [c for c in celltypes if c in ec_niches.columns and c != "Undefined"]
    ec_niches["row_sum"] = ec_niches[celltype_cols].sum(axis=1)

    print("Rows before row_sum filter:", ec_niches.shape[0])
    print("Row_sum == 0:", int((ec_niches["row_sum"] == 0).sum()))

    ec_niches = ec_niches.loc[ec_niches["row_sum"] != 0].copy()

    if age is not None:
        ec_niches = ec_niches[ec_niches["age_months"] == age].copy()

    ec_niches = ec_niches.drop(columns=["row_sum"], errors="ignore")

    print("Final ec_niches shape:", ec_niches.shape)
    return ec_niches


def build_niches(
    adata,
    areas_to_keep,
    radii,
    coord_column,
    sample_col,
    celltype_col,
    age_col,
):
    all_niches = []

    for radius in radii:
        print(f"\n=== Processing radius {radius} µm ===")

        radius_key = f"spatial_multi_{radius}"

        sq.gr.spatial_neighbors(
            adata,
            spatial_key=coord_column,
            coord_type="generic",
            radius=radius,
            delaunay=False,
            key_added=radius_key,
            library_key=sample_col,
        )

        print(f"Spatial neighbourhood graph computed for radius {radius}")

        adata = compute_nneighbours_per_celltype(
            adata,
            radius_key,
            celltype_col,
        )

        ec_mask = (adata.obs[celltype_col] == "ECs").to_numpy()
        ec_adata = adata[ec_mask, :].copy()

        if ec_adata.n_obs == 0:
            print(f"Skipping radius {radius}: no EC cells")
            continue

        ec_adata = annotate_ec_subtypes(ec_adata)
        print(f"EC niches annotated for radius {radius}")

        for area in areas_to_keep:
            area_mask = (ec_adata.obs["brain_area"] == area).to_numpy()

            if area_mask.sum() == 0:
                print(f"Skipping {area}: no EC cells")
                continue

            temp_adata = ec_adata[area_mask, :].copy()

            ec_niches = calculate_ec_niches(
                adata,
                temp_adata,
                celltype_col,
                "celltype_nhoods",
                "ec_subtype",
                age_col,
                sample_col=sample_col,
                brain_area_col="brain_area",
            )

            if ec_niches.empty:
                print(f"Skipping {area} at radius {radius}: no non-empty niches")
                continue

            ec_niches["radius"] = radius
            all_niches.append(ec_niches)

        gc.collect()

    if not all_niches:
        raise RuntimeError("No niche tables were generated. Check EC labels, brain areas, and radii.")

    final_niches_df = pd.concat(all_niches, axis=0, ignore_index=True)
    return final_niches_df


def compute_niches_df_long(final_niches_df, adata, celltype_col):
    cell_types = list(adata.obs[celltype_col].astype("category").cat.categories)

    rename_map = {
        cell_type: str(cell_type).replace("-", "_")
        for cell_type in cell_types
        if cell_type in final_niches_df.columns
    }

    final_niches_df = final_niches_df.rename(columns=rename_map)

    value_vars = [
        str(cell_type).replace("-", "_")
        for cell_type in cell_types
        if str(cell_type).replace("-", "_") in final_niches_df.columns
    ]

    id_vars = [
        "EC_subtypes",
        "sample",
        "EC_cell_ID",
        "brain_area",
        "radius",
        "age_months",
    ]

    missing_id_vars = [col for col in id_vars if col not in final_niches_df.columns]
    if missing_id_vars:
        raise KeyError(f"Missing required columns in niche table: {missing_id_vars}")

    final_niches_df_long = final_niches_df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="cell_type",
        value_name="cell_count",
    )

    final_niches_df_long["EC_cell_ID"] = (
        final_niches_df_long["EC_cell_ID"].astype(str).str.replace("-", "_", regex=False)
    )

    return final_niches_df, final_niches_df_long

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build all-age EC niche tables across spatial radii for mixed-model analysis."
    )

    parser.add_argument("--anndata_file", required=True, help="Path to AnnData .h5ad/.h5ad.gz file")
    parser.add_argument("--brain_areas", required=True, help="Path to brain area annotation CSV")
    parser.add_argument("--out", required=True, help="Output folder")
    parser.add_argument("--counts", default="counts", help="Layer containing raw counts")
    parser.add_argument("--log_norm",default="librarysize_log1p_norm",help="Layer containing log1p-normalised counts")
    parser.add_argument("--cell_types",default="cell_type_incl_low_quality_revised",help="obs column containing cell types")
    parser.add_argument("--sample_col", default="sample", help="obs column containing sample IDs")
    parser.add_argument("--age_col", default="age_months", help="obs column containing ages")
    parser.add_argument("--coord_column", default="spatial_microns", help="obsm key containing spatial coordinates")
    parser.add_argument("--areas",default=",".join(DEFAULT_AREAS_TO_KEEP),help="Comma-separated brain areas to keep")
    parser.add_argument("--radii",default=",".join(str(x) for x in DEFAULT_RADII),help="Comma-separated radii in microns")
    parser.add_argument("--exclude_sample",default="aging_s1_r0",help="Sample to exclude. Use 'none' to keep all samples.")
    parser.add_argument("--layers_to_remove",default=",".join(DEFAULT_LAYERS_TO_REMOVE),help="Comma-separated layers to remove during minimization")
    return parser.parse_args()

def main():
    args = parse_args()
    out_dir = args.out.strip().rstrip(",")
    csv_dir = os.path.join(out_dir, "csvs")
    os.makedirs(csv_dir, exist_ok=True)

    areas_to_keep = parse_csv_list(args.areas)
    radii = parse_int_list(args.radii)
    layers_to_remove = parse_csv_list(args.layers_to_remove)

    obs_to_keep = list(dict.fromkeys([
        args.cell_types,
        args.sample_col,
        args.age_col,
    ]))

    print("Loading AnnData...")
    adata = sc.read_h5ad(args.anndata_file.strip().rstrip(","))

    print("Loading brain area annotations...")
    brain_areas = pd.read_csv(args.brain_areas.strip().rstrip(","))

    print("Preparing AnnData...")
    adata = minimize_anndata(
        adata,
        obs_to_keep=obs_to_keep,
        coord_column=args.coord_column,
        layers_to_remove=layers_to_remove,
    )

    adata = prepare_anndata(
        adata,
        counts=args.counts,
        log_norm=args.log_norm,
        celltype=args.cell_types,
    )

    adata = annotate_brain_areas(adata, brain_areas)
    adata = filter_brain_areas(adata, "brain_area", areas_to_keep)

    if args.exclude_sample.lower() != "none":
        if args.sample_col not in adata.obs.columns:
            raise KeyError(f"Sample column not found in adata.obs: {args.sample_col}")

        mask = (adata.obs[args.sample_col] != args.exclude_sample).to_numpy()
        adata = adata[mask, :].copy()

    adata.obs = adata.obs.rename(columns={args.cell_types: "cell_type"})
    celltype_col = "cell_type"

    celltype_counts = (
        adata.obs.groupby([args.age_col, args.sample_col], observed=True)
        .size()
        .reset_index(name="n_cells")
    )
    celltype_counts.to_csv(os.path.join(csv_dir, "all_ages_cells_per_age_sample.csv"), index=False)

    print("Building EC niche table...")
    final_niches_df = build_niches(
        adata=adata,
        areas_to_keep=areas_to_keep,
        radii=radii,
        coord_column=args.coord_column,
        sample_col=args.sample_col,
        celltype_col=celltype_col,
        age_col=args.age_col,
    )

    print("Final wide dataframe shape:", final_niches_df.shape)

    final_niches_df, final_niches_df_long = compute_niches_df_long(
        final_niches_df,
        adata,
        celltype_col=celltype_col,
    )

    wide_path = os.path.join(csv_dir, "all_ages_niches_wide.csv")
    long_path = os.path.join(csv_dir, "all_ages_niches_long_glmm.csv")

    final_niches_df.to_csv(wide_path, index=False)
    final_niches_df_long.to_csv(long_path, index=False)

    print("Done.")
    print(f"Wide table saved to: {wide_path}")
    print(f"Long GLMM table saved to: {long_path}")


if __name__ == "__main__":
    main()
