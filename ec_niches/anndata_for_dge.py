#!/usr/bin/env python

import argparse
import os
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree


def prepare_anndata(
    adata,
    counts_layer="counts",
    log1p_layer="librarysize_log1p_norm",
    celltype_col="cell_type_incl_low_quality_revised",
):
    if counts_layer in adata.layers:
        adata.raw = ad.AnnData(
            X=adata.layers[counts_layer].copy(),
            obs=adata.obs.copy(),
            var=adata.var.copy(),
        )

    if log1p_layer in adata.layers:
        adata.X = adata.layers[log1p_layer].copy()

    adata = adata[adata.obs[celltype_col] != "Undefined"].copy()
    return adata


def annotate_brain_areas(adata, brain_areas):
    brain_areas = pd.read_csv(brain_areas)

    brain_areas = brain_areas.rename(columns={"Unnamed: 0": "cell_id"})
    brain_areas = brain_areas.set_index("cell_id")
    brain_areas = brain_areas.reindex(index=adata.obs.index)

    adata.obs["brain_area"] = brain_areas["label"]
    adata = adata[adata.obs["brain_area"].notna()].copy()
    adata.obs["brain_area"] = adata.obs["brain_area"].str.replace("/", "_", regex=False)

    return adata


def reduce_anndata_size(
    adata,
    spatial_key="spatial_microns",
    obs_to_keep=None,
):
    if obs_to_keep is None:
        obs_to_keep = [
            "cell_type_incl_low_quality_revised",
            "sample",
            "age_months",
            "brain_area",
        ]

    obs_existing = [col for col in obs_to_keep if col in adata.obs.columns]
    adata.obs = adata.obs[obs_existing].copy()

    if spatial_key not in adata.obsm:
        raise KeyError(f"Expected coordinates in adata.obsm[{spatial_key!r}].")

    coords = adata.obsm[spatial_key].copy()
    adata.obsm.clear()
    adata.obsm[spatial_key] = coords

    adata.uns.clear()
    adata.obsp.clear()
    adata.varm.clear()

    for layer in ["volume_log1p_norm", "volume_norm", "zscore"]:
        adata.layers.pop(layer, None)

    return adata


def annotate_smc(
    adata,
    celltype_col="cell_type_incl_low_quality_revised",
):
    smc_markers = {
        "arterial_arteriolar_SMCs": [
            "Acta2", "Tagln", "Myh11", "Pln", "Cnn1", "Adamts1",
            "Arl4d", "Atf3", "Cd93", "H2afj", "Nanos1",
        ],
        "venous_SMCs": [
            "Abca1", "AI593442", "Car4", "Col6a2",
        ],
    }

    smc_adata = adata[adata.obs[celltype_col] == "SMCs"].copy()

    for subtype, markers in smc_markers.items():
        present_markers = [gene for gene in markers if gene in smc_adata.var_names]

        if present_markers:
            sc.tl.score_genes(
                smc_adata,
                gene_list=present_markers,
                score_name=subtype,
            )
        else:
            smc_adata.obs[subtype] = 0.0

    selected_subtype_scores = list(smc_markers.keys())
    max_scores = smc_adata.obs[selected_subtype_scores].max(axis=1)
    best_subtypes = smc_adata.obs[selected_subtype_scores].idxmax(axis=1)

    smc_adata.obs["smc_subtype"] = best_subtypes
    smc_adata.obs.loc[max_scores <= 0.0, "smc_subtype"] = "other_SMC"

    adata.obs["smc_subtype"] = "non_SMC"
    adata.obs.loc[smc_adata.obs_names, "smc_subtype"] = smc_adata.obs["smc_subtype"]

    return adata


def annotate_ec_subtypes_distance(
    adata,
    celltype_col="cell_type_incl_low_quality_revised",
    spatial_key="spatial_microns",
):
    ec_mask = adata.obs[celltype_col] == "ECs"
    ec_adata = adata[ec_mask].copy()

    all_coords = np.asarray(adata.obsm[spatial_key])

    if all_coords.ndim != 2 or all_coords.shape[1] < 2:
        raise ValueError(f"adata.obsm[{spatial_key!r}] must have at least 2 columns.")

    coords_df = pd.DataFrame(
        all_coords[:, :2],
        index=adata.obs_names,
        columns=["x_coord", "y_coord"],
    )

    mural_labels = pd.Series(index=adata.obs_names, dtype="object")
    mural_labels.loc[adata.obs[celltype_col] == "Pericytes"] = "Pericytes"

    if "smc_subtype" in adata.obs.columns:
        mural_labels.loc[
            adata.obs["smc_subtype"] == "arterial_arteriolar_SMCs"
        ] = "arterial_arteriolar_SMCs"

        mural_labels.loc[
            adata.obs["smc_subtype"] == "venous_SMCs"
        ] = "venous_SMCs"

    mural_mask = mural_labels.notna()

    if mural_mask.sum() == 0:
        raise ValueError(
            "No Pericytes, arterial SMCs, or venous SMCs found for EC annotation."
        )

    mural_coords = coords_df.loc[mural_mask, ["x_coord", "y_coord"]].to_numpy()
    mural_celltypes = mural_labels.loc[mural_mask].to_numpy()
    ec_coords = coords_df.loc[ec_adata.obs_names, ["x_coord", "y_coord"]].to_numpy()

    distances, nearest_indices = cKDTree(mural_coords).query(ec_coords, k=1)

    ec_adata.obs["nearest_mural_celltype"] = mural_celltypes[nearest_indices]
    ec_adata.obs["distance_to_nearest_mural"] = distances
    ec_adata.obs["ec_subtype"] = "other_EC"

    ec_adata.obs.loc[
        ec_adata.obs["nearest_mural_celltype"] == "arterial_arteriolar_SMCs",
        "ec_subtype",
    ] = "aEC"

    ec_adata.obs.loc[
        ec_adata.obs["nearest_mural_celltype"] == "venous_SMCs",
        "ec_subtype",
    ] = "vEC"

    ec_adata.obs.loc[
        ec_adata.obs["nearest_mural_celltype"] == "Pericytes",
        "ec_subtype",
    ] = "capEC"

    adata.obs["nearest_mural_celltype"] = "non_EC"
    adata.obs["ec_subtype"] = "non_EC"
    adata.obs["distance_to_nearest_mural"] = np.nan

    adata.obs.loc[
        ec_adata.obs_names,
        "nearest_mural_celltype",
    ] = ec_adata.obs["nearest_mural_celltype"].astype(str)

    adata.obs.loc[
        ec_adata.obs_names,
        "ec_subtype",
    ] = ec_adata.obs["ec_subtype"].astype(str)

    adata.obs.loc[
        ec_adata.obs_names,
        "distance_to_nearest_mural",
    ] = ec_adata.obs["distance_to_nearest_mural"].values

    return adata


def annotate_ec_subtypes_marker_genes(ec_adata):
    selected_ec_subtypes = {
        "aECs": [
            "Bmx", "Efnb2", "Vegfc", "Mgp", "Cytl1", "Sema3g",
            "Gkn3", "Fbln2", "Hey1", "Egfl8", "Jag1", "Igf2",
            "Notch3", "Mgp", "Clu",
        ],
        "capECs": [
            "Slc7a5", "Mfsd2a", "Tfrc", "Slc16a1", "Meox1",
            "Col4a3", "Angpt2", "Rgcc", "Cxcl12", "Ecscr",
            "Apln", "Car4",
        ],
        "vECs": [
            "Nr2f2", "Slc38a5", "Flrt2", "Ier3", "Ackr1",
            "Lcn2", "Vcam1", "Ly6c1", "Ly6a", "Ctsc",
        ],
    }

    for subtype, markers in selected_ec_subtypes.items():
        present_markers = [gene for gene in markers if gene in ec_adata.var_names]

        if present_markers:
            sc.tl.score_genes(
                ec_adata,
                gene_list=present_markers,
                score_name=subtype,
            )
        else:
            ec_adata.obs[subtype] = 0.0

    selected_subtype_scores = list(selected_ec_subtypes.keys())
    max_scores = ec_adata.obs[selected_subtype_scores].max(axis=1)
    best_subtypes = ec_adata.obs[selected_subtype_scores].idxmax(axis=1)

    ec_adata.obs["ec_subtype"] = best_subtypes
    ec_adata.obs.loc[max_scores <= 0.0, "ec_subtype"] = "unassigned"

    ec_adata = ec_adata[ec_adata.obs["ec_subtype"] != "unassigned"].copy()

    return ec_adata


def create_overlap_adata(
    adata_base,
    adata_distance_only,
    celltype_col="cell_type_incl_low_quality_revised",
):
    adata_ec_marker = adata_base[adata_base.obs[celltype_col] == "ECs"].copy()
    adata_ec_marker = annotate_ec_subtypes_marker_genes(adata_ec_marker)

    adata_ec_distance = adata_distance_only[
        adata_distance_only.obs[celltype_col] == "ECs"
    ].copy()

    rename_map = {
        "aECs": "aEC",
        "capECs": "capEC",
        "vECs": "vEC",
    }

    adata_ec_marker.obs["ec_subtype"] = adata_ec_marker.obs["ec_subtype"].replace(
        rename_map
    )

    common_cells = adata_ec_marker.obs_names.intersection(adata_ec_distance.obs_names)

    adata_ec_marker_sub = adata_ec_marker[common_cells].copy()
    adata_ec_distance_sub = adata_ec_distance[common_cells].copy()
    adata_ec_distance_sub = adata_ec_distance_sub[adata_ec_marker_sub.obs_names].copy()

    match_mask = (
        adata_ec_marker_sub.obs["ec_subtype"].notna()
        & adata_ec_distance_sub.obs["ec_subtype"].notna()
        & (
            adata_ec_marker_sub.obs["ec_subtype"]
            == adata_ec_distance_sub.obs["ec_subtype"]
        )
    )

    adata_ec_match = adata_ec_marker_sub[match_mask].copy()
    adata_ec_match.obs["ec_subtype_match"] = (
        adata_ec_marker_sub.obs.loc[match_mask, "ec_subtype"].astype("string").values
    )

    adata_overlap = adata_base.copy()
    adata_overlap.obs["ec_subtype"] = "non_EC"

    common_cells_full = adata_overlap.obs_names.intersection(adata_ec_match.obs_names)

    adata_overlap.obs.loc[common_cells_full, "ec_subtype"] = (
        adata_ec_match.obs.loc[common_cells_full, "ec_subtype_match"].astype("string")
    )

    adata_overlap.obs["ec_subtype"] = pd.Categorical(
        adata_overlap.obs["ec_subtype"],
        categories=["aEC", "capEC", "vEC", "non_EC"],
    )

    return adata_overlap

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create adata_distance_only.h5ad and adata_overlap.h5ad from an input AnnData file."
        )
    )

    parser.add_argument("--input",required=True,help="Path to input .h5ad file.")
    parser.add_argument("--output",required=True, help="Directory where output .h5ad files will be written.")

    parser.add_argument("--brain_areas",help=(
            "spatial registration CSV with brain-area labels."))
    parser.add_argument("--age",default="3",help="Age value to keep from adata.obs['age_months']. Default: 3.")
    parser.add_argument("--keep-areas",nargs="+",default=["BS", "CTX", "HIP", "STR", "fiber_tracts", "VS"],help="Brain areas to keep.")
    parser.add_argument("--celltype-col",default="cell_type_incl_low_quality_revised",help="Column in adata.obs containing cell type labels.")
    parser.add_argument("--spatial-key",default="spatial_microns",help="Key in adata.obsm containing spatial coordinates.")
    parser.add_argument("--counts-layer",default="counts",help="Layer used for adata.raw. Default: counts.")
    parser.add_argument("--log1p-layer",default="librarysize_log1p_norm",help="Layer used as adata.X. Default: librarysize_log1p_norm.")
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    distance_out = os.path.join(args.output, "adata_distance_only.h5ad")
    overlap_out = os.path.join(args.output, "adata_overlap.h5ad")

    adata = sc.read_h5ad(args.input)

    adata = reduce_anndata_size(
        adata,
        spatial_key=args.spatial_key,
    )

    adata = prepare_anndata(
        adata,
        counts_layer=args.counts_layer,
        log1p_layer=args.log1p_layer,
        celltype_col=args.celltype_col,
    )

    adata = annotate_brain_areas(adata, args.brain_areas)

    adata = adata[adata.obs["age_months"].astype(str) == str(args.age)].copy()
    adata = adata[adata.obs["brain_area"].isin(args.keep_areas)].copy()

    adata_base = annotate_smc(
        adata,
        celltype_col=args.celltype_col,
    )

    adata_distance_only = annotate_ec_subtypes_distance(
        adata_base.copy(),
        celltype_col=args.celltype_col,
        spatial_key=args.spatial_key,
    )

    adata_overlap = create_overlap_adata(
        adata_base=adata_base,
        adata_distance_only=adata_distance_only,
        celltype_col=args.celltype_col,
    )

    adata_distance_only.write_h5ad(distance_out, compression="gzip")
    adata_overlap.write_h5ad(overlap_out, compression="gzip")

    print(f"Saved: {distance_out}")
    print(f"Saved: {overlap_out}")
    print("Distance-only EC subtype counts:")
    print(adata_distance_only.obs["ec_subtype"].value_counts())
    print("Overlap EC subtype counts:")
    print(adata_overlap.obs["ec_subtype"].value_counts())


if __name__ == "__main__":
    main()