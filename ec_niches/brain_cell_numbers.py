#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
warnings.filterwarnings("ignore", category=UserWarning, module="xarray_schema")
warnings.filterwarnings("ignore", category=FutureWarning, module="squidpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")

import dask
dask.config.set({"dataframe.query-planning": True})

import anndata as ad
import squidpy as sq
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import spatialdata as sd
from matplotlib.colors import ListedColormap
import matplotlib as mp
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import argparse

def prepare_anndata(adata, counts, log_norm, celltype):
    adata.raw = ad.AnnData(
        X=adata.layers[counts].copy(),
        obs=adata.obs.copy(),
        var=adata.var.copy()
    )
    adata.X = adata.layers[log_norm].copy()
    adata = adata[adata.obs[celltype] != "Undefined"].copy()
    return adata


def annotate_brain_areas(adata, brain_areas_csv):
    brain_areas_csv = brain_areas_csv.rename(columns={"Unnamed: 0": "cell_id"})
    brain_areas_csv = brain_areas_csv.set_index("cell_id")
    brain_areas_csv = brain_areas_csv.reindex(index=adata.obs.index)

    adata.obs["brain_area"] = brain_areas_csv["label"]
    adata.obs["brain_area"] = (
        adata.obs["brain_area"]
        .astype("string")
        .str.replace("/", "_", regex=False)
    )
    adata = adata[adata.obs["brain_area"].notna(), :]
    return adata

def annotate_ec_subtypes(ec_adata):
    # from https://github.com/simonmfr/cellseg-benchmark/blob/main/cellseg_benchmark/_constants.py#L705
    selected_ec_subtypes = {
        "aECs": [
            "Bmx",
            "Efnb2",
            "Vegfc",
            "Mgp",
            "Cytl1",
            "Sema3g",
            "Gkn3",
            "Fbln2",
            "Hey1",
            "Egfl8",
            "Jag1",
            "Igf2",
            "Notch3",
            "Mgp",
            "Clu",
        ],
        "capECs": [
            "Slc7a5",
            "Mfsd2a",
            "Tfrc",
            "Slc16a1",
            "Meox1",
            "Col4a3",
            "Angpt2",
            "Rgcc",
            "Cxcl12",
            "Ecscr",
            "Apln",
            "Car4",
        ],
        "vECs": [
            "Nr2f2",
            "Slc38a5",
            "Flrt2",
            "Ier3",
            "Ackr1",
            "Lcn2",
            "Vcam1",
            "Ly6c1",
            "Ly6a",
            "Ctsc",
        ],
    }

    for subtype, markers in selected_ec_subtypes.items():
        sc.tl.score_genes(ec_adata, gene_list=markers, score_name=subtype)

    selected_subtype_scores = list(selected_ec_subtypes.keys())

    # Compute which subtype has the highest score per cell
    max_scores = ec_adata.obs[selected_subtype_scores].max(axis=1)
    best_subtypes = ec_adata.obs[selected_subtype_scores].idxmax(axis=1)

    # Apply threshold: assign subtype only if max score > 0
    ec_adata.obs["ec_subtype"] = best_subtypes
    ec_adata.obs.loc[max_scores <= 0, "ec_subtype"] = "unassigned"
    ec_adata = ec_adata[ec_adata.obs.ec_subtype != "unassigned"]
    ec_adata = ec_adata.copy()
    return ec_adata

def parse_args():
    parser = argparse.ArgumentParser(
        description="Count ECs and total cells per brain area."
    )

    parser.add_argument("--anndata_file", required=True, help="Path to AnnData .h5ad file")
    parser.add_argument("--brain_areas", required=True, help="Path to brain area annotation CSV")
    parser.add_argument("--out", required=True, help="Output folder for figures")
    parser.add_argument("--counts", default="counts", help="Layer containing raw counts")
    parser.add_argument("--log_norm",default="librarysize_log1p_norm",help="Layer containing log1p-normalised counts")
    parser.add_argument("--cell_types",default="cell_type_incl_low_quality_revised",help="obs column containing cell types")
    return parser.parse_args()

def main():
    args = parse_args()
    sc.settings.figdir = args.out
    os.makedirs(args.out, exist_ok=True)
    csv_dir = os.path.join(args.out, "csvs")
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    adata = sc.read_h5ad(args.anndata_file)
    brain_areas = pd.read_csv(args.brain_areas)

    adata = prepare_anndata(
        adata,
        counts=args.counts,
        log_norm=args.log_norm,
        celltype=args.cell_types,
    )

    adata = annotate_brain_areas(adata, brain_areas)
    ec_adata = adata[adata.obs[args.cell_types] == "ECs"].copy()
    ec_adata = annotate_ec_subtypes(ec_adata)

    numbers_ec_types = (
    ec_adata.obs.groupby(["brain_area", "ec_subtype"])
    .size()
    .reset_index(name="cell_numbers")
    )
    numbers_ec_types.to_csv(
    os.path.join(csv_dir, "numbers_ec_types.csv"),
    index=False
    )
    print("Numbers for EC subtypes calculated")

    plt.figure(figsize=(8,5))

    ax = sns.barplot(
        data=numbers_ec_types,
        x="brain_area",
        y="cell_numbers",
        hue="ec_subtype"
    )

    ax.set_title("Cell numbers per brain area and EC sub type", pad=10)

    plt.xticks(rotation=45, ha="right")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.legend(
        title="EC subtype",
        loc="upper right",
        frameon=True
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, "ec_numbers_area.pdf"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    numbers_area = (
        adata.obs.groupby("brain_area")
        .size()
        .reset_index(name="cell_numbers")
    )
    numbers_area.to_csv(
        os.path.join(csv_dir, "numbers_area.csv"),
        index=False
    )
    
    plt.figure(figsize=(8,5))
    
    ax = sns.barplot(
        data=numbers_area,
        x="brain_area",
        y="cell_numbers",
    )

    ax.set_title("Cell numbers per brain area", pad=10)
    plt.xticks(rotation=45, ha="right")

    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, "cell_numbers_area.pdf"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    
    print("Done")

if __name__ == "__main__":
    main()







