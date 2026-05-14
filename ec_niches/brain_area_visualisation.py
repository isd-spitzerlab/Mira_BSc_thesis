#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
warnings.filterwarnings("ignore", category=UserWarning, module="xarray_schema")
warnings.filterwarnings("ignore", category=FutureWarning, module="squidpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")

import dask
dask.config.set({"dataframe.query-planning": True})

import argparse
import os
import anndata as ad
import pandas as pd
import scanpy as sc
import squidpy as sq

def prepare_anndata(adata, counts, log_norm, celltype):
    adata.raw = ad.AnnData(
        X=adata.layers[counts].copy(),
        obs=adata.obs.copy(),
        var=adata.var.copy()
    )

    adata.X = adata.layers[log_norm].copy()
    adata = adata[adata.obs[celltype] != "Undefined"].copy()

    return adata


def annotate_brain_areas(adata, brain_areas_df):
    brain_areas_df = brain_areas_df.rename(columns={"Unnamed: 0": "cell_id"})
    brain_areas_df = brain_areas_df.set_index("cell_id")
    brain_areas_df = brain_areas_df.reindex(index=adata.obs.index)

    adata.obs["brain_area"] = brain_areas_df["label"]
    adata = adata[adata.obs["brain_area"].notna()].copy()

    adata.obs["brain_area"] = (
        adata.obs["brain_area"].str.replace("/", "_", regex=False)
    )

    return adata


def plot_brain_areas(adata, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

    samples = adata.obs["sample"].unique()
    categories = adata.obs["brain_area"].astype("category").cat.categories

    color_dict = {
        "BS": "#FFFF00",
        "BS_STR": "#1CE6FF",
        "CAsp": "#FF34FF",
        "CTX": "#FF4A46",
        "DG-sg": "#008941",
        "HIP": "#006FA6",
        "Meninges": "#A30059",
        "STR": "#FFDBE5",
        "STR_CTX": "#7A4900",
        "VS": "#0000A6",
        "fiber_tracts": "#63FFAC",
    }

    adata.uns["brain_area_colors"] = [
        color_dict.get(c, "#808080") for c in categories
    ]

    for s in samples:
        adata_s = adata[adata.obs["sample"] == s].copy()

        adata_s.obsm["spatial"] = adata_s.obsm["spatial_microns"].copy()

        adata_s.obs["brain_area"] = adata_s.obs["brain_area"].astype("category")
        adata_s.obs["brain_area"] = (
            adata_s.obs["brain_area"].cat.set_categories(categories)
        )

        adata_s.uns["spatial"] = {
            s: {
                "images": {},
                "scalefactors": {
                    "tissue_hires_scalef": 1.0,
                    "spot_diameter_fullres": 1.0,
                },
            }
        }

        sq.pl.spatial_scatter(
            adata_s,
            color="brain_area",
            spatial_key="spatial",
            size=70,
            img=None,
            figsize=(8, 8),
            ncols=1,
            title=s,
            save=f"_{s}_area_annotation.pdf",
        )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise annotated brain areas on tissue slide level."
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

    adata = sc.read_h5ad(args.anndata_file)
    brain_areas = pd.read_csv(args.brain_areas)

    adata = prepare_anndata(
        adata,
        counts=args.counts,
        log_norm=args.log_norm,
        celltype=args.cell_types,
    )

    adata = annotate_brain_areas(adata, brain_areas)

    plot_brain_areas(adata, args.out)

    print("Done")

if __name__ == "__main__":
    main()


