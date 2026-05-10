# Mira_BSc_thesis

Scripts that were used to generate the analyses and plots described in my BSc thesis on EC niches in vascular aging in the mouse brain.

## set up ec_niches environment (python)

```bash
conda env create -f ec_niches.yaml
conda activate ec_niches
```

## Analyses within the ec_niche environment

To visualise brain area annotation on the slide level run: 

```bash
python brain_area_visualisation.py \
  --anndata_file path_to_anndata_object \
  --brain_areas path_to_csv_containing_coordinates_for_brain_area_annotation \
  --out path_to_out_dir
```

To calculate cell numbers per brain area and visualise the results:

```bash
python brain_cell_numbers.py \
  --anndata_file \
  --brain_areas \
  --out
```

To compute ec niches at baseline:

```bash
python mixed_effect.py \
  --anndata_file \
  --brain_areas \
  --out
```
  
To compute niches across ages:

```bash
python niches_all_ages.py \
  --anndata_file \
  --brain_areas \
  --out
```

For the annotation of EC and SMC sub types:

```bash
python SMC_EC_classification.py /
--anndata_file \
--out
```
Script for generating the anndata objects required for distance - and overlap-based EC sub type annotation

```bash
python anndata_for_dge.py \
  --input path_to_anndata_object\
  --output path_to_out_put\
  --brain_areas path_to_csv_containing_brain_area_annotation
  ```


## set up dge_ecs environment (python)

```bash
conda env create -f dge_ecs.yaml
conda activate dge_ecs
```

## Analyses within dge_ecs environment (python)

Script for differential transcript expression analysis of EC sub types annotated based on distance to closest mural cell

```bash
python distance_dge.py \
  --anndata_file path_to_anndata_file_with_distance_based_EC_annotation \
  --out path_to_out_put_dir
```

Script for differential transcript expression analysis of EC sub types annotated based on consensus of distance to closest mural cell and EC marker gene expression

```bash
python overlap_dge.py \
--anndata_file path_to_anndata_file_with_consensus_based_EC_annotation\
--out
```
## Analyses within glmm_env environment (R)

```bash
conda env create -f glmm_env.yaml
conda activate glmm_env
```

