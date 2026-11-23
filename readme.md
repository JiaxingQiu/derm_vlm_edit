## DermEdit: Editing VLMs using Real Dermotologist Corrections

### Requirements

```bash
cd derm_vlm_edit
conda create -n dermedit python=3.10 -y
conda activate dermedit
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

(optional) jupyter notebook
```bash
conda install jupyterlab ipykernel notebook -y
jupyter kernelspec uninstall dermedit -y
python -m ipykernel install --user --name dermedit --display-name "dermedit"
jupyter kernelspec list
```


# Editing Pipeline

Two stages: tune the balancing weight, then edit and evaluate.

## Project layout

```
jobs/
├── tune_scaler/
│   └── tune.sbatch               # Stage 1: one encoder → coco + derm weights
└── eval/
    ├── run_combined.sbatch        # Stage 2: combined (hartvigsen_lab)
    ├── run_photo.sbatch           #           photo   (cama)
    ├── run_dscope.sbatch          #           dscope  (cama)
    └── run.sbatch                 #           generic (any image, hartvigsen_lab)

editing/
├── eval/run.py                    # driver: both suites, both models, one config
├── perturb_generality/            # reliability, text/image generality, locality
│   ├── run.py                     #   single-suite runner
│   └── utils/                     #   scoring, perturbation, bootstrapping
├── ood_generality/                # OOD generality / locality / accuracy
│   ├── run.py                     #   single-suite runner
│   └── utils/                     #   scoring, bootstrapping
├── reliability/                   # standalone reliability checks
├── shared/
│   ├── config.py                  # paths, constants, NORM_ANSWER
│   ├── retrieval_images.py        # results_tag(), key_image switch
│   ├── derm_dataset.py            # DermEditDataset wrapper
│   └── data_engineering/          # normalize_dx, build_edit_set, parse_annotations
├── data_summary/                  # annotator accuracy vs MIDAS ground truth
└── report.ipynb                   # read JSONs → 3-sheet xlsx per (encoder, dataset)

reasonedit/
├── auto_scaler/
│   ├── scalers.yaml               # recorded balancing weights
│   ├── tune.py                    # scaler sweep logic
│   └── download_derm.py           # Fitzpatrick17k subset downloader
└── encoders/
    ├── dinov2.py                  # DINOv2-large (1024-d, default)
    ├── panderm.py                 # PanDerm-large (1024-d, derm-specific)
    └── biomedclip.py              # BiomedCLIP (512-d, biomedical)
```

## Stage 1 — Scaler tuning (`jobs/tune_scaler`)

One job per image encoder records two balancing weights: one from COCO images and one from a Fitzpatrick17k dermatology subset. Existing entries are kept unless `--overwrite`.

```bash
cd /scratch/jq2uw/MME/derm_vlm_edit/jobs/tune_scaler

sbatch tune.sbatch dinov2 sbert
sbatch tune.sbatch panderm sbert
sbatch tune.sbatch biomedclip sbert
```

Weights are stored in `reasonedit/auto_scaler/scalers.yaml`:

```
reasonedit/auto_scaler/scalers.yaml
  {encoder}__{text_encoder}__{dataset}:    # e.g. panderm_cls__sbert__coco
    lang_scaler: <float>
```

## Stage 2 — Editing and evaluation (`jobs/eval`)

One job is one (encoder × retrieval image × scaler dataset) configuration. It edits MedGemma then DermatoLlama, and writes all eight metrics. Generation always uses the combined image; `photo` and `dscope` change only the retrieval key.

| Hyperparameter | Options |
|---|---|
| Image encoder | `dinov2`, `panderm`, `biomedclip` |
| Retrieval image | `combined`, `photo`, `dscope` |
| Scaler dataset | `coco`, `derm` |

That is 3 × 3 × 2 = **18 jobs**.

Three sbatch files bake in the retrieval image and the account:

| File | Retrieval image | Account |
|---|---|---|
| `run_combined.sbatch` | combined | hartvigsen_lab |
| `run_photo.sbatch` | photo | cama |
| `run_dscope.sbatch` | dscope | cama |

```bash
cd /scratch/jq2uw/MME/derm_vlm_edit/jobs/eval

# dinov2, coco  (done)
# sbatch run_combined.sbatch
# sbatch run_photo.sbatch
# sbatch run_dscope.sbatch

# dinov2, derm
sbatch run_combined.sbatch --scaler-dataset derm
sbatch run_photo.sbatch    --scaler-dataset derm
sbatch run_dscope.sbatch   --scaler-dataset derm

# panderm, coco
sbatch run_combined.sbatch --image-encoder panderm
sbatch run_photo.sbatch    --image-encoder panderm
sbatch run_dscope.sbatch   --image-encoder panderm

# panderm, derm
sbatch run_combined.sbatch --image-encoder panderm --scaler-dataset derm
sbatch run_photo.sbatch    --image-encoder panderm --scaler-dataset derm
sbatch run_dscope.sbatch   --image-encoder panderm --scaler-dataset derm

# biomedclip, coco
sbatch run_combined.sbatch --image-encoder biomedclip
sbatch run_photo.sbatch    --image-encoder biomedclip
sbatch run_dscope.sbatch   --image-encoder biomedclip

# biomedclip, derm
sbatch run_combined.sbatch --image-encoder biomedclip --scaler-dataset derm
sbatch run_photo.sbatch    --image-encoder biomedclip --scaler-dataset derm
sbatch run_dscope.sbatch   --image-encoder biomedclip --scaler-dataset derm
```

## Result folder structure

`dinov2` and `coco` are the defaults, so those tokens are omitted from folder names. Another encoder or `derm` appends its name.

```
results/
├── perturb_gen/
│   └── {tag}/collect/
│       └── {metric}__all__{model}.json
└── ood_gen/
    └── {tag}/collect/
        └── generality__all__{model}.json
```

Tag examples:

| Encoder | Image | Scaler | Tag |
|---|---|---|---|
| dinov2 | combined | coco | `standalone_dual_10` |
| dinov2 | photo | coco | `standalone_dual_10_photo` |
| dinov2 | combined | derm | `standalone_dual_10_derm` |
| panderm | photo | coco | `standalone_dual_10_photo_panderm` |
| panderm | photo | derm | `standalone_dual_10_photo_panderm_derm` |
| biomedclip | dscope | derm | `standalone_dual_10_dscope_biomedclip_derm` |

Perturb metrics: `reliability`, `text_generality`, `image_generality`, `locality`.
OOD metric: `generality` (contains accuracy before/after, generality, locality).
Model: `medgemma` or `dermatollama`.

## Reporting

`report.ipynb` reads the JSON files and writes a three-sheet workbook (Combined, Clinical, Dscope). Set `IMAGE_ENCODER` and `SCALER_DATASET` at the top. Default `dinov2` + `coco` writes `results/report_YYYYMMDD.xlsx`; other combos write `results/report_{encoder}_{dataset}_YYYYMMDD.xlsx`.
