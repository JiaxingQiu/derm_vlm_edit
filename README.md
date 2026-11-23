# DermEdit

Edit MedGemma and DermatoLlama using real dermatologist corrections of the model's top-1 diagnosis.

## Setup

```bash
cd derm_vlm_edit
conda create -n dermedit python=3.10 -y
conda activate dermedit
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Optional — Jupyter kernel:
```bash
conda install jupyterlab ipykernel notebook -y
python -m ipykernel install --user --name dermedit --display-name "dermedit"
```

`data/` is local (gitignored): annotation export, images, and MIDAS tables.

---

## Pipeline

### Stage 1 — Normalize corrections (`data_norm/`)

Rule-map free-text corrections to the 16-class MIDAS `y16` taxonomy, then ask GPT-4o only where the rule label disagrees with MIDAS pathology ground truth. Output goes to `data/normed/`. See [`data_norm/README.md`](data_norm/README.md).

```bash
cd data_norm && jupyter notebook soft_normalize.ipynb   # interactive (recommended)
python data_norm/soft_normalize.py                      # or CLI
```

### Stage 2 — Tune the balancing weight (`reasonedit/auto_scaler/`)

One job per image encoder records two balancing weights: one from COCO images and one from a Fitzpatrick17k dermatology subset. Existing entries are kept unless `--overwrite`.

```bash
cd jobs/tune_scaler
sbatch tune.sbatch dinov2 sbert
sbatch tune.sbatch panderm sbert
sbatch tune.sbatch biomedclip sbert
```

Weights land in `reasonedit/auto_scaler/scalers.yaml`.

### Stage 3 — Edit and evaluate (`editing/eval/run.py`)

One job = one (encoder × retrieval image × scaler dataset) configuration. Edits MedGemma then DermatoLlama and writes all eight metrics.

| Hyperparameter | Options |
|---|---|
| Image encoder | `dinov2`, `panderm`, `biomedclip` |
| Retrieval image | `combined`, `photo`, `dscope` |
| Scaler dataset | `coco`, `derm` |

3 × 3 × 2 = **18 jobs**. Three sbatch files bake in the retrieval image and account:

| File | Retrieval image | Account |
|---|---|---|
| `run_combined.sbatch` | combined | hartvigsen_lab |
| `run_photo.sbatch` | photo | cama |
| `run_dscope.sbatch` | dscope | cama |

```bash
cd jobs/eval
sbatch run_combined.sbatch --image-encoder panderm --scaler-dataset derm
sbatch run_photo.sbatch    --image-encoder panderm --scaler-dataset derm
sbatch run_dscope.sbatch   --image-encoder panderm --scaler-dataset derm
```

Defaults are `dinov2` + `coco`, so those flags can be omitted.

---

## Results

`dinov2` and `coco` are omitted from folder names (they are the defaults). Other encoders or `derm` are appended.

```
results/perturb_gen/{tag}/collect/{metric}__all__{model}.json
results/ood_gen/{tag}/collect/generality__all__{model}.json
```

| Encoder | Image | Scaler | Tag example |
|---|---|---|---|
| dinov2 | combined | coco | `standalone_dual_10` |
| panderm | photo | derm | `standalone_dual_10_photo_panderm_derm` |
| biomedclip | dscope | derm | `standalone_dual_10_dscope_biomedclip_derm` |

Perturb metrics: `reliability`, `text_generality`, `image_generality`, `locality`.
OOD metric: `generality` (includes accuracy before/after, generality, locality).
Model: `medgemma` or `dermatollama`.

`report.ipynb` reads the JSONs into a three-sheet workbook. Set `IMAGE_ENCODER` and `SCALER_DATASET` at the top.

---

## Layout

```
data_norm/                 Stage 1: soft y16 normalization
reasonedit/auto_scaler/    Stage 2: scaler tuning (scalers.yaml, tune.py)
editing/eval/run.py        Stage 3: edit + evaluate (one config, both models)
editing/shared/            Paths, config, annotation parsing, diagnosis normalization
editing/perturb_generality/   Reliability, text/image generality, locality
editing/ood_generality/       OOD generality / locality / accuracy
jobs/tune_scaler/          Stage 2 sbatch
jobs/eval/                 Stage 3 sbatch
results/                   JSON metrics per (tag, metric, model)
report.ipynb               JSONs → xlsx workbook
```
