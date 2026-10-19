# Run

The first step is `auto_scaler`: one balancing weight per image encoder, text encoder, and dataset.

```bash
python -m reasonedit.run.auto_scaler
```

That records DINOv2-large (CLS) + `sbert` on COCO in `reasonedit/auto_scaler/scalers.yaml`. An existing combo is kept unless `--overwrite`.

```bash
python -m reasonedit.run.auto_scaler --image_encoder biomedclip --text_encoder minilm --dataset coco
python -m reasonedit.run.auto_scaler --image_encoder dinov2 --text_encoder sbert --dataset coco --overwrite
python -m reasonedit.run.auto_scaler --image_encoder dinov2 --text_encoder sbert --dataset myset \
    --images_dir /path/to/images --captions_file /path/to/captions.json
```

From the repo root, the same default is `cd jobs/tune_scaler && sbatch tune.sbatch`.
