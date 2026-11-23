# Run

The first step is `auto_scaler`. One job records both balancing weights for an encoder combo: COCO, then the Fitzpatrick17k dermatology subset.

```bash
python -m reasonedit.run.auto_scaler
```

That records DINOv2-large (CLS) + `sbert` on COCO in `reasonedit/auto_scaler/scalers.yaml`. An existing combo is kept unless `--overwrite`.

```bash
cd jobs/tune_scaler
sbatch tune.sbatch dinov2 sbert
sbatch tune.sbatch biomedclip sbert
```

`derm` downloads 50 Fitzpatrick17k images into `reasonedit/auto_scaler/data_derm/` on the first run. An existing combo is kept unless `--overwrite`. The eval job chooses the stored weight with `--scaler-dataset coco` or `--scaler-dataset derm`.
