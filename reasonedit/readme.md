# step 1

```bash
cd jobs/tune_scaler && sbatch tune.sbatch
```

Writes `auto_scaler/scalers.yaml` (default: DINOv2-large CLS + sbert on COCO).

# step 2

```python
load_config("medgemma", mode="standalone_dual",
            image_encoder="dinov2", image_encoder_pool="cls",
            text_encoder="sbert", scaler_dataset="coco")
```

`ReasonEdit` reads the recorded weight. Pass `lang_scaler=` only to override it.
