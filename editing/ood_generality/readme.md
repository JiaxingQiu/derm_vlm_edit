`run.py` builds one codebook per model from all three dermatologists, then scores OOD generality for `standalone_dual` and `language_last` at reject percentiles 1 and 5. Resume keeps a finished generality file. Results go under `results/ood_gen/{embed_mode}_{threshold}/collect/`. The earlier runs are `standalone_dual_0.1` and `language_last_0.1`.

    python run.py --pct_threshold 1
    python run.py --pct_threshold 5

`run_collect.ipynb` is the same run for one embedding mode and one threshold. Submit the two percentiles with `jobs/ood_gen/pct1.sbatch` and `jobs/ood_gen/pct5.sbatch`.

A full configuration, perturb metrics and OOD generality together, is `jobs/eval/run.sbatch`. One job is one retrieval image and one encoder, and it loops MedGemma then DermatoLlama.

`photo` and `dscope` use the same retrieval-key switch as perturb_generality. Generation stays on the combined image. Results land in `results/ood_gen/{embed_mode}_{threshold}_{photo|dscope}/collect/`. `dinov2` keeps those names. Another image encoder adds its short name, for example `standalone_dual_10_photo_panderm`. The saved prediction CSVs are `data/medgemma_predictions_reason.csv` and `data/dermato_llama_predictions_reason.csv`. This folder's own jobs still score only OOD generality. Submit from `jobs/ood_gen`:

    sbatch seq_key.sbatch photo --pct_threshold 10
    sbatch seq_key.sbatch dscope --pct_threshold 10
    sbatch seq_key.sbatch photo --image-encoder panderm --pct_threshold 10

Reliability and locality are the perturb_generality collect scores. This run does not recompute them. It edits once, on the original image and question, then generates on leftover combined MIDAS cases.

A case marked wrong by any user is an edit. If two users mark it wrong and give different corrections, the case is printed and left out of the codebook. If one user marks it wrong and another marks it correct, the correction is kept.

**Generality** uses combined MIDAS cases these users did not complete. Each case has a `y16` label and a saved model prediction. One generation with the codebook is scored against that `y16`.

`gen_all before` is the saved top-1 against `y16`. `gen_all after` is the edited generation against the same `y16`. `gen_acc` is how many originally wrong cases become correct. `gen_loc` is how many originally correct cases stay correct.
