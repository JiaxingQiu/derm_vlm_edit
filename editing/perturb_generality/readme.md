`run.py` does the collective run only. Each uses `standalone_dual`, then `language_last`, at reject percentiles 1 and 5. Resume keeps finished metrics. Results go under `results/perturb_gen/{embed_mode}_{threshold}/collect/`. The earlier runs are `standalone_dual_0.1` and `language_last_0.1`.

A percentile that comes out as distance 0 is replaced by the smallest key-to-key distance above 0. If every pair is 0, the gate stays shut.

    python run.py --pct_threshold 1
    python run.py --pct_threshold 5

The notebooks are the same steps for one mode at a time. `run_individual.ipynb` builds one codebook per dermatologist. `run_collect.ipynb` puts all three dermatologists into one codebook, one case per sample.

A case marked wrong by any user is an edit. If two users mark it wrong and give different corrections, the case is printed and left out of the codebook. If one user marks it wrong and another marks it correct, the correction is kept.

A full configuration, perturb metrics and OOD generality together, is `jobs/eval/run.sbatch`. One job is one retrieval image and one encoder, and it loops MedGemma then DermatoLlama.

`photo` and `dscope` are retrieval-key ablations. The VLM still generates from the combined image. The codebook and the query use `data/images/{id}_photo.jpg` or `{id}_dscope.jpg`. `combined` keeps the folders above. The other two land in `results/perturb_gen/{embed_mode}_{threshold}_{photo|dscope}/collect/`. `dinov2` keeps those names. Another image encoder adds its short name, for example `standalone_dual_10_photo_panderm`. This folder's own jobs still score only the perturb suite. Submit from `jobs/perturb_gen`:

    sbatch seq_key.sbatch photo --pct_threshold 10
    sbatch seq_key.sbatch dscope --pct_threshold 10
    sbatch seq_key.sbatch photo --image-encoder panderm --pct_threshold 10

Each run edits once, on the original image and question, then scores in order:

1. **Reliability** uses the original image and question on the edit set.
2. **Text generality** swaps in one rephrased question on the edit set.
3. **Image generality** perturbs the retrieval image. For `combined` that is the combined canvas. For `photo` and `dscope` it is the sibling file, and generation stays on the combined image.
4. **Locality** uses cases no user marked wrong. The score is how often the top-1 after retrieval still matches the top-1 before retrieval.
