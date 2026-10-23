`run.py` does the collective run only. Each uses `standalone_dual`, then `language_last`, at reject percentiles 1 and 5. Resume keeps finished metrics. Results go under `results/{embed_mode}_{threshold}/collect/`. The earlier runs are `standalone_dual_0.1` and `language_last_0.1`.

A percentile that comes out as distance 0 is replaced by the smallest key-to-key distance above 0. If every pair is 0, the gate stays shut.

    python run.py --pct_threshold 1
    python run.py --pct_threshold 5

The notebooks are the same steps for one mode at a time. `run_individual.ipynb` builds one codebook per dermatologist. `run_collect.ipynb` puts all three dermatologists into one codebook, one case per sample.

A case marked wrong by any user is an edit. If two users mark it wrong and give different corrections, the case is printed and left out of the codebook. If one user marks it wrong and another marks it correct, the correction is kept.

Each run edits once, on the original image and question, then scores in order:

1. **Reliability** uses the original image and question on the edit set.
2. **Text generality** swaps in one rephrased question on the edit set.
3. **Image generality** swaps in one perturbed image on the edit set.
4. **Locality** uses cases no user marked wrong. The score is how often the top-1 after retrieval still matches the top-1 before retrieval.
