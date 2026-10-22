`run.py` does collect, then individual. Each uses `standalone_dual`, then `language_last`. Resume keeps finished metrics. Results go under `results/{embed_mode}/{collect|individual}/`.

    python run.py

The notebooks are the same steps for one mode at a time. `run_individual.ipynb` builds one codebook per dermatologist. `run_collect.ipynb` puts all three dermatologists into one codebook, one case per sample.

A case marked wrong by any user is an edit. If two users mark it wrong and give different corrections, the case is printed and left out of the codebook. If one user marks it wrong and another marks it correct, the correction is kept.

Each run edits once, on the original image and question, then scores in order:

1. **Reliability** uses the original image and question on the edit set.
2. **Text generality** swaps in one rephrased question on the edit set.
3. **Image generality** swaps in one perturbed image on the edit set.
4. **Locality** uses cases no user marked wrong. The score is how often the top-1 after retrieval still matches the top-1 before retrieval.

**Overall** is reliability and locality combined: successes on both sets divided by the number of cases in both sets.
