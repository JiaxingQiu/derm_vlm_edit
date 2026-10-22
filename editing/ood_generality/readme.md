`run.py` builds one codebook per model from all three dermatologists, then scores `standalone_dual` and `language_last`. Resume keeps finished metrics. Results go under `results/{embed_mode}/collect/`.

    python run.py

`run_collect.ipynb` is the same collective run for one embedding mode. Submit the full run with `jobs/ood_gen/run.sbatch`.

A case marked wrong by any user is an edit. If two users mark it wrong and give different corrections, the case is printed and left out of the codebook. If one user marks it wrong and another marks it correct, the correction is kept.

Each run edits once, on the original image and question, then scores in order:

1. **Reliability** uses the original image and question on the edit set. The target is the human correction.
2. **Locality** uses cases no user marked wrong. The score is how often the top-1 after retrieval still matches the top-1 before retrieval.
3. **Generality** uses leftover combined MIDAS cases: a `y16` label, a saved model prediction, and no review from these users. Before is that saved top-1 against `y16`. After is one generation with the codebook, against the same `y16`. The after number is generality.

**Overall** is reliability and locality combined: successes on both sets divided by the number of cases in both sets.

Generality generates on every leftover combined case, so it is the long step. A finished reliability or locality file is kept if the job stops during generality.
