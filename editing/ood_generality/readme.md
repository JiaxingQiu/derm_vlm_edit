`run.py` builds one codebook per model from all three dermatologists, then scores OOD generality for `standalone_dual` and `language_last`. Resume keeps a finished generality file. Results go under `results/{embed_mode}/collect/`.

    python run.py

`run_collect.ipynb` is the same run for one embedding mode. Submit both modes with `jobs/ood_gen/run.sbatch`.

Reliability and locality are the perturb_generality collect scores. This run does not recompute them. It edits once, on the original image and question, then generates on leftover combined MIDAS cases.

A case marked wrong by any user is an edit. If two users mark it wrong and give different corrections, the case is printed and left out of the codebook. If one user marks it wrong and another marks it correct, the correction is kept.

**Generality** uses combined MIDAS cases these users did not complete. Each case has a `y16` label and a saved model prediction. Before is that saved top-1 against `y16`. After is one generation with the codebook, against the same `y16`. The after number is generality.
