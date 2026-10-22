Both notebooks are front ends for the same three steps. `run_individual.ipynb` builds one codebook per dermatologist. `run_collect.ipynb` puts all three dermatologists into one codebook, prints cases they share, then scores that codebook.

Each run edits once, on the original image and question, then scores in order:

1. **Reliability** uses the original image and question.
2. **Text generality** swaps in one rephrased question.
3. **Image generality** swaps in one perturbed image.
