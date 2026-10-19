"""Standalone AutoScaler: find optimal lang_scaler for dual embeddings.

Unlike the original AutoScaler in ``reasonedit/revlm/editors/auto_q``, this
version does **not** require a loaded VLM.  It operates entirely with a
standalone ``ImageEncoder`` (e.g. DINOv2, BiomedCLIP) and Sentence-BERT.

Usage (notebook-friendly)::

    from reasonedit.auto_scaler import StandaloneAutoScaler
    from reasonedit.encoders import get_image_encoder

    img_enc = get_image_encoder("dinov2", device="cuda:0")
    searcher = StandaloneAutoScaler(img_enc, device="cuda:0", n_samples=10)
    results = searcher.search(images, texts)
    searcher.plot(results)
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .modularity_core import ModularityCore
from ..encoders.base import ImageEncoder
from ..encoders.text import SentenceEncoder


class StandaloneAutoScaler(ModularityCore):
    """Find optimal lang_scaler using standalone image + text encoders.

    Measures Vision Q (image clustering) and Language Q (text clustering)
    on a grid of ``lang_scaler`` values and picks the scaler that
    maximises the harmonic mean of the two.

    Args:
        image_encoder: An ``ImageEncoder`` instance (e.g. DINOv2Encoder).
        device: torch device for Q computation.
        n_samples: Number of (image, text) samples to draw.
        edge_filter: Edge filtering method (``"none"``, ``"percentile"``,
            ``"knn"``, ``"disparity"``).
        edge_filter_kwargs: Extra kwargs for the filter.
        sbert_model: Sentence-BERT model name.
    """

    def __init__(
        self,
        image_encoder: ImageEncoder,
        device: Union[str, torch.device] = "cpu",
        n_samples: int = 10,
        edge_filter: str = "none",
        edge_filter_kwargs: Optional[dict] = None,
        text_encoder: Optional[SentenceEncoder] = None,
        sbert_model: str = "sentence-transformers/paraphrase-mpnet-base-v2",
    ):
        dev = torch.device(device) if isinstance(device, str) else device
        super().__init__(dev)

        self.image_encoder = image_encoder
        self.n_samples = n_samples
        self.edge_filter = edge_filter
        self.edge_filter_kwargs = edge_filter_kwargs or {}
        if text_encoder is None:
            from ..encoders import get_text_encoder
            text_encoder = get_text_encoder("sbert", device=dev, model_name=sbert_model, short_name="sbert")
        self.text_encoder = text_encoder
        self._img_aug = None

        self._images: Optional[List[Image.Image]] = None
        self._texts: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # lazy helpers
    # ------------------------------------------------------------------

    def _encode_text(self, text: str) -> torch.Tensor:
        """[1, dim] float32."""
        emb = self.text_encoder.encode([text] if isinstance(text, str) else text)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return emb.to(self.device, torch.float32)

    def _augment_image(self, image: Image.Image) -> Image.Image:
        """Light photometric jitter. The caption is kept, so no extra language model is loaded."""
        if self._img_aug is None:
            from torchvision import transforms as T
            self._img_aug = T.Compose([
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ])
        return self._img_aug(image.convert("RGB"))

    # ------------------------------------------------------------------
    # encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_dual(self, image: Image.Image, text: str, lang_scaler: float) -> torch.Tensor:
        """Encode a single (image, text) pair → [1, img_dim + sbert_dim]."""
        img_emb = self.image_encoder.encode([image])
        txt_emb = self._encode_text(text) * lang_scaler
        return torch.cat([img_emb.to(self.device), txt_emb], dim=-1).cpu()

    @torch.no_grad()
    def _encode_image_only(self, image: Image.Image) -> torch.Tensor:
        """Vision-only baseline → [1, img_dim]."""
        return self.image_encoder.encode([image]).cpu()

    @torch.no_grad()
    def _encode_text_only(self, text: str) -> torch.Tensor:
        """SBERT-only baseline → [1, 768]."""
        return self._encode_text(text).cpu()

    @torch.no_grad()
    def _encode_all_pairs(self, encode_fn, verbose=True):
        """Encode all n×n (image, text) pairs."""
        n = len(self._images)
        embs = []
        pairs = [(i, j) for i in range(n) for j in range(n)]
        for i, j in (tqdm(pairs, desc="encoding", leave=False) if verbose else pairs):
            embs.append(encode_fn(self._images[i], self._texts[j]))
        return torch.cat(embs, dim=0)

    @torch.no_grad()
    def _encode_bimodal(self, lang_scaler: float, verbose=True):
        """Encode bimodal set: anchors + augmented + cross-combinations.

        Returns (embs, img_labels, text_labels).
        """
        n = len(self._images)
        embs, img_labels, text_labels = [], [], []
        n_aug = 2 * (n - 1)

        # Anchors + image-jittered copies. Captions stay fixed.
        pairs = [(i, a) for i in range(n) for a in range(1 + n_aug)]
        for i, aug_idx in (tqdm(pairs, desc="bimodal-aug", leave=False) if verbose else pairs):
            img, text = self._images[i], self._texts[i]
            if aug_idx > 0:
                img = self._augment_image(img)
            embs.append(self._encode_dual(img, text, lang_scaler))
            img_labels.append(i)
            text_labels.append(i)

        # Cross-combinations: <img_i, text_j> for i != j
        cross = [(i, j) for i in range(n) for j in range(n) if i != j]
        for i, j in (tqdm(cross, desc="bimodal-cross", leave=False) if verbose else cross):
            emb = self._encode_dual(self._images[i], self._texts[j], lang_scaler)
            embs.append(emb)
            img_labels.append(i)
            text_labels.append(j)

        return torch.cat(embs, dim=0), img_labels, text_labels

    # ------------------------------------------------------------------
    # edge filter helper
    # ------------------------------------------------------------------

    def _get_edge_filter_tuple(self):
        if self.edge_filter == "none" or not self.edge_filter:
            return None
        elif self.edge_filter == "percentile":
            return ("percentile", self.edge_filter_kwargs.get("percentile", 0.25))
        elif self.edge_filter == "knn":
            return ("knn", self.edge_filter_kwargs.get("k", 10),
                    self.edge_filter_kwargs.get("mutual", True))
        elif self.edge_filter == "disparity":
            return ("disparity", self.edge_filter_kwargs.get("alpha", 0.05))
        raise ValueError(f"Unknown edge_filter: {self.edge_filter}")

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    @torch.no_grad()
    def search(
        self,
        images: List[Image.Image],
        texts: List[str],
        lang_scalers: Optional[List[float]] = None,
        shift_mode: str = "global",
        verbose: bool = True,
    ) -> Dict:
        """Sweep ``lang_scaler`` and compute modularity scores.

        Args:
            images: List of PIL images (at least ``n_samples``).
            texts: Corresponding list of text prompts/questions.
            lang_scalers: Scalers to try (default: log-spaced 0.1–100).
            shift_mode: ``"global"``, ``"langq_only"``, or ``"none"``.
            verbose: Print progress.

        Returns:
            Dict with ``"scalers"``, ``"baselines"``, ``"shift_mode"``.
        """
        if lang_scalers is None:
            raw = np.logspace(-1, 2, 41)
            rounded = np.where(
                raw < 1, np.round(raw, 2),
                np.where(raw < 10, np.round(raw, 1), np.round(raw, 0)),
            )
            lang_scalers = np.unique(rounded).tolist()

        # Sample data
        n = min(self.n_samples, len(images))
        idx = random.sample(range(len(images)), n)
        self._images = [images[i] for i in idx]
        self._texts = [texts[i] for i in idx]

        if verbose:
            print(f"[StandaloneAutoScaler] {n} samples → {n*n} pairs, "
                  f"image={self.image_encoder.name}, text={self.text_encoder.name}, "
                  f"scalers={lang_scalers}")

        edge_filter = self._get_edge_filter_tuple()

        # Baselines: vision-only and text-only
        baselines = {}
        for layer, enc_fn in [
            ("vision_layer", lambda img, txt: self._encode_image_only(img)),
            ("lang_layer", lambda img, txt: self._encode_text_only(txt)),
        ]:
            if verbose:
                print(f"  Computing {layer} baseline ...")
            embs = self._encode_all_pairs(enc_fn, verbose)
            baselines[layer] = self.compute_scores(embs, n, edge_filter)
            if verbose:
                b = baselines[layer]
                print(f"    {layer}: vis_Q={b['vision_Q']:.4f}, "
                      f"lang_Q={b['language_Q']:.4f}, H={b['harmonic']:.4f}")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Scaler sweep
        results: Dict[float, Dict] = {}
        for scaler in (tqdm(lang_scalers, desc="scalers") if verbose else lang_scalers):
            embs = self._encode_all_pairs(
                lambda img, txt, s=scaler: self._encode_dual(img, txt, s), verbose,
            )
            results[scaler] = self.compute_scores(embs, n, edge_filter)

            bimodal_embs, il, tl = self._encode_bimodal(scaler, verbose)
            for mode_name, mode in [("and", "and"), ("or", "or"), ("and_or", "and_or")]:
                target = self.build_bimodal_target(il, tl, mode).to(self.device)
                results[scaler][f"bimodal_{mode_name}_Q"] = self.compute_Q(
                    bimodal_embs, target, edge_filter,
                )

            if verbose:
                r = results[scaler]
                tqdm.write(
                    f"  scaler={scaler}: vis={r['vision_Q']:.3f}, "
                    f"lang={r['language_Q']:.3f}, "
                    f"bi_and={r['bimodal_and_Q']:.3f}, "
                    f"bi_or={r['bimodal_or_Q']:.3f}"
                )
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Shift Q values
        if results:
            all_vis = [r["vision_Q"] for r in results.values()]
            all_lang = [r["language_Q"] for r in results.values()]
            if shift_mode == "global":
                gmin = min(all_vis + all_lang)
                vshift, lshift = gmin, gmin
            elif shift_mode == "langq_only":
                vshift, lshift = 0.0, min(all_lang)
            else:
                vshift, lshift = 0.0, 0.0
            for scaler in results:
                r = results[scaler]
                r["vision_Q_shifted"] = r["vision_Q"] - vshift
                r["language_Q_shifted"] = r["language_Q"] - lshift
                v, l = r["vision_Q_shifted"], r["language_Q_shifted"]
                r["harmonic_shifted"] = 2 * v * l / (v + l) if (v > 0 and l > 0) else 0.0

        best = max(results, key=lambda s: results[s]["harmonic_shifted"])
        if verbose:
            print(f"[StandaloneAutoScaler] Best scaler={best} "
                  f"(H_shifted={results[best]['harmonic_shifted']:.4f})")

        return {"scalers": results, "baselines": baselines, "shift_mode": shift_mode}

    # ------------------------------------------------------------------
    # save / load / aggregate  (format-compatible with original AutoScaler)
    # ------------------------------------------------------------------

    def _get_model_tag(self) -> str:
        return f"{self.image_encoder.name}__{self.text_encoder.name}"

    def save_results(self, results: Dict, run_id: int = None, out_dir: str = None) -> str:
        out_dir = out_dir or f"results/auto_scaler_{self.n_samples}"
        tag = self._get_model_tag()
        os.makedirs(out_dir, exist_ok=True)
        suffix = f"_run{run_id}" if run_id is not None else ""
        out_path = os.path.join(out_dir, f"{tag}{suffix}.json")
        payload = {
            "model_tag": tag,
            "n_samples": self.n_samples,
            "run_id": run_id,
            "shift_mode": results.get("shift_mode", "global"),
            "scalers": results["scalers"],
            "baselines": results.get("baselines", {}),
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[StandaloneAutoScaler] Saved to {out_path}")
        return out_path

    def load_results(self, run_id: int = None, out_dir: str = None):
        out_dir = out_dir or f"results/auto_scaler_{self.n_samples}"
        tag = self._get_model_tag()
        suffix = f"_run{run_id}" if run_id is not None else ""
        in_path = os.path.join(out_dir, f"{tag}{suffix}.json")
        if not os.path.exists(in_path):
            return None
        with open(in_path) as f:
            data = json.load(f)
        scalers = {float(k): v for k, v in data["scalers"].items()}
        return {"scalers": scalers, "baselines": data.get("baselines", {}),
                "shift_mode": data.get("shift_mode", "global")}

    def load_results_k(self, out_dir: str = None):
        """Load all runs and aggregate into mean/std per scaler."""
        import glob as _glob

        out_dir = out_dir or f"results/auto_scaler_{self.n_samples}"
        pattern = os.path.join(out_dir, f"{self._get_model_tag()}_run*.json")
        files = sorted(_glob.glob(pattern))
        if not files:
            print(f"[StandaloneAutoScaler] No run files found: {pattern}")
            return None

        all_results = []
        for fp in files:
            with open(fp) as f:
                data = json.load(f)
                scalers = {float(k): v for k, v in data["scalers"].items()}
                all_results.append({"scalers": scalers, "baselines": data.get("baselines", {})})

        scaler_keys = list(all_results[0]["scalers"].keys())
        metrics = ["vision_Q", "language_Q", "harmonic", "harmonic_shifted",
                    "bimodal_and_Q", "bimodal_or_Q", "bimodal_and_or_Q"]
        agg_scalers = {}
        for s in scaler_keys:
            agg_scalers[s] = {}
            for m in metrics:
                vals = [r["scalers"][s][m] for r in all_results if s in r["scalers"] and m in r["scalers"][s]]
                if vals:
                    agg_scalers[s][m] = {"mean": np.mean(vals), "std": np.std(vals)}
        agg_baselines = {}
        for bl_key in ["vision_layer", "lang_layer"]:
            if bl_key in all_results[0].get("baselines", {}):
                agg_baselines[bl_key] = {}
                for m in ["vision_Q", "language_Q", "harmonic"]:
                    vals = [r["baselines"][bl_key][m] for r in all_results if bl_key in r["baselines"]]
                    if vals:
                        agg_baselines[bl_key][m] = {"mean": np.mean(vals), "std": np.std(vals)}
        print(f"[StandaloneAutoScaler] Aggregated {len(files)} runs")
        return {"scalers": agg_scalers, "baselines": agg_baselines}

    def get_best_from_agg(self, agg_results: Dict, metric: str = "harmonic_shifted") -> float:
        scalers = agg_results["scalers"]
        sample = next(iter(scalers.values()))
        if metric not in sample:
            metric = "harmonic"
        best = max(scalers, key=lambda s: scalers[s].get(metric, {}).get("mean", 0))
        m = scalers[best][metric]
        print(f"Best scaler (mean {metric}): {best} = {m['mean']:.4f} ± {m['std']:.4f}")
        return best

    # ------------------------------------------------------------------
    # plot  (mirrors original AutoScaler.plot signature)
    # ------------------------------------------------------------------

    def plot(
        self,
        results: Dict,
        figsize: Tuple[int, int] = (7, 6),
        metrics: List[str] = ("vision_Q", "language_Q", "harmonic", "bimodal_and_Q"),
        show: bool = True,
    ):
        """Plot Vision Q / Language Q tradeoff with optional error bars."""
        import matplotlib.pyplot as plt

        scalers = results["scalers"]
        baselines = results.get("baselines", {})
        x_values = sorted(scalers.keys())
        sample = scalers[x_values[0]]

        is_agg = isinstance(sample.get("vision_Q"), dict) and "mean" in sample["vision_Q"]
        has_shifted = "harmonic_shifted" in sample
        harm_key = "harmonic_shifted" if has_shifted else "harmonic"

        fig, ax = plt.subplots(figsize=figsize, dpi=300)

        metric_config = {
            "vision_Q": ("vision_Q", "green", "o", "Vision Q"),
            "language_Q": ("language_Q", "blue", "o", "Language Q"),
            "harmonic": (harm_key, "red", "s", "Harmonic"),
            "bimodal_and_Q": ("bimodal_and_Q", "orange", "^", "Bimodal Q"),
            "bimodal_or_Q": ("bimodal_or_Q", "purple", "v", "Bimodal OR"),
            "bimodal_and_or_Q": ("bimodal_and_or_Q", "brown", "d", "Bimodal AND/OR"),
        }

        plotted_data = {}
        for metric in metrics:
            if metric not in metric_config:
                continue
            key, color, marker, label = metric_config[metric]
            if key not in sample:
                continue
            if is_agg:
                vals = np.array([scalers[x][key]["mean"] for x in x_values])
                stds = np.array([scalers[x][key]["std"] for x in x_values])
                lw = 2 if metric == "harmonic" else 1.5
                ms = 6 if metric == "harmonic" else 5
                ax.errorbar(x_values, vals, yerr=stds, fmt=f'{marker}-', color=color,
                            label=label, ms=ms, lw=lw, capsize=3)
            else:
                vals = np.array([scalers[x][key] for x in x_values])
                lw = 2 if metric == "harmonic" else 1.5
                ms = 6 if metric == "harmonic" else 5
                ax.plot(x_values, vals, f'{marker}-', color=color, label=label, ms=ms, lw=lw)
            plotted_data[metric] = vals

        ax.set_xscale('log')
        ax.set_xlim(max(0.1, min(x_values)), max(x_values))

        if "harmonic" in plotted_data:
            harm_vals = plotted_data["harmonic"]
            best_idx = int(np.argmax(harm_vals))
            best_s = int(np.ceil(x_values[best_idx]))
            ax.scatter([x_values[best_idx]], [harm_vals[best_idx]], c='red', s=150, marker='*',
                       zorder=5, edgecolors='black', label=f'Best Harmonic ({best_s})')

        if baselines:
            xmin, xmax = x_values[0], x_values[-1]
            for bl_key, style, lbl in [("vision_layer", "--", "Image-only"),
                                        ("lang_layer", ":", "Text-only")]:
                if bl_key in baselines:
                    bl = baselines[bl_key]
                    v = bl["vision_Q"]["mean"] if is_agg else bl["vision_Q"]
                    l = bl["language_Q"]["mean"] if is_agg else bl["language_Q"]
                    ax.hlines(v, xmin, xmax, colors='green', linestyles=style, lw=1.5, alpha=0.5)
                    ax.hlines(l, xmin, xmax, colors='blue', linestyles=style, lw=1.5, alpha=0.5)
                    ax.plot([], [], style, color='gray', lw=1.5, label=lbl)

        ax.set_xlabel('lang_scaler', fontsize=12, fontweight='bold')
        ax.set_ylabel('Modularity Q (↑)', fontsize=12, fontweight='bold')
        ax.set_title(f'Standalone Dual ({self.image_encoder.name}) — Q Trade-off',
                     fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=9)
        plt.tight_layout()

        if show:
            plt.show()
            return None
        return fig, ax

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        self._images = None
        self._texts = None
