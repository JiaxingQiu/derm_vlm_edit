"""ModularityCore: Base class for Vision Q and Language Q computation.

Provides target matrices, modularity Q, and edge filtering (NetBone).
"""

import torch


class NetBone:
    """Network backbone edge filters for similarity matrices.

    Usage:
        sim_filtered = NetBone.percentile(sim, percentile=0.25)
        sim_filtered = NetBone.knn(sim, k=25, mutual=True)
        sim_filtered = NetBone.disparity(sim, alpha=0.05, pre_topk=50)
    """

    @staticmethod
    def percentile(sim, percentile=0.25):
        """Keep edges with similarity >= percentile threshold."""
        if percentile <= 0:
            return sim
        N = sim.shape[0]
        triu_idx = torch.triu_indices(N, N, offset=1, device=sim.device)
        thresh_val = torch.quantile(sim[triu_idx[0], triu_idx[1]], percentile)
        mask = sim >= thresh_val
        mask.fill_diagonal_(True)
        return sim * mask.float()

    @staticmethod
    def knn(sim, k=10, mutual=True):
        """Keep only top-k neighbors per node."""
        N = sim.shape[0]
        k = min(k, N - 1)
        sim_no_diag = sim.clone()
        sim_no_diag.fill_diagonal_(-float('inf'))
        _, topk_idx = sim_no_diag.topk(k, dim=1)
        knn_mask = torch.zeros_like(sim, dtype=torch.bool)
        rows = torch.arange(N, device=sim.device).unsqueeze(1).expand(-1, k)
        knn_mask[rows, topk_idx] = True
        if mutual:
            mask = knn_mask & knn_mask.t()
        else:
            mask = knn_mask | knn_mask.t()
        mask.fill_diagonal_(True)
        return sim * mask.float()

    @staticmethod
    def disparity(sim, alpha=0.05, pre_topk=50):
        """Disparity filter (Serrano-Boguna-Vespignani backbone)."""
        N = sim.shape[0]
        sim = sim.clone()
        sim.fill_diagonal_(0)
        if pre_topk is not None and pre_topk < N - 1:
            sim = NetBone.knn(sim, k=pre_topk, mutual=False)
            sim.fill_diagonal_(0)
        strength = sim.sum(dim=1, keepdim=True) + 1e-10
        degree = (sim > 0).sum(dim=1).float()
        p = sim / strength
        k = degree.unsqueeze(1).expand(-1, N)
        pval = (1 - p + 1e-10).pow(k - 1)
        sig_mask = (pval < alpha) | (pval.t() < alpha)
        sig_mask.fill_diagonal_(True)
        return sim * sig_mask.float()


class ModularityCore:
    """Base class for modularity-based layer/scaler selection."""

    def __init__(self, device=None):
        self.device = device or torch.device("cpu")

    # ==================== Target Matrices ====================

    @staticmethod
    def build_vision_target(n):
        """Vision Q target: cluster by image.

        For n samples with n*n pairs ordered as:
        <i1,t1>, <i1,t2>, ..., <i1,tn>, <i2,t1>, ..., <in,tn>
        """
        N = n * n
        target = torch.zeros(N, N)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    target[i * n + j, i * n + k] = 1.0
        return target

    @staticmethod
    def build_language_target(n):
        """Language Q target: cluster by text."""
        N = n * n
        target = torch.zeros(N, N)
        for j in range(n):
            for i in range(n):
                for k in range(n):
                    target[i * n + j, k * n + j] = 1.0
        return target

    # ==================== Q Computation ====================

    def compute_similarity(self, embs, debug=False):
        """Convert embeddings to similarity matrix via L2 distance."""
        embs = embs.to(self.device)
        l2_dist = torch.cdist(embs, embs, p=2)
        l2_max, l2_min = l2_dist.max(), l2_dist.min()
        denom = l2_max - l2_min
        if debug:
            print(f"  [DEBUG] L2 dist range: [{l2_min:.4f}, {l2_max:.4f}], denom={denom:.4f}")
        if denom < 1e-8:
            if debug:
                print(f"  [DEBUG] WARNING: Degenerate embeddings (all identical)")
            return torch.ones_like(l2_dist)
        return (l2_max - l2_dist) / denom

    def compute_Q(self, embs, target, edge_filter=None):
        """Compute modularity Q score (higher = better clustering)."""
        embs = embs.to(self.device)
        target = target.to(self.device)
        target = target.clone()
        target.fill_diagonal_(0.0)
        sim = self.compute_similarity(embs)
        if edge_filter is not None:
            method = edge_filter[0] if isinstance(edge_filter, tuple) else edge_filter
            if method == "percentile":
                percentile = edge_filter[1] if len(edge_filter) > 1 else 0.25
                sim = NetBone.percentile(sim, percentile)
            elif method == "knn":
                k = edge_filter[1] if len(edge_filter) > 1 else 10
                mutual = edge_filter[2] if len(edge_filter) > 2 else True
                sim = NetBone.knn(sim, k, mutual)
            elif method == "disparity":
                alpha = edge_filter[1] if len(edge_filter) > 1 else 0.05
                sim = NetBone.disparity(sim, alpha)
        sim_no_diag = sim.clone()
        sim_no_diag.fill_diagonal_(0.0)
        m = sim_no_diag.sum() / 2
        if m <= 0:
            return 0.0
        k = sim_no_diag.sum(dim=1)
        Q = ((sim_no_diag - torch.outer(k, k) / (2 * m)) * target).sum() / (2 * m)
        return float(Q)

    def compute_scores(self, embs, n, edge_filter=None):
        """Compute Vision Q, Language Q, and Harmonic mean."""
        vision_target = self.build_vision_target(n).to(self.device)
        language_target = self.build_language_target(n).to(self.device)
        vis_Q = self.compute_Q(embs, vision_target, edge_filter)
        lang_Q = self.compute_Q(embs, language_target, edge_filter)
        if vis_Q > 0 and lang_Q > 0:
            harmonic = 2 * vis_Q * lang_Q / (vis_Q + lang_Q)
        else:
            harmonic = 0.0
        return {"vision_Q": vis_Q, "language_Q": lang_Q, "harmonic": harmonic}

    @staticmethod
    def build_bimodal_target(img_labels, text_labels, mode="and_or"):
        """Build target matrix for bi-modality Q."""
        img_labels = torch.tensor(img_labels)
        text_labels = torch.tensor(text_labels)
        same_img = (img_labels.unsqueeze(0) == img_labels.unsqueeze(1))
        same_text = (text_labels.unsqueeze(0) == text_labels.unsqueeze(1))
        if mode == "and":
            target = (same_img & same_text).float()
        elif mode == "or":
            target = (same_img | same_text).float()
        elif mode == "and_or":
            target = same_img.float() + same_text.float()
        else:
            raise ValueError(f"Unknown bimodal_mode: {mode}")
        return target
