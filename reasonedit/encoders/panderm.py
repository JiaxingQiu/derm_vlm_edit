"""PanDerm Large image encoder.

ViT-L/16 from the Nature Medicine 2025 PanDerm paper. The checkpoint is the
linear-probe encoder, and the embedding is the CLS token, 1024-d. Weights are
the public file ``panderm_ll_data6_checkpoint-499.pth``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .base import ImageEncoder

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
CHECKPOINT_NAME = "panderm_ll_data6_checkpoint-499.pth"
DRIVE_ID = "1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.228, 0.224, 0.225)


def _trunc_normal_(tensor, std=0.02):
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, std=std)
    return tensor


class _PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.patch_shape = (img_size // patch_size, img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        height, width = x.shape[-2:]
        if (height, width) != self.img_size:
            raise ValueError(f"PanDerm expects {self.img_size}, got {(height, width)}")
        return self.proj(x).flatten(2).transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, length, _ = x.shape
        bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias)
        qkv = qkv.reshape(batch, length, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = (query * self.scale) @ key.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = (attn @ value).transpose(1, 2).reshape(batch, length, -1)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, init_values=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, hidden)
        self.mlp.act = nn.GELU()
        self.mlp.fc2 = nn.Linear(hidden, dim)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        x = x + self.gamma_1 * self.attn(self.norm1(x))
        hidden = self.mlp.act(self.mlp.fc1(self.norm2(x)))
        x = x + self.gamma_2 * self.mlp.fc2(hidden)
        return x


class _PanDermViT(nn.Module):
    """PanDerm Large at 224. ``forward_features`` returns every token."""

    def __init__(self):
        super().__init__()
        embed_dim, depth, num_heads = 1024, 24, 16
        self.embed_dim = embed_dim
        self.patch_embed = _PatchEmbed(embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(self._sincos(embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList(_Block(embed_dim, num_heads) for _ in range(depth))
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        _trunc_normal_(self.cls_token, std=0.02)

    def _sincos(self, embed_dim, temperature=10000.0):
        height, width = self.patch_embed.patch_shape
        grid_w = torch.arange(width, dtype=torch.float32)
        grid_h = torch.arange(height, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = 1.0 / (temperature ** (torch.arange(pos_dim, dtype=torch.float32) / pos_dim))
        out_w = torch.einsum("m,d->md", grid_w.flatten(), omega)
        out_h = torch.einsum("m,d->md", grid_h.flatten(), omega)
        pos = torch.cat((out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()), dim=1)[None]
        return torch.cat((torch.zeros(1, 1, embed_dim), pos), dim=1)

    def forward_features(self, x):
        x = self.patch_embed(x)
        batch = x.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed.expand(batch, -1, -1).to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def _checkpoint_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    import os
    env = os.environ.get("PANDERM_CHECKPOINT")
    if env:
        return Path(env)
    return WEIGHTS_DIR / CHECKPOINT_NAME


def _download(dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[PanDermEncoder] downloading {CHECKPOINT_NAME}")
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "PanDerm weights are not on disk and gdown is not installed. "
            f"Save {DRIVE_ID} from Google Drive to {dest}"
        ) from exc
    gdown.download(id=DRIVE_ID, output=str(dest), quiet=False)
    if not dest.exists() or dest.stat().st_size < 1_000_000:
        raise RuntimeError(f"PanDerm download failed: {dest}")


def _load_state(path: Path) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and "state_dict" in blob:
        blob = blob["state_dict"]
    elif isinstance(blob, dict) and "model" in blob and isinstance(blob["model"], dict):
        blob = blob["model"]
    state = {}
    for key, value in blob.items():
        name = key
        for prefix in ("module.", "encoder."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        state[name] = value
    return state


class PanDermEncoder(ImageEncoder):
    """Frozen PanDerm Large. ``pool='cls'`` is the paper's linear-probe token."""

    def __init__(self, device: Union[str, torch.device] = "cpu", pool: str = "cls", checkpoint: str = None):
        if pool not in ("cls", "mean"):
            raise ValueError(f"pool must be cls or mean, got {pool!r}")
        self._device = torch.device(device) if isinstance(device, str) else device
        self._pool = pool
        self._checkpoint = checkpoint
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        path = _checkpoint_path(self._checkpoint)
        if not path.exists():
            _download(path)
        print(f"[PanDermEncoder] Loading {path}")
        model = _PanDermViT()
        missing, _unexpected = model.load_state_dict(_load_state(path), strict=False)
        missing = [key for key in missing if not key.startswith("head.")]
        if missing:
            shown = ", ".join(missing[:8])
            raise RuntimeError(f"PanDerm checkpoint is missing {len(missing)} tensors: {shown}")
        model.eval()
        self._model = model.to(self._device)
        self._dim = model.embed_dim
        print(f"[PanDermEncoder] Loaded ({self._dim}-d, pool={self._pool})")

    def _preprocess(self, images: List[Image.Image]) -> torch.Tensor:
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        return torch.stack([transform(image.convert("RGB")) for image in images]).to(self._device)

    @torch.no_grad()
    def encode(self, images: Union[List[Image.Image], Image.Image]) -> torch.Tensor:
        self._load()
        pixels = self._preprocess(self._ensure_list(images))
        tokens = self._model.forward_features(pixels)
        if self._pool == "cls":
            emb = tokens[:, 0, :]
        else:
            emb = tokens[:, 1:, :].mean(dim=1)
        return emb.to(torch.float32)

    @property
    def dim(self) -> int:
        self._load()
        return self._dim

    @property
    def name(self) -> str:
        return f"panderm_{self._pool}"
