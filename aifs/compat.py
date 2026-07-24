"""
Import this module *before* importing anything from ``anemoi`` to enable
compatibility with the available GPU / CPU backend.

.. warning::
   This module installs a ``flash_attn`` compatibility shim that routes attention
   through :func:`torch.nn.functional.scaled_dot_product_attention` (SDPA). The
   AIFS checkpoints published by ECMWF were trained with the real
   `flash-attn <https://github.com/Dao-AILab/flash-attention>`_ CUDA kernels.
   The SDPA shim reproduces flash-attn's sliding-window semantics as faithfully
   as possible, but small numerical differences are expected — particularly on
   CPU and MPS backends, where reduction order, precision (fp32 vs fp16/bf16)
   and vendor library implementations differ from CUDA + flash-attn. Forecasts
   produced through this shim should therefore be treated as *approximations*
   of the reference AIFS output and must not be used for operational,
   life-safety or safety-of-life decision-making.

   The ``flash_attn.layers.rotary.RotaryEmbedding`` code path is **not**
   provided by this shim; models that exercise it (e.g. some AIFS-ENS
   configurations) will raise :class:`NotImplementedError` at runtime.
"""

import logging
import sys
import time
import types
import warnings

import torch
import torch.nn.functional as F

LOG = logging.getLogger(__name__)

# Emit the numerical-divergence caveat once, at import time.
warnings.warn(
    "aifs.compat: flash-attn is being replaced by a torch SDPA shim. "
    "Outputs may differ numerically from the reference CUDA/flash-attn "
    "implementation used to train AIFS. Do not use these forecasts for "
    "operational or safety-critical purposes.",
    RuntimeWarning,
    stacklevel=2,
)

# ── SDPA-based attention replacement ─────────────────────────────────────────


def _sdpa_compat(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """
    Drop-in replacement for ``flash_attn_func``.

    Signature mirrors flash-attn 2.x. Input tensors are shaped (batch, seq, heads, dim).
    """
    if softcap not in (None, 0.0):
        raise NotImplementedError(
            "softcap is not supported by the SDPA compatibility shim"
        )
    if alibi_slopes is not None:
        raise NotImplementedError(
            "alibi_slopes is not supported by the SDPA compatibility shim"
        )
    if return_attn_probs:
        raise NotImplementedError(
            "return_attn_probs is not supported by the SDPA compatibility shim"
        )

    t0 = time.perf_counter()

    # flash-attn layout: (B, S, H, D)  →  SDPA layout: (B, H, S, D)
    q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))

    left, right = window_size

    S = q.shape[-2]

    if q.device.type == "cuda":
        out = _chunked_windowed_attention(
            q, k, v, left, right, causal, dropout_p, softmax_scale, chunk_size=2048
        )

    elif q.device.type == "mps":
        # MPS: chunked to avoid OOM on large sequences.
        chunk = max(left if left > 0 else 0, right if right > 0 else 0) or 512
        out = _chunked_windowed_attention(
            q, k, v, left, right, causal, dropout_p, softmax_scale, chunk_size=chunk
        )

    else:
        # CPU fallback — move to CPU in case tensors are on an unsupported device.
        q_cpu, k_cpu, v_cpu = q.cpu(), k.cpu(), v.cpu()
        out = _chunked_windowed_attention(
            q_cpu,
            k_cpu,
            v_cpu,
            left,
            right,
            causal,
            dropout_p,
            softmax_scale,
            chunk_size=1024,
        )
        out = out.to(q.device)

    if q.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    LOG.debug(
        f"  [compat] attn {elapsed:.3f}s  device={q.device.type}  S={S}  window=({left},{right})  causal={causal}"
    )

    return out.permute(0, 2, 1, 3)


def _window_bounds(seq_len, left, right, device):
    """
    Build the per-query (lower, upper) inclusive key bounds implied by
    ``window_size=(left, right)`` following flash-attn semantics:
    """
    idx = torch.arange(seq_len, device=device)
    lower = torch.clamp(idx - left, min=0)
    upper = torch.clamp(idx + right, max=seq_len - 1)
    return lower, upper


def _chunked_windowed_attention(
    q, k, v, left, right, causal, dropout_p, softmax_scale, chunk_size
):
    """
    Memory-friendly windowed/causal attention: iterate over query chunks and
    only materialize the (small) slice of keys/values each chunk can attend
    to, with a per-row mask inside that slice to get exact bounds right.
    """
    B, H, S, D = q.shape
    out = torch.empty_like(q)
    lower_full, upper_full = _window_bounds(S, left, right, q.device)

    for qs in range(0, S, chunk_size):
        qe = min(S, qs + chunk_size)

        # Superset of keys any query in [qs, qe) could need.
        k_start = int(lower_full[qs:qe].min().item())
        k_end = int(upper_full[qs:qe].max().item()) + 1

        q_chunk = q[:, :, qs:qe]
        k_chunk = k[:, :, k_start:k_end]
        v_chunk = v[:, :, k_start:k_end]

        # Per-row mask within this (small) chunk to enforce exact bounds.
        key_idx = torch.arange(k_start, k_end, device=q.device).view(1, -1)
        lower_c = lower_full[qs:qe].view(-1, 1)
        upper_c = upper_full[qs:qe].view(-1, 1)
        allowed = (key_idx >= lower_c) & (key_idx <= upper_c)

        out[:, :, qs:qe] = F.scaled_dot_product_attention(
            q_chunk,
            k_chunk,
            v_chunk,
            attn_mask=allowed,
            dropout_p=dropout_p,
            scale=softmax_scale,
        )

    return out


# ── Build modules ────────────────────────────────────────────────────────


def _patch():
    """Install the flash_attn stub into ``sys.modules``."""
    if "flash_attn" in sys.modules:
        return

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__version__ = "2.6.0"  # version Anemoi checks against
    flash_attn.flash_attn_func = _sdpa_compat

    # flash_attn.layers.rotary  (imported but only used on specific GPU paths)
    layers_mod = types.ModuleType("flash_attn.layers")
    rotary_mod = types.ModuleType("flash_attn.layers.rotary")

    def _rotary_not_implemented(*args, **kwargs):
        raise NotImplementedError(
            "flash_attn.layers.rotary.RotaryEmbedding is not available in the SDPA "
            "compatibility shim; this code path requires real flash-attn on CUDA."
        )

    rotary_mod.RotaryEmbedding = _rotary_not_implemented
    layers_mod.rotary = rotary_mod
    flash_attn.layers = layers_mod

    # flash_attn.flash_attn_interface
    interface_mod = types.ModuleType("flash_attn.flash_attn_interface")
    interface_mod.flash_attn_func = _sdpa_compat
    flash_attn.flash_attn_interface = interface_mod

    sys.modules["flash_attn"] = flash_attn
    sys.modules["flash_attn.layers"] = layers_mod
    sys.modules["flash_attn.layers.rotary"] = rotary_mod
    sys.modules["flash_attn.flash_attn_interface"] = interface_mod


_patch()
