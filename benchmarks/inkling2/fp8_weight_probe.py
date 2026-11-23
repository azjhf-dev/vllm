# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure FP8 W8A16 quantization error on the real Inkling 2 Mini weights.

Downloads only the requested tensor's byte range from the HF safetensors
file (no full 19.5 GB checkpoint pull) and reports per-row and global
quantization error for the scheme in ``nvidia/ops/fp8_w8a16.py``
(float8_e4m3fn, per-output-row absmax scale).

Usage:
  python benchmarks/inkling2/fp8_weight_probe.py \
      --tensor model.llm.embed.weight
  python benchmarks/inkling2/fp8_weight_probe.py \
      --tensor model.llm.unembed.weight --logit-sim 256
"""

from __future__ import annotations

import argparse
import json
import struct
import tempfile
import urllib.request

import numpy as np
import torch

URL = (
    "https://huggingface.co/thinkingmachines/Inkling-2-Mini"
    "/resolve/main/model-00001-of-00001.safetensors"
)
FP8_MAX = 448.0


def fetch_range(lo: int, hi: int) -> bytes:
    req = urllib.request.Request(URL, headers={"Range": f"bytes={lo}-{hi}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def load_tensor(
    name: str, from_file: str | None = None
) -> tuple[np.memmap, tuple[int, int]]:
    """Returns a zero-copy uint16 memmap over the raw tensor bytes and the
    shape. With --from-file, reads a previously saved byte-range dump
    instead of downloading the tensor."""
    hdr_len = struct.unpack("<Q", fetch_range(0, 7))[0]
    hdr = json.loads(fetch_range(8, 8 + hdr_len - 1))
    t = hdr[name]
    assert t["dtype"] == "BF16", t
    shape = tuple(t["shape"])
    b0, b1 = t["data_offsets"]
    if from_file is not None:
        path = from_file
    else:
        raw = fetch_range(8 + hdr_len + b0, 8 + hdr_len + b1 - 1)
        fh = tempfile.NamedTemporaryFile(prefix="inkling2_probe_", delete=False)
        fh.write(raw)
        fh.close()
        path = fh.name
        del raw
    u16 = np.memmap(path, dtype=np.uint16, mode="r")
    assert u16.size * 2 == b1 - b0
    return u16, shape


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tensor", default="model.llm.embed.weight")
    p.add_argument(
        "--logit-sim",
        type=int,
        default=64,
        help="random unit-norm hidden vectors for a logit/row dot-product "
        "error simulation (0 to skip)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--from-file",
        default=None,
        help="read raw tensor bytes from a local file (a previously saved "
        "byte-range dump) instead of downloading the tensor",
    )
    args = p.parse_args()

    u16, (n, k) = load_tensor(args.tensor, args.from_file)
    print(f"{args.tensor}: {n}x{k} bf16")

    chunk = 8192
    row_rel = torch.empty(n)
    err_fro2 = 0.0
    w_fro2 = 0.0
    g = torch.Generator().manual_seed(args.seed)
    H = (
        torch.randn(args.logit_sim, k, generator=g)
        if args.logit_sim
        else torch.empty(0, k)
    )
    if args.logit_sim:
        H = torch.nn.functional.normalize(H, dim=1)
    dot_num = torch.zeros(args.logit_sim)
    dot_den = torch.zeros(args.logit_sim)

    for i in range(0, n, chunk):
        # bf16 -> fp32 without extra deps: top 16 bits of the fp32 word.
        slc = np.ascontiguousarray(u16[i * k : (i + chunk) * k])
        wc = torch.from_numpy((slc.astype(np.uint32) << 16).view(np.float32)).reshape(-1, k)
        scale = wc.abs().amax(dim=1).clamp(min=1e-12) / FP8_MAX
        q = (wc / scale[:, None]).to(torch.float8_e4m3fn)
        w_hat = q.to(torch.float32) * scale[:, None]
        err = w_hat - wc
        row_rel[i : i + chunk] = err.norm(dim=1) / wc.norm(dim=1).clamp(min=1e-12)
        err_fro2 += err.pow(2).sum().item()
        w_fro2 += wc.pow(2).sum().item()
        if args.logit_sim:
            dot_num += (H @ err.t()).pow(2).sum(dim=1)
            dot_den += (H @ wc.t()).pow(2).sum(dim=1)

    qs = torch.quantile(row_rel, torch.tensor([0.5, 0.9, 0.99]).float())
    print(f"global relative Frobenius error: {(err_fro2 / w_fro2) ** 0.5:.5f}")
    print(
        f"per-row relative error: median {qs[0]:.5f}, p90 {qs[1]:.5f}, "
        f"p99 {qs[2]:.5f}, max {row_rel.max():.5f}"
    )
    print(f"worst rows: {row_rel.argsort(descending=True)[:5].tolist()}")
    if args.logit_sim:
        print(
            f"dot-product sim ({args.logit_sim} unit-norm rows of R^{k}): "
            f"relative logit error {(dot_num.sum() / dot_den.sum()) ** 0.5:.5f}"
        )


if __name__ == "__main__":
    main()
