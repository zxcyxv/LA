"""학습된 두 모델의 기억 수명을 같은 자로 잰다 — 유효한 비교.

`diag_vs_mamba3.py` 는 **초기화** 비교라 무효였다 (CORRECTION_mamba3.md §4.4).
그 무효 사유 둘이 모두 해소된 뒤의 재측정이다:
  · mamba3_ref 의 A 우선순위 버그 수정 (감쇠가 살아났다)
  · 둘 다 같은 태스크·같은 조건으로 학습 (B/C bias 가 학습으로 움직였다)

측정은 diag_snr.py 와 동일한 재제시 회상이다:
    길이 T 의 토큰열에서 마지막 토큰을 위치 T−1−Δ 의 것과 같게 놓고
    신호 = 그 위치의 커널값, 잡음 = 나머지 위치의 rms, SNR = 신호/잡음.

    python diag_memory_trained.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diag_snr import last_row  # noqa: E402
from mamba3_ref import Mamba3Config, Mamba3Model, heavy_tail  # noqa: E402
from model1 import Model1  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OURS = "ckpt_ours_ordinal_64.pt"
M3_L4 = "ckpt_mamba3_64.pt"
M3_L1 = "ckpt_mamba3_L1.pt"
LAGS = [1, 2, 3, 5, 8, 12, 20, 32, 48, 64, 96, 128, 192, 256, 384, 512]


def root(p):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), p)


# ------------------------------------------------------------------ Mamba-3 커널
@torch.no_grad()
def m3_last_row(layer, u):
    """w[:, :, T−1, :] — use_quadratic 경로와 같은 식, t=T−1 고정. (B,H,T)"""
    cfg = layer.cfg
    parts = layer.in_proj(u).split(
        [layer.d_inner, layer.d_inner, cfg.d_state, cfg.d_state,
         layer.nheads, layer.nheads, layer.nheads, layer.n_ang], dim=-1)
    _z, _x, Bp, Cp, dd_dt, dd_A, _trap, ang = parts
    if cfg.a_precedence_bug:
        _A = -heavy_tail(dd_A.float()).clamp(max=-cfg.A_floor)
    else:
        _A = (-heavy_tail(dd_A.float())).clamp(max=-cfg.A_floor)
    DT = F.softplus(dd_dt + layer.dt_bias)
    ADT = _A * DT
    inc = torch.tanh(ang) * DT.mean(-1, keepdim=True) * math.pi
    phi = torch.remainder(torch.cumsum(inc.double(), 1), 2 * math.pi).float()
    Bv = layer._rope(layer.B_norm(Bp) + layer.B_bias, phi)
    Cv = layer._rope(layer.C_norm(Cp) + layer.C_bias, phi)
    cumA = torch.cumsum(ADT, dim=1)
    t = u.shape[1] - 1
    dec = torch.exp(cumA[:, t, :].unsqueeze(-1) - cumA.transpose(1, 2))
    score = torch.einsum("bn,bsn->bs", Cv[:, t], Bv)
    w = dec * score.unsqueeze(1) * DT.transpose(1, 2)
    causal = torch.arange(u.shape[1], device=u.device) <= t
    return w * causal[None, None, :]


def snr_from_row(row, T, D, head_dim=False):
    """row: (B,T) 또는 (B,H,T) → (신호, 잡음, SNR) 평균."""
    if head_dim:
        sig = row[:, :, T - 1 - D].abs()
        mask = torch.ones(T, dtype=torch.bool, device=row.device)
        mask[T - 1] = False
        mask[T - 1 - D] = False
        noise = row[:, :, mask].pow(2).mean(-1).sqrt()
    else:
        sig = row[:, T - 1 - D].abs()
        mask = torch.ones(T, dtype=torch.bool, device=row.device)
        mask[T - 1] = False
        mask[T - 1 - D] = False
        noise = row[:, mask].pow(2).mean(-1).sqrt()
    return sig.mean().item(), noise.mean().item(), (sig / noise).mean().item()


@torch.no_grad()
def curve_ours(ck, lags, T, B=32, seed=0):
    m = Model1(ck["mcfg"]).to(DEV).eval()
    m.load_state_dict(ck["state"])
    V = ck["task"].vocab_size
    g = torch.Generator().manual_seed(seed)
    out = []
    for D in lags:
        if D >= T - 1:
            continue
        x = torch.randint(0, V, (B, T), generator=g).to(DEV)
        x[:, T - 1] = x[:, T - 1 - D]
        out.append((D, *snr_from_row(last_row(m, m.embed(x)), T, D)))
    return np.array(out)


@torch.no_grad()
def curve_m3(ck, lags, T, B=32, seed=0, layer_idx=0):
    m = Mamba3Model(ck["cfg"]).to(DEV).eval()
    m.load_state_dict(ck["state"])
    V = ck["cfg"].vocab_size
    g = torch.Generator().manual_seed(seed)
    n1, mix, _, _ = m.layers[layer_idx]
    out = []
    for D in lags:
        if D >= T - 1:
            continue
        x = torch.randint(0, V, (B, T), generator=g).to(DEV)
        x[:, T - 1] = x[:, T - 1 - D]
        h = m.embed(x)
        for li in range(layer_idx):          # 앞선 층을 통과시킨다
            a, mix0, b, mlp0 = m.layers[li]
            h = h + mix0(a(h))
            h = h + mlp0(b(h))
        out.append((D, *snr_from_row(m3_last_row(mix, n1(h)), T, D, head_dim=True)))
    return np.array(out)


def life(r):
    ok = r[r[:, 3] >= 1.0]
    return int(ok[-1, 0]) if len(ok) else 0


def main():
    for p in (OURS, M3_L4, M3_L1):
        if not os.path.exists(root(p)):
            print(f"체크포인트 없음: {p} — 먼저 학습해야 한다")
            return
    T = 600
    ours = torch.load(root(OURS), map_location=DEV, weights_only=False)
    m3l4 = torch.load(root(M3_L4), map_location=DEV, weights_only=False)
    m3l1 = torch.load(root(M3_L1), map_location=DEV, weights_only=False)

    print("=" * 88)
    print("학습된 모델의 기억 수명 — 재제시 회상 SNR")
    print("=" * 88)
    print(f"  둘 다 selective copy 로 학습. 측정은 무작위 토큰열(V=18), T={T}.")
    print(f"  우리: model1 R=4 ordinal_only, l_noise=64, 27k 파라미터")
    print(f"  Mamba-3: 수정본, l_noise=64(L=4) / 32(L=1)")
    print()
    cs = {
        "우리 (27k, ordinal)": curve_ours(ours, LAGS, T),
        "Mamba-3 L=4 (637k)": curve_m3(m3l4, LAGS, T),
        "Mamba-3 L=1 (161k)": curve_m3(m3l1, LAGS, T),
    }
    ks = list(cs)
    print(f"  {'Δ':>5s}" + "".join(f"{k:>22s}" for k in ks))
    print("-" * 88)
    n = min(len(v) for v in cs.values())
    for i in range(n):
        D = int(cs[ks[0]][i, 0])
        print(f"  {D:5d}" + "".join(f"{cs[k][i,3]:22.3f}" for k in ks))
    print()
    for k in ks:
        r = cs[k]
        print(f"  {k:24s} 수명(SNR≥1) = {life(r):5d} 토큰   SNR@Δ=1 = {r[0,3]:8.2f}")
    print()
    print("  주의: Mamba-3 는 헤드별 커널이라 헤드 평균이다. 우리는 채널을 하나의")
    print("        스칼라로 합친 값이다 (a_tn). 층 0 의 커널만 본다.")
    print("        그리고 이것은 **주소지정 정밀도**이지 회상 충실도가 아니다.")


if __name__ == "__main__":
    main()
