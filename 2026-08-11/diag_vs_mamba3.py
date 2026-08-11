"""우리 모델과 Mamba-3 의 기억 수명을 **같은 자로** 잰다.

`diag_snr.py` 의 재제시 회상을 두 아키텍처에 똑같이 적용한다.
둘 다 선형 재귀이므로 입력-출력 사상이 커널 `a_tn` 으로 쓰인다.

    우리     : a_tn = p^{-1/2} Σ_j γ_j (q̄_t k_n)_j e^{-α_j Δ}          (attention_matrix)
    Mamba-3  : w_htn = exp(cumA_t − cumA_n) · (C_t·B_n) · DT_n         (use_quadratic 경로)

측정은 동일하다. 마지막 토큰을 위치 `T−1−Δ` 의 것과 같게 놓고
    신호 = 그 위치의 커널값,  잡음 = 나머지 위치의 rms,  SNR = 신호/잡음.
`SNR ≥ 1` 인 최대 Δ 가 유효 기억 수명이다.

**공정성에 대한 경고는 §M3 을 읽을 것.** 요약: Mamba-3 의 `dt` 는 학습되고
우리 `τ` 는 학습되지 않으므로(CAPACITY §4.2), 초기화 비교는 Mamba-3 에게 불리하다.
따라서 **Mamba-3 가 초기화에서 이기면 결정적이고, 우리가 이기면 결정적이지 않다.**

    python diag_vs_mamba3.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diag_snr import BASE, LAGS, snr_curve  # noqa: E402
from mamba3_ref import Mamba3Config, Mamba3Layer  # noqa: E402

torch.manual_seed(0)


# ------------------------------------------------------------------ Mamba-3 커널
@torch.no_grad()
def mamba3_last_row(layer: Mamba3Layer, u: torch.Tensor) -> torch.Tensor:
    """w[:, :, T−1, :] — 마지막 질의 위치의 커널. (B, H, T)

    forward 의 use_quadratic 분기와 같은 식이되 t = T−1 로 고정한다.
    """
    import torch.nn.functional as F

    from mamba3_ref import heavy_tail

    cfg = layer.cfg
    parts = layer.in_proj(u).split(
        [layer.d_inner, layer.d_inner, cfg.d_state, cfg.d_state,
         layer.nheads, layer.nheads, layer.nheads, layer.n_ang], dim=-1)
    _z, _x, Bp, Cp, dd_dt, dd_A, _trap, ang = parts

    _A = -heavy_tail(dd_A.float()).clamp(max=-cfg.A_floor)
    DT = F.softplus(dd_dt + layer.dt_bias)                       # (B,T,H)
    ADT = _A * DT
    inc = torch.tanh(ang) * DT.mean(-1, keepdim=True) * math.pi
    phi = torch.remainder(torch.cumsum(inc.double(), 1), 2 * math.pi).float()

    Bv = layer._rope(layer.B_norm(Bp) + layer.B_bias, phi)       # key
    Cv = layer._rope(layer.C_norm(Cp) + layer.C_bias, phi)       # query

    cumA = torch.cumsum(ADT, dim=1)                              # (B,T,H)
    t = u.shape[1] - 1
    dec = torch.exp(cumA[:, t, :].unsqueeze(-1)                  # (B,H,1)
                    - cumA.transpose(1, 2))                      # (B,H,T)
    score = torch.einsum("bn,bsn->bs", Cv[:, t], Bv)             # (B,T)
    w = dec * score.unsqueeze(1) * DT.transpose(1, 2)            # (B,H,T)
    causal = torch.arange(u.shape[1], device=u.device) <= t
    return w * causal[None, None, :]


@torch.no_grad()
def mamba3_snr(lags, d_state=32, headdim=32, d_model=128, T=1600, B=12, V=512, seed=0):
    torch.manual_seed(seed)
    cfg = Mamba3Config(d_model=d_model, d_state=d_state, headdim=headdim,
                       vocab_size=V, use_quadratic=True)
    layer = Mamba3Layer(cfg).eval()
    emb = torch.nn.Embedding(V, d_model)
    torch.nn.init.normal_(emb.weight, std=1.0)                   # 우리와 같은 척도
    g = torch.Generator().manual_seed(seed + 1)
    out = []
    for D in lags:
        x = torch.randint(0, V, (B, T), generator=g)
        x[:, T - 1] = x[:, T - 1 - D]
        w = mamba3_last_row(layer, emb(x))                       # (B,H,T)
        sig = w[:, :, T - 1 - D].abs()                           # (B,H)
        mask = torch.ones(T, dtype=torch.bool)
        mask[T - 1] = False
        mask[T - 1 - D] = False
        noise = w[:, :, mask].pow(2).mean(-1).sqrt()             # (B,H)
        snr = (sig / noise)                                      # (B,H)
        out.append((D, sig.mean().item(), noise.mean().item(),
                    snr.mean().item(), snr.mean(0).max().item()))
    return np.array(out)


def life(r, col=3):
    ok = r[r[:, col] >= 1.0]
    return int(ok[-1, 0]) if len(ok) else 0


# ================================================================== M1
print("=" * 92)
print("[M1] 재귀 상태의 크기 — 무엇을 비교하고 있는가")
print("=" * 92)
ours = 64 * 128 * 2
for ds, hd, dm, nl in ((32, 32, 128, 1), (32, 32, 128, 4), (64, 32, 128, 1),
                       (128, 32, 128, 1)):
    nh = dm // hd
    st = nh * hd * ds * nl
    print(f"  Mamba-3  d_state={ds:3d} headdim={hd} d_model={dm} n_layer={nl}"
          f"  → 상태 {st:7,d} 실수")
print(f"  우리      p=64 d=128 (S ∈ ℂ^(64×128))            → 상태 {ours:7,d} 실수")
print()
print("  주의: 우리 상태는 R 회 재사용되지만 weight-tied 라 **상태는 하나**다.")
print("        Mamba-3 는 층마다 독립 상태를 갖는다.")

# ================================================================== M2
print()
print("=" * 92)
print("[M2] 같은 자로 잰 기억 수명   ⚠ Mamba-3 쪽은 무효다 — [M3] 을 먼저 읽을 것")
print("=" * 92)
print("  Mamba-3 는 헤드마다 커널이 다르므로 헤드 평균과 최고 헤드를 같이 낸다.")
print()
ours_r = snr_curve(BASE, LAGS)
rows = {"우리 (p=64, r_max=.999)": (ours_r, 3)}
for ds in (32, 64, 128):
    rows[f"Mamba-3 (d_state={ds}, 1층)"] = (mamba3_snr(LAGS, d_state=ds), 3)

print(f"  {'Δ':>6s}" + "".join(f"{k:>26s}" for k in rows))
print("-" * 92)
for i, D in enumerate(LAGS):
    print(f"  {D:6d}" + "".join(f"{v[0][i,3]:26.3f}" for v in rows.values()))
print()
for k, (r, c) in rows.items():
    extra = f"   최고헤드 수명 {life(r, 4)}" if r.shape[1] > 4 else ""
    print(f"  {k:30s} 수명 = {life(r, c):5d} 토큰   SNR@Δ=1 = {r[0,3]:7.2f}{extra}")

# ================================================================== M3
print()
print("=" * 92)
print("[M3] 이 측정은 Mamba-3 에 대해 **무효**다 — 두 가지 이유")
print("=" * 92)
import torch.nn.functional as F  # noqa: E402

from mamba3_ref import heavy_tail  # noqa: E402

torch.manual_seed(0)
_cfg = Mamba3Config(d_model=128, d_state=32, headdim=32, vocab_size=512,
                    use_quadratic=True)
_L = Mamba3Layer(_cfg).eval()
_e = torch.nn.Embedding(512, 128)
torch.nn.init.normal_(_e.weight, std=1.0)
with torch.no_grad():
    _u = _e(torch.randint(0, 512, (8, 256)))
    _p = _L.in_proj(_u).split([_L.d_inner, _L.d_inner, 32, 32,
                               _L.nheads, _L.nheads, _L.nheads, _L.n_ang], -1)
    Bn, Cn = _L.B_norm(_p[2]), _L.C_norm(_p[3])
    s_bias = torch.einsum("btn,bsn->bts", Cn + _L.C_bias, Bn + _L.B_bias)
    s_pure = torch.einsum("btn,bsn->bts", Cn, Bn)
    _dt = F.softplus(_p[4] + _L.dt_bias)
    _A = -heavy_tail(_p[5].float()).clamp(max=-_cfg.A_floor)
    _tau = (1.0 / (_A * _dt).abs())

print("  (1) 초기화에서 커널이 **내용을 구별하지 않는다** — 바이어스가 지배한다")
print(f"      score = C·B :  평균 {s_bias.mean():8.3f}  표준편차 {s_bias.std():7.3f}"
      f"   변동/평균 = {(s_bias.std()/s_bias.mean().abs()):.3f}")
print(f"      B_bias·C_bias 상수항만 = {(_L.C_bias @ _L.B_bias).item():.3f}"
      f"  → 평균의 {100*(_L.C_bias@_L.B_bias).item()/s_bias.mean().item():.1f}%")
print(f"      바이어스 제거 시 : 평균 {s_pure.mean():8.3f}  표준편차 {s_pure.std():7.3f}")
print("      B_bias, C_bias 가 ones(d_state) 로 초기화되므로 모든 (t,n) 쌍에")
print("      같은 상수 32 가 얹힌다. **맞는 항목이 특별해지지 않는다.**")
print()
print("  (2) 초기화에서 **감쇠가 사실상 없다**")
print(f"      τ = 1/|A·DT| :  중앙값 {_tau.median():,.0f} 토큰")
print(f"      A_floor=1e-4, DT∈[1e-3,1e-1] 이므로 |A·DT| ~ 1e-6 이다.")
print("      T=1600 구간에서 아무것도 안 잊는다 → SNR 곡선이 평평한 이유.")
print()
print("  → 위 [M2] 의 Mamba-3 '수명' 은 SNR 이 1 근처에서 흔들린 결과이지")
print("     회상 능력이 아니다. **이 비교로 우열을 말하면 안 된다.**")
print()
print("  그리고 이 무효성 자체가 구조적 사실을 말한다:")
print("  **Mamba-3 의 기억 프로필은 전부 학습되는 양이고(dt_bias, A, B/C bias),")
print("    우리 것은 초기화에 박제된다(CAPACITY §4.2: τ 는 학습되지 않는다).**")

print()
print("=" * 92)
print("[M4] 공정성 — 이 비교가 무엇을 말하고 무엇을 말하지 않는가")
print("=" * 92)
print("  (a) 둘 다 **학습 전**이다.")
print("      우리 τ 는 학습되지 않는다 (CAPACITY §4.2: 초기화가 0.68 bpb 를 결정).")
print("      Mamba-3 의 dt_bias 는 **학습되는 파라미터**다.")
print("      → Mamba-3 가 여기서 이기면 결정적, 우리가 이기면 결정적이지 않다.")
print()
print("  (b) Mamba-3 의 감쇠는 **입력 의존**이다 (ADT = A(u)·DT(u)).")
print("      무작위 토큰에서는 그 선택성이 발휘될 수 없다.")
print("      우리 α 는 고정이므로 이 측정이 우리에게는 최종값이지만")
print("      Mamba-3 에게는 **하한**이다.")
print()
print("  (c) 우리 커널은 채널 전체를 하나의 스칼라로 합치고(a_tn),")
print("      Mamba-3 는 헤드마다 스칼라를 갖는다. 헤드별로 잰 뒤 평균했다.")
