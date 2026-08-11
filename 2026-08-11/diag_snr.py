"""SNR 망각 곡선 — FORGETTING.md §9-1 이 남긴 구멍을 닫는다.

`FORGETTING.md` 는 **신호 포락선** `K(Δ) = Σ_j γ_j e^{−Δ/τ_j}` 만 재고
"SNR 이 아니라 신호만 쟀다. 이 문서의 모든 결론이 여기 걸려 있다" 고 적었다.
Fusi 의 거듭제곱은 *신호/간섭잡음* 곡선이므로 그것으로 다시 재야 한다.

측정 방식 — **재제시(re-presentation) 회상**:
    길이 T 의 무작위 토큰열을 만들고, 마지막 위치의 토큰을 위치 `T−1−Δ` 의 것과
    **같게** 놓는다. 그러면 마지막 행 `A[T−1, ·]` 에서
        신호 = A[T−1, T−1−Δ]      (내용이 맞는 항목)
        잡음 = rms A[T−1, m]       (m ≠ 자기·신호 — 다른 기억들의 간섭)
    이다. `SNR(Δ) = |신호| / 잡음`.

이 값이 1 을 넘는 최대 Δ 가 **유효 기억 수명**이고, 레포에 없던 장기기억 눈금자다.

    python diag_snr.py            # 전부 (~2분)
    python diag_snr.py --only S2
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 레포 루트

from model1 import Model1, Model1Config

torch.manual_seed(0)


# ------------------------------------------------------------------ 커널 한 행
@torch.no_grad()
def last_row(model: Model1, h: torch.Tensor) -> torch.Tensor:
    """A[:, T−1, :] 만 계산한다 (전체 (T,T) 를 만들지 않는다).

    attention_matrix 와 같은 식이되 t = T−1 로 고정:
        A[b,t,n] = p^{−1/2} Σ_j (qr·kr + qi·ki)[b,t,n,j] · exp(−α_j (t−n))
    """
    blk = model.block
    _, q, k = blk.project(h)
    Phi = blk.cumulative_phase(h)
    rot = torch.polar(torch.ones_like(Phi), -Phi)
    qs, ks = q * rot, k * rot
    B, T, p = qs.shape
    t = T - 1
    inner = (qs.real[:, t, :][:, None, :] * ks.real          # (B,T,p)
             + qs.imag[:, t, :][:, None, :] * ks.imag)
    lag = (t - torch.arange(T, device=h.device)).clamp(min=0).to(blk.alpha.dtype)
    decay = torch.exp(-lag[None, :, None] * blk.alpha[None, None, :])
    return (inner * decay).sum(-1) / math.sqrt(p)            # (B,T)


@torch.no_grad()
def snr_curve(cfg_kw: dict, lags, T=1600, B=24, V=512, seed=0):
    """각 lag 에서 신호·잡음·SNR 을 잰다."""
    torch.manual_seed(seed)
    model = Model1(Model1Config(vocab_size=V, **cfg_kw)).eval()
    g = torch.Generator().manual_seed(seed + 1)
    out = []
    for D in lags:
        x = torch.randint(0, V, (B, T), generator=g)
        x[:, T - 1] = x[:, T - 1 - D]                        # 재제시
        row = last_row(model, model.embed(x))                # (B,T)
        sig = row[:, T - 1 - D].abs()
        mask = torch.ones(T, dtype=torch.bool)
        mask[T - 1] = False                                  # 자기항 제외
        mask[T - 1 - D] = False                              # 신호 제외
        noise = row[:, mask].pow(2).mean(1).sqrt()
        out.append((D, sig.mean().item(), noise.mean().item(),
                    (sig / noise).mean().item()))
    return np.array(out)                                     # (L,4)


def loglog_beta(D, y, lo, hi):
    m = (D >= lo) & (D <= hi) & (y > 0)
    if m.sum() < 4:
        return float("nan"), float("nan")
    s = np.gradient(np.log(y[m]), np.log(D[m]))
    return -s.mean(), s.std()


LAGS = [1, 2, 3, 5, 8, 12, 20, 32, 50, 80, 128, 200, 320, 500, 800, 1200]
BASE = dict(d=128, p=64, R=4, psi_inhib=True)


# ================================================================== S1
def sec_S1():
    print("=" * 90)
    print("[S1] SNR 망각 곡선 — 실제 모델 커널")
    print("=" * 90)
    print("  재제시 회상. 신호 = 내용이 맞는 항목, 잡음 = 나머지 기억들의 간섭(rms).")
    print()
    r = snr_curve(BASE, LAGS)
    print(f"  {'Δ':>6s} {'신호':>11s} {'잡음':>11s} {'SNR':>9s}")
    print("-" * 90)
    for D, s, n, q in r:
        star = "  ← SNR<1" if q < 1 else ""
        print(f"  {int(D):6d} {s:11.5f} {n:11.5f} {q:9.3f}{star}")
    ok = r[r[:, 3] >= 1.0]
    print()
    print(f"  **유효 기억 수명 (SNR ≥ 1 인 최대 Δ) = {int(ok[-1,0]) if len(ok) else 0} 토큰**")


# ================================================================== S2
def sec_S2():
    print("=" * 90)
    print("[S2] β 가 보존되는가 — 신호 vs SNR 의 로그-로그 기울기")
    print("=" * 90)
    print("  FORGETTING.md 는 신호에서 β=1.485 를 얻었다. SNR 도 같은 지수인가.")
    print()
    r = snr_curve(BASE, LAGS)
    D = r[:, 0]
    lo, hi = 20, 500
    for tag, col in (("신호 K(Δ)", 1), ("잡음", 2), ("SNR", 3)):
        b, sd = loglog_beta(D, r[:, col], lo, hi)
        print(f"  {tag:12s}  지수 = {b:6.3f} ± {sd:.3f}    (창 Δ∈[{lo},{hi}])")
    print()
    print("  이론 예측: 잡음은 최근 항목들이 지배하므로 신호의 Δ 와 무관해야 한다.")
    print("             그러면 SNR 지수 = 신호 지수 이고 β 가 보존된다.")


# ================================================================== S3
def sec_S3():
    print("=" * 90)
    print("[S3] 잡음이 정말 Δ 에 무관한가")
    print("=" * 90)
    r = snr_curve(BASE, LAGS)
    n = r[:, 2]
    print(f"  잡음 범위 [{n.min():.5f}, {n.max():.5f}]   "
          f"최대/최소 = {n.max()/n.min():.3f}")
    print(f"  신호 범위 [{r[:,1].min():.5f}, {r[:,1].max():.5f}]   "
          f"최대/최소 = {r[:,1].max()/r[:,1].min():.1f}")
    print()
    print("  → 잡음이 거의 상수면 SNR 의 모양은 신호가 정한다.")


# ================================================================== S4
def sec_S4():
    print("=" * 90)
    print("[S4] 창 넓히기 — FORGETTING.md §11-2 의 예측을 SNR 로 시험")
    print("=" * 90)
    print("  예측: r_max 를 올리면 유효 기억 수명이 늘어난다. 파라미터 0 개.")
    print()
    print(f"  {'설정':34s} {'τ_max':>9s} {'수명(SNR≥1)':>12s} {'SNR@Δ=200':>11s}")
    print("-" * 90)
    for rmax, p, tag in ((0.999, 64, "현재 기본값"), (0.9999, 64, ""),
                         (0.99999, 64, ""), (0.9999, 256, "")):
        kw = dict(BASE, r_max=rmax, p=p)
        r = snr_curve(kw, LAGS)
        ok = r[r[:, 3] >= 1.0]
        life = int(ok[-1, 0]) if len(ok) else 0
        at200 = r[np.argmin(np.abs(r[:, 0] - 200)), 3]
        name = f"p={p}, r_max={rmax:g}" + (f" ({tag})" if tag else "")
        print(f"  {name:34s} {-1/math.log(rmax):9.1f} {life:12d} {at200:11.3f}")


# ================================================================== S5
def sec_S5():
    print("=" * 90)
    print("[S5] logtau 와 처방 — FORGETTING.md §6 을 SNR 로 재판정")
    print("=" * 90)
    print(f"  {'설정':34s} {'수명(SNR≥1)':>12s} {'SNR 지수':>16s}")
    print("-" * 90)
    for kw, name in ((dict(BASE, r_max=0.9999), "uniform (현재)"),
                     (dict(BASE, r_max=0.9999, decay_init="logtau"), "logtau")):
        r = snr_curve(kw, LAGS)
        ok = r[r[:, 3] >= 1.0]
        b, sd = loglog_beta(r[:, 0], r[:, 3], 20, 500)
        print(f"  {name:34s} {int(ok[-1,0]) if len(ok) else 0:12d} "
              f"{b:9.3f} ± {sd:.3f}")


# ================================================================== S6
class gamma_dial:
    """진폭 가중을 γ → γ·τ^{+a} 로 바꾼다 (§7 의 다이얼, β ≈ 1.49 − a).

    `gamma` 가 클래스 property 라 인스턴스로 못 덮는다. 블록 클래스에 잠시 씌우고
    빠져나올 때 되돌린다. a=0 이면 원본과 정확히 같다.
    """

    def __init__(self, cls, a):
        self.cls, self.a, self.orig = cls, a, cls.gamma

    def __enter__(self):
        orig, a = self.orig, self.a
        self.cls.gamma = property(
            lambda s: orig.fget(s) * (1.0 / s.alpha).pow(a))
        return self

    def __exit__(self, *exc):
        self.cls.gamma = self.orig
        return False


def sec_S6():
    print("=" * 90)
    print("[S6] β 다이얼을 SNR 로 — 수명을 어디까지 밀 수 있나")
    print("=" * 90)
    print("  진폭을 γ·τ^{+a} 로 두면 §7 의 다이얼이 β ≈ 1.49 − a 로 움직인다.")
    print("  β 가 낮을수록 천천히 잊으므로 a 를 올리면 수명이 늘어야 한다.")
    print("  대가는 최근 SNR 이다 (§15.1 의 거래).")
    print()
    from model1 import DrivenComplexLinearAttention as Blk

    lg = np.log(np.array(LAGS, dtype=float))
    print(f"  {'설정':30s} {'수명':>6s} {'SNR@1':>8s} {'SNR@200':>9s} "
          f"{'∫총':>7s} {'∫Δ≤32':>7s} {'∫Δ>32':>7s}")
    print("-" * 90)
    for name, kw in (("uniform (현재)", dict(BASE)),
                     ("logtau", dict(BASE, decay_init="logtau"))):
        for a in (0.0, 0.5, 1.0):
            with gamma_dial(Blk, a):
                r = snr_curve(kw, LAGS)
            ok = r[r[:, 3] >= 1.0]
            s = r[:, 3]
            m = np.array(LAGS) <= 32
            print(f"  {name + f',  a={a:.1f}':30s} "
                  f"{int(ok[-1,0]) if len(ok) else 0:6d} {r[0,3]:8.2f} "
                  f"{r[np.argmin(np.abs(r[:,0]-200)),3]:9.3f} "
                  f"{np.trapezoid(s, lg):7.2f} {np.trapezoid(s[m], lg[m]):7.2f} "
                  f"{np.trapezoid(s[~m], lg[~m]):7.2f}")
    print()
    print("  a=0 은 현재 구조와 정확히 같다 (환원 확인).")


SECTIONS = {"S1": sec_S1, "S2": sec_S2, "S3": sec_S3, "S4": sec_S4,
            "S5": sec_S5, "S6": sec_S6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(SECTIONS))
    a = ap.parse_args()
    for i, k in enumerate(([a.only] if a.only else sorted(SECTIONS))):
        if i:
            print()
        SECTIONS[k]()


if __name__ == "__main__":
    main()
