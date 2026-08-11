"""수명이 늘어난 것은 공짜인가 거래인가 — 그리고 왜 r_max 는 무효인가.

[T1] 전체 SNR 곡선 대조: 최근을 팔아 과거를 산 것인가
[T2] r_max 무효의 원인: 긴 채널의 **개수**가 안 늘어난다
[T3] 총 회상 예산: ∫ SNR dlogΔ 가 보존되는가 (Fusi 의 거래 가설)
"""
from __future__ import annotations

import math

import numpy as np
import torch

from diag_snr import BASE, snr_curve

LAGS = [1, 2, 3, 5, 8, 12, 20, 32, 50, 80, 128, 200, 320, 500, 800, 1200]

ARMS = {
    "uniform (현재 기본)": dict(BASE),
    "uniform r_max=.99999": dict(BASE, r_max=0.99999),
    "logtau": dict(BASE, decay_init="logtau"),
}

print("=" * 94)
print("[T1] 전체 SNR 곡선 — 최근을 팔아 과거를 샀는가")
print("=" * 94)
curves = {}
for name, kw in ARMS.items():
    curves[name] = snr_curve(kw, LAGS)

hdr = "".join(f"{n:>22s}" for n in ARMS)
print(f"  {'Δ':>6s}{hdr}")
print("-" * 94)
for i, D in enumerate(LAGS):
    row = "".join(f"{curves[n][i,3]:22.3f}" for n in ARMS)
    print(f"  {D:6d}{row}")

print()
print("  수명(SNR≥1):")
for n in ARMS:
    r = curves[n]
    ok = r[r[:, 3] >= 1.0]
    print(f"    {n:24s} {int(ok[-1,0]) if len(ok) else 0:6d} 토큰   "
          f"SNR@Δ=1 = {r[0,3]:7.2f}")

print()
print("=" * 94)
print("[T2] r_max 가 왜 무효인가 — 긴 채널의 개수")
print("=" * 94)
print("  큰 Δ 에서 신호는 τ_j > Δ 인 채널만 낸다. 잡음은 **전 채널**이 낸다.")
print("  따라서 SNR 을 정하는 것은 τ 의 최댓값이 아니라 **긴 채널의 비율**이다.")
print()
p = 64
print(f"  {'설정':26s} {'τ 중앙값':>9s} {'τ>80':>7s} {'τ>500':>7s} {'τ_max':>10s}")
print("-" * 94)
for name, rmax, kind in (("uniform r_max=.999", 0.999, "u"),
                         ("uniform r_max=.9999", 0.9999, "u"),
                         ("uniform r_max=.99999", 0.99999, "u"),
                         ("logtau", None, "l")):
    if kind == "u":
        r = np.linspace(0.90, rmax, p)
    else:
        a_hi, a_lo = -math.log(0.90), -math.log(0.999)
        r = np.exp(-np.exp(np.linspace(math.log(a_lo), math.log(a_hi), p)))
    tau = -1.0 / np.log(r)
    print(f"  {name:26s} {np.median(tau):9.1f} {int((tau>80).sum()):5d}/{p} "
          f"{int((tau>500).sum()):5d}/{p} {tau.max():10.1f}")

print()
print("  → r_max 를 100 배 올려도 긴 채널 **개수**는 거의 그대로다.")
print("     몇 개가 아주 길어질 뿐이고, 잡음은 64 개 전부가 만든다.")

print()
print("=" * 94)
print("[T3] 총 회상 예산 — Fusi 의 거래 가설")
print("=" * 94)
print("  가소성-안정성 딜레마가 참이면 ∫ SNR dlogΔ 가 대략 보존되어야 한다.")
print("  (로그 척도 적분 = '몇 자릿수의 과거를 몇 배의 신뢰도로 아는가')")
print()
lg = np.log(np.array(LAGS, dtype=float))
print(f"  {'설정':26s} {'∫SNR dlogΔ':>12s} {'∫ (Δ≤32)':>11s} {'∫ (Δ>32)':>11s}")
print("-" * 94)
for n in ARMS:
    s = curves[n][:, 3]
    tot = np.trapezoid(s, lg)
    m = np.array(LAGS) <= 32
    early = np.trapezoid(s[m], lg[m])
    late = np.trapezoid(s[~m], lg[~m])
    print(f"  {n:26s} {tot:12.2f} {early:11.2f} {late:11.2f}")
print()
print("  거래라면: 총합은 비슷하고 early↓ late↑ 여야 한다.")
print("  공짜라면: 총합 자체가 커진다.")
