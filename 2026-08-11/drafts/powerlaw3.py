"""v3 — 실제 커널로. MISALIGNMENT §8.6 의 식:

    a_tn = p^{-1/2} Σ_j γ_j |z_tj||z_nj| e^{-α_j Δ} cos(...)      γ_j = √(1-|λ_j|²)

즉 채널 j 의 lag-Δ 기여는  γ_j · e^{-Δ/τ_j}  에 비례한다. γ 가 **진폭 가중**이다.

거듭제곱 조건:  (진폭 가중) × (시정수 밀도) ∝ τ^{-1-β}
"""
import numpy as np

D = np.unique(np.round(np.logspace(0, 4, 400)).astype(int))


def slope(K):
    return np.gradient(np.log(np.maximum(K, 1e-300)), np.log(D))


def fit(name, r, use_gamma, P=4096):
    tau = -1.0 / np.log(r)
    g = np.sqrt(1 - r ** 2) if use_gamma else np.ones_like(r)
    K = (g[None, :] * np.exp(-D[:, None] / tau[None, :])).sum(1)
    s = slope(K)
    lo, hi = 5 * tau.min(), 0.2 * tau.max()
    w = (D >= lo) & (D <= hi)
    if w.sum() < 4:
        print(f"  {name:46s}  창이 너무 좁다")
        return
    print(f"  {name:46s} β = {-s[w].mean():5.3f} ± {s[w].std():.3f}   "
          f"창 Δ∈[{int(lo)},{int(hi)}]")


P = 4096
print("=" * 96)
print("이론 예측")
print("=" * 96)
print("  r ~ U(a,b) 균등  →  τ = −1/ln r,  ρ(τ) ∝ |dr/dτ| = e^{-1/τ}/τ² ∝ τ^-2")
print("  γ(τ) = √(1−e^{-2/τ}) ≈ √(2/τ) ∝ τ^-0.5   (τ ≫ 1)")
print()
print("    γ 없이 : ρ ∝ τ^-2   = τ^{-1-β} → β = 1.0")
print("    γ 포함 : γ·ρ ∝ τ^-2.5 = τ^{-1-β} → β = 1.5")
print()
print("  log τ 균등  →  ρ(τ) ∝ τ^-1 → β = 0  (거듭제곱이 아니라 **로그 감쇠**)")
print("    γ 포함 : γ·ρ ∝ τ^-1.5 → β = 0.5")

print()
print("=" * 96)
print("수치 확인")
print("=" * 96)
r_u = np.linspace(0.90, 0.99999, P)                       # decay_init="uniform"
r_l = np.exp(-np.exp(np.linspace(np.log(1e-5), np.log(0.1), P)))  # "logtau": log α 균등

print("\n[decay_init='uniform'  — 레포 기본값]")
fit("γ 없이 (이상화)", r_u, False)
fit("γ 포함 (실제 커널)", r_u, True)

print("\n[decay_init='logtau'  — EXTRAPOLATION.md 가 도입]")
fit("γ 없이 (이상화)", r_l, False)
fit("γ 포함 (실제 커널)", r_l, True)

print()
print("=" * 96)
print("레포의 실제 설정 (p=64, r∈[0.90,0.999]) — 창이 얼마나 좁은가")
print("=" * 96)
for P2, rmax in ((64, 0.999), (64, 0.9999), (256, 0.99999)):
    r = np.linspace(0.90, rmax, P2)
    tau = -1 / np.log(r)
    g = np.sqrt(1 - r ** 2)
    K = (g[None, :] * np.exp(-D[:, None] / tau[None, :])).sum(1)
    s = slope(K)
    lo, hi = 5 * tau.min(), 0.2 * tau.max()
    w = (D >= lo) & (D <= hi)
    ok = w.sum() >= 4
    print(f"  p={P2:4d}  r_max={rmax:<8g}  τ_max={tau.max():9.1f}  "
          f"창 Δ∈[{int(lo)},{int(hi)}] (배율 {hi/lo:5.1f}×)  "
          + (f"β={-s[w].mean():5.3f}±{s[w].std():.3f}" if ok else "창 부족"))

print()
print("=" * 96)
print("γ 를 직접 다이얼로 쓰면 — 진폭을 γ·τ^-a 로 재가중")
print("=" * 96)
r = np.linspace(0.90, 0.99999, 4096)
tau = -1 / np.log(r)
g0 = np.sqrt(1 - r ** 2)
for a in (-1.0, -0.5, 0.0, 0.5):
    K = ((g0 * tau ** -a)[None, :] * np.exp(-D[:, None] / tau[None, :])).sum(1)
    s = slope(K)
    w = (D >= 5 * tau.min()) & (D <= 0.2 * tau.max())
    print(f"  진폭 = γ·τ^{-a:+.1f}    →   β = {-s[w].mean():5.3f} ± {s[w].std():.3f}")
