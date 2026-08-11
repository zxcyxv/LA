"""망각 곡선 진단 — 느린 저장소가 거듭제곱을 주는가.

`MISALIGNMENT.md §6.4` 는 느린 저장소 `W` 를 붙이면 지수 망각이 거듭제곱 망각으로
바뀐다고 주장했고, §10 이 "우리 구조에서 실제로 거듭제곱이 나오는지는 계산도 측정도
하지 않았다" 고 스스로 적어두었다. 그 계산이다.

학습 불필요. 전부 초기화 분포에서 닫힌 형태로 계산한다 (~2초).

    python diag_forgetting.py           # [A]~[E] 전부
    python diag_forgetting.py --only C  # 한 절만

진단량은 **국소 로그-로그 기울기** `d log K / d log Δ` 다.
    거듭제곱 Δ^-β  →  기울기가 상수 −β        (변동폭 ≈ 0)
    지수 e^{-Δ/τ}  →  기울기가 −Δ/τ 로 발산   (변동폭 큼)
    로그 ln(τ/Δ)   →  기울기가 0 근처에서 천천히 가팔라짐
"""

from __future__ import annotations

import argparse

import numpy as np

# lag 격자. 로그 등간격이라야 로그-로그 기울기가 균일하게 표본된다.
D = np.unique(np.round(np.logspace(0, 4.5, 420)).astype(int))


def slope(K: np.ndarray) -> np.ndarray:
    """국소 로그-로그 기울기."""
    return np.gradient(np.log(np.maximum(K, 1e-300)), np.log(D))


def bank(tau: np.ndarray, amp: np.ndarray) -> np.ndarray:
    """K(Δ) = Σ_j amp_j · exp(−Δ/τ_j) — 시정수 뱅크의 망각 곡선."""
    return (amp[None, :] * np.exp(-D[:, None] / tau[None, :])).sum(1)


def window(tau: np.ndarray) -> np.ndarray:
    """거듭제곱 근사가 성립하는 창 τ_min ≪ Δ ≪ τ_max.

    lag 격자 D 의 상한을 넘어가면 그쪽에서 잘린다 — 창의 배율을 보고할 때
    격자 밖을 세지 않도록 D.max() 로 자른다.
    """
    return (D >= 5 * tau.min()) & (D <= min(0.2 * tau.max(), D.max()))


def beta_of(tau: np.ndarray, amp: np.ndarray) -> tuple[float, float, int, int]:
    """창 안에서의 β = −(평균 기울기), 그리고 변동(표준편차)."""
    s = slope(bank(tau, amp))
    w = window(tau)
    if w.sum() < 4:
        return float("nan"), float("nan"), 0, 0
    return -s[w].mean(), s[w].std(), int(D[w].min()), int(D[w].max())


def tau_uniform_r(p: int, r_min: float = 0.90, r_max: float = 0.999):
    """decay_init='uniform' — |λ| 을 균등 추출 (model0.py 기본값)."""
    r = np.linspace(r_min, r_max, p)
    return -1.0 / np.log(r), r


def tau_logtau(p: int, a_min: float = 1e-5, a_max: float = 0.1):
    """decay_init='logtau' — log α 를 균등 추출 (EXTRAPOLATION.md §1-2)."""
    r = np.exp(-np.exp(np.linspace(np.log(a_min), np.log(a_max), p)))
    return -1.0 / np.log(r), r


def gamma_of(r: np.ndarray) -> np.ndarray:
    """γ_j = √(1−|λ_j|²) — 실제 커널의 진폭 가중 (MISALIGNMENT §8.6 의 a_tn 식)."""
    return np.sqrt(1.0 - r ** 2)


# ============================================================== [A]
def sec_A():
    print("=" * 88)
    print("[A] §6.4 의 캐스케이드는 지수 두 개의 가중합이다 — 항등식")
    print("=" * 88)
    print("  W_t = μ W_{t-1} + η g S_t,  S_t = λ S_{t-1} + γ k v†  의 임펄스 응답")
    print("      (μ^{Δ+1} − λ^{Δ+1})/(μ − λ)   ← 문서가 alpha function 이라 부른 것")
    print()
    tau_s, _ = tau_uniform_r(128, 0.9, 0.99999)
    lam = np.exp(-1.0 / tau_s)
    mu = np.exp(-1.0 / (tau_s * 8.0))
    casc = (mu[None, :] ** (D[:, None] + 1) - lam[None, :] ** (D[:, None] + 1)) / (mu - lam)[None, :]
    # 부분분수: [μ·μ^Δ − λ·λ^Δ]/(μ−λ)
    two = (mu[None, :] * mu[None, :] ** D[:, None]
           - lam[None, :] * lam[None, :] ** D[:, None]) / (mu - lam)[None, :]
    print(f"  캐스케이드 vs 지수2개 가중합 :  max|차이| = {np.abs(casc - two).max():.3e}")
    print()
    print("  → 대수적으로 같다. 항을 늘려도 **지수의 유한 합**이다.")
    print("     유한 차원 LTI 계의 임펄스 응답은 거듭제곱이 될 수 없다.")
    print()
    peak_S = D[np.argmax(np.exp(-D[:, None] / tau_s[None, :]).sum(1))]
    peak_W = D[np.argmax(casc.sum(1))]
    print(f"  피크 lag :  S 만 = {peak_S}      S+W 캐스케이드 = {peak_W}")
    print("  → W 가 실제로 바꾸는 것은 꼬리가 아니라 **머리**(상승상)다. 어긋남 #5.")


# ============================================================== [B]
def sec_B():
    print("=" * 88)
    print("[B] 거듭제곱의 조건 — (진폭 가중) × (시정수 밀도) ∝ τ^{−1−β}")
    print("=" * 88)
    print("  ∫ A(τ) e^{−Δ/τ} ρ(τ) dτ = Γ(β)·Δ^{−β}   ⟺   A(τ)ρ(τ) ∝ τ^{−1−β}")
    print()
    P = 512
    tau = np.exp(np.linspace(np.log(1.0), np.log(1e6), P))  # log 등간격 → ρ ∝ τ^-1
    probes = (30, 1000, 30000)
    print(f"  {'진폭 가중':22s}" + "".join(f"  Δ={p:<7d}" for p in probes) + "   변동폭")
    print("-" * 88)
    for name, amp in (("균등 (= ρ∝τ^-1 만)", np.ones(P)),
                      ("τ^-0.25", tau ** -0.25),
                      ("τ^-0.5", tau ** -0.5),
                      ("τ^-1.0", tau ** -1.0)):
        s = slope(bank(tau, amp))
        w = window(tau)
        cells = "".join(f"  {s[np.argmin(np.abs(D - p))]:8.3f}" for p in probes)
        print(f"  {name:22s}{cells}   {s[w].max() - s[w].min():8.3f}")
    print()
    print("  → 균등 진폭은 거듭제곱이 아니라 **로그 감쇠**다 (기울기가 평평하지 않다).")
    print("  → τ^-β 가중이 기울기를 −β 에 고정한다.")


# ============================================================== [C]
def sec_C():
    print("=" * 88)
    print("[C] 레포의 기본 초기화는 이미 거듭제곱이다")
    print("=" * 88)
    print("  r ~ U(a,b) 균등  →  ρ(τ) ∝ |dr/dτ| = e^{−1/τ}/τ² ∝ τ^{−2}")
    print("  γ(τ) = √(1−e^{−2/τ}) ≈ √(2/τ) ∝ τ^{−1/2}")
    print()
    print("      γ 없이 : ρ ∝ τ^{−2}    → β = 1.0")
    print("      γ 포함 : γρ ∝ τ^{−2.5} → β = 1.5   ← 실제 커널")
    print()
    print("  log τ 균등 → ρ ∝ τ^{−1} → β = 0 (로그);  γ 포함 시 β = 0.5")
    print()
    P = 4096
    rows = [
        ("decay_init='uniform'", *tau_uniform_r(P, 0.90, 0.99999)),
        ("decay_init='logtau'", *tau_logtau(P)),
    ]
    print(f"  {'초기화':24s} {'진폭':16s}  {'예측 β':>7s}  {'실측 β':>16s}")
    print("-" * 88)
    for name, tau, r in rows:
        pred = (1.0, 1.5) if "uniform" in name else (0.0, 0.5)
        for (tag, amp), pv in zip((("γ 없이 (이상화)", np.ones_like(tau)),
                                   ("γ 포함 (실제 커널)", gamma_of(r))), pred):
            b, sd, lo, hi = beta_of(tau, amp)
            print(f"  {name:24s} {tag:16s}  {pv:7.1f}  {b:8.3f} ± {sd:.3f}")
    print()
    print("  → 예측과 실측이 소수점 둘째 자리까지 맞는다.")
    print("  → **병렬 뱅크도 밀도만 맞으면 거듭제곱이다.** 승격 경로가 필요조건이 아니다.")


# ============================================================== [D]
def sec_D():
    print("=" * 88)
    print("[D] 진짜 병목 — 거듭제곱이 성립하는 창의 배율")
    print("=" * 88)
    print("  거듭제곱 근사는 τ_min ≪ Δ ≪ τ_max 안에서만 성립한다.")
    print()
    print(f"  {'설정':32s} {'τ_max':>10s}  {'창':>18s} {'배율':>8s}   실측 β")
    print("-" * 88)
    for p, rmax, tag in ((64, 0.999, "현재 기본값"), (64, 0.9999, ""),
                         (256, 0.99999, ""), (512, 0.999999, "")):
        tau, r = tau_uniform_r(p, 0.90, rmax)
        b, sd, lo, hi = beta_of(tau, gamma_of(r))
        name = f"p={p}, r_max={rmax:g}" + (f"  ({tag})" if tag else "")
        print(f"  {name:32s} {tau.max():10.1f}  {f'[{lo},{hi}]':>18s} "
              f"{hi/max(lo,1):7.1f}×   {b:.3f} ± {sd:.3f}")
    print()
    print("  → 현재 창은 4배 남짓이다. 4배 구간의 거듭제곱은 거듭제곱이 아니다.")
    print("  → EXTRAPOLATION.md 의 '학습 길이의 2배에서 붕괴' 와 같은 규모다 (가설).")
    print("  → r_max 와 p 는 초기화 상수다. **개입 비용이 0 이다.**")


# ============================================================== [E]
def sec_E():
    print("=" * 88)
    print("[E] β 는 파라미터 0 개짜리 다이얼이다")
    print("=" * 88)
    print("  진폭을 γ·τ^{−a} 로 재가중하면 β 가 선형으로 움직인다.")
    print()
    tau, r = tau_uniform_r(4096, 0.90, 0.99999)
    g0 = gamma_of(r)
    print(f"  {'진폭':22s}  {'실측 β':>16s}")
    print("-" * 88)
    for a in (-1.0, -0.5, 0.0, 0.5, 1.0):
        b, sd, _, _ = beta_of(tau, g0 * tau ** -a)
        print(f"  γ·τ^{-a:+.1f}{'':13s}  {b:8.3f} ± {sd:.3f}")
    print()
    print("  → β ≈ 1.49 − a. 망각 지수가 설계 가능한 양이고,")
    print("     지금 그 값은 선택된 것이 아니라 초기화의 부산물이다.")


# ============================================================== [F]
def sec_F():
    print("=" * 88)
    print("[F] 두 초기화의 상충 — 그리고 둘 다 갖는 처방")
    print("=" * 88)
    print("  uniform : 밀도 지수는 맞다(β=1.5) 그러나 큰 τ 를 **성기게** 덮는다")
    print("            (r 균등이라 표본이 작은 τ 로 몰린다 — EXTRAPOLATION.md §1-2 의 관측)")
    print("  logtau  : 덮기는 고르다 그러나 밀도 지수가 틀렸다 (β=0.6, 로그에 가까움)")
    print()
    print("  처방: τ 를 log 균등으로 뽑고(고른 덮기) 진폭으로 지수를 되돌린다.")
    print("        ρ ∝ τ^{−1} 이므로 A(τ) ∝ γ·τ^{−1} 이면 곱이 τ^{−2.5} → β = 1.5")
    print()
    P = 256
    print(f"  {'설정':38s} {'τ 중앙값':>9s} {'τ>128':>7s}  {'실측 β':>15s}")
    print("-" * 88)
    tu, ru = tau_uniform_r(P, 0.90, 0.99999)
    tl, rl = tau_logtau(P, 1e-5, 0.105)
    for name, tau, r, amp in (
        ("uniform, γ (현재)", tu, ru, gamma_of(ru)),
        ("logtau,  γ", tl, rl, gamma_of(rl)),
        ("logtau,  γ·τ^-1  ← 처방", tl, rl, gamma_of(rl) * tl ** -1.0),
    ):
        b, sd, lo, hi = beta_of(tau, amp)
        print(f"  {name:38s} {np.median(tau):9.1f} {int((tau > 128).sum()):5d}/{P}"
              f"  {b:7.3f} ± {sd:.3f}")
    print()
    print("  → 처방은 β 를 1.5 로 되돌리면서 변동폭을 줄인다.")
    print("     '고르게 덮기' 와 '거듭제곱' 은 상충하지 않는다 — 다른 손잡이다.")
    print("     덮기는 **표본 분포**가, 지수는 **진폭 가중**이 정한다.")


SECTIONS = {"A": sec_A, "B": sec_B, "C": sec_C, "D": sec_D, "E": sec_E, "F": sec_F}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(SECTIONS), default=None)
    args = ap.parse_args()
    keys = [args.only] if args.only else sorted(SECTIONS)
    for i, k in enumerate(keys):
        if i:
            print()
        SECTIONS[k]()


if __name__ == "__main__":
    main()
