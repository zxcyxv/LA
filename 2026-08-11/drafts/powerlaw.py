"""§6.4 의 느린 저장소가 거듭제곱 망각을 주는가 — 종이 유도의 수치 확인.

망각 곡선 K(Δ) 를 만들고 **국소 로그-로그 기울기** d log K / d log Δ 를 본다.
  거듭제곱 Δ^-β  → 기울기가 상수 −β
  지수 e^{-Δ/τ}  → 기울기가 −Δ/τ 로 계속 가팔라짐 (발산)
  로그 ln(τ/Δ)   → 기울기가 0 으로 천천히 감
"""
import numpy as np

D = np.unique(np.round(np.logspace(0, 4, 260)).astype(int))  # Δ = 1 .. 10000


def loglog_slope(K):
    lx, ly = np.log(D), np.log(np.maximum(K, 1e-300))
    return np.gradient(ly, lx)


def report(name, K, probes=(10, 100, 1000, 5000)):
    s = loglog_slope(K)
    cells = []
    for p in probes:
        i = int(np.argmin(np.abs(D - p)))
        cells.append(f"{s[i]:8.3f}")
    # 유효 구간(K 가 초기값의 1e-6 위)에서 기울기의 변동폭
    live = K > K[0] * 1e-6
    rng = (s[live].max() - s[live].min()) if live.sum() > 3 else np.nan
    print(f"  {name:34s} " + " ".join(cells) + f"   |  변동폭 {rng:7.3f}")


# ---------------------------------------------------------------- 시정수 뱅크
p = 64


def taus(kind):
    if kind == "uniform_r":          # 현재 기본값 decay_init="uniform"
        r = np.linspace(0.90, 0.999, p)
        return -1.0 / np.log(r)
    if kind == "logtau":             # EXTRAPOLATION.md 의 decay_init="logtau"
        return np.exp(np.linspace(np.log(2.0), np.log(2000.0), p))
    raise ValueError(kind)


def bank(tau, w):
    """K(Δ) = Σ_j w_j exp(−Δ/τ_j)"""
    return (w[None, :] * np.exp(-D[:, None] / tau[None, :])).sum(1)


print("=" * 92)
print("국소 로그-로그 기울기  d log K / d log Δ        (거듭제곱이면 상수여야 한다)")
print("=" * 92)
print(f"  {'':34s} " + " ".join(f"Δ={p:<6d}" for p in (10, 100, 1000, 5000)))
print("-" * 92)

print("\n[기준선]")
report("단일 지수  τ=100", np.exp(-D / 100.0))
report("이론 거듭제곱  Δ^-0.5", D.astype(float) ** -0.5)
report("이론 거듭제곱  Δ^-1.0", D.astype(float) ** -1.0)

print("\n[현재 구조 — 병렬 뱅크, 진폭 균등]")
for kind in ("uniform_r", "logtau"):
    t = taus(kind)
    report(f"S 뱅크 ({kind})", bank(t, np.ones(p)))

print("\n[§6.4 캐스케이드 — 채널마다 느린 저장소 W 추가]")
# W 기여: η g γ (μ^{Δ+1} − λ^{Δ+1})/(μ − λ),  |μ| > |λ|
for kind in ("uniform_r", "logtau"):
    t = taus(kind)
    lam = np.exp(-1.0 / t)
    for mult in (4.0, 20.0):
        mu = np.exp(-1.0 / (t * mult))          # τ_W = mult × τ_S
        casc = np.zeros((len(D), p))
        for j in range(p):
            l, m = lam[j], mu[j]
            casc[:, j] = (m ** (D + 1) - l ** (D + 1)) / (m - l)
        K = bank(t, np.ones(p)) + 1.0 * casc.sum(1)
        report(f"S + W  ({kind}, τ_W={mult:g}τ_S)", K)

print("\n[처방 — 진폭을 τ^-β 로 가중  (ρ(τ) ∝ τ^{-1-β})]")
t = taus("logtau")
for beta in (0.5, 1.0):
    report(f"S 뱅크 (logtau, w ∝ τ^-{beta})", bank(t, t ** (-beta)))

print("\n[처방을 캐스케이드에 적용]")
lam = np.exp(-1.0 / t)
mu = np.exp(-1.0 / (t * 4.0))
casc = np.zeros((len(D), p))
for j in range(p):
    casc[:, j] = (mu[j] ** (D + 1) - lam[j] ** (D + 1)) / (mu[j] - lam[j])
for beta in (0.5, 1.0):
    w = t ** (-beta)
    report(f"S+W (logtau, w ∝ τ^-{beta})", bank(t, w) + (w[None, :] * casc).sum(1))

print()
print("=" * 92)
print("범위 확인 — 거듭제곱 근사는 뱅크가 덮는 τ 구간 안에서만 성립한다")
print("=" * 92)
for kind in ("uniform_r", "logtau"):
    t = taus(kind)
    print(f"  {kind:12s}  τ ∈ [{t.min():8.2f}, {t.max():8.2f}]   "
          f"중앙 {np.median(t):8.2f}   τ>128 채널 {int((t > 128).sum())}/{p}")
