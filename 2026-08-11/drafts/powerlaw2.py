"""v2 — 세 가지를 갈라서 본다.

  Q1. §6.4 캐스케이드가 거듭제곱을 주는가
  Q2. 무엇이 거듭제곱을 주는가
  Q3. W 가 '느린 지수 하나 더'  이상의 것을 주는가

주의: 거듭제곱 근사는 τ_min ≪ Δ ≪ τ_max 안에서만 성립하므로 창 안에서 재야 한다.
"""
import numpy as np


def slope(D, K):
    return np.gradient(np.log(np.maximum(K, 1e-300)), np.log(D))


def report(name, D, K, win):
    s = slope(D, K)
    m = (D >= win[0]) & (D <= win[1])
    probes = [win[0], int(np.sqrt(win[0] * win[1])), win[1]]
    cells = " ".join(f"{s[np.argmin(np.abs(D - p))]:7.3f}" for p in probes)
    print(f"  {name:40s} {cells}   변동폭 {s[m].max()-s[m].min():6.3f}")


# ============================================================ Q1 / Q2
print("=" * 96)
print("Q1·Q2  넓은 뱅크(τ ∈ [1, 1e6], 512채널)에서 창 Δ ∈ [30, 3e4] 안의 기울기")
print("=" * 96)
D = np.unique(np.round(np.logspace(0.5, 5, 400)).astype(int))
WIN = (30, 30000)
print(f"  {'':40s}  Δ=30    Δ≈950   Δ=3e4")
print("-" * 96)

P = 512
tau = np.exp(np.linspace(np.log(1.0), np.log(1e6), P))
E = np.exp(-D[:, None] / tau[None, :])          # (Δ, j)

report("이론  Δ^-0.5", D, D.astype(float) ** -0.5, WIN)
print()
report("뱅크, 진폭 균등  (= 현재 구조)", D, (E * 1.0).sum(1), WIN)
for b in (0.25, 0.5, 1.0):
    report(f"뱅크, 진폭 ∝ τ^-{b}", D, (E * tau ** -b).sum(1), WIN)

print()
print("  → 균등 진폭은 '로그 감쇠'다: 기울기가 0 근처에서 아주 천천히 가팔라진다.")
print("  → τ^-β 가중이 기울기를 −β 로 **평평하게** 고정한다. 이것이 거듭제곱의 조건.")

# ============================================================ Q1 캐스케이드
print()
print("=" * 96)
print("Q1  §6.4 캐스케이드 — W 를 붙이면 감쇠의 '종류'가 바뀌는가")
print("=" * 96)
lam = np.exp(-1.0 / tau)


def cascade(tau_s, mult):
    """(μ^{Δ+1} − λ^{Δ+1})/(μ − λ),  τ_W = mult·τ_S"""
    l = np.exp(-1.0 / tau_s)
    m = np.exp(-1.0 / (tau_s * mult))
    return (m[None, :] ** (D[:, None] + 1) - l[None, :] ** (D[:, None] + 1)) / (m - l)[None, :]


print(f"  {'':40s}  Δ=30    Δ≈950   Δ=3e4")
print("-" * 96)
report("뱅크 S 만, 균등", D, E.sum(1), WIN)
for mult in (4.0, 20.0):
    report(f"S + W (τ_W={mult:g}τ_S), 균등", D, E.sum(1) + cascade(tau, mult).sum(1), WIN)
report("뱅크 S 만, τ^-0.5", D, (E * tau ** -0.5).sum(1), WIN)
report("S + W (τ_W=4τ_S), τ^-0.5", D,
       ((E + cascade(tau, 4.0)) * tau ** -0.5).sum(1), WIN)

print()
print("  → 균등 진폭이면 W 를 붙여도 여전히 로그. τ^-β 면 W 유무와 무관하게 −β.")
print("  → **감쇠의 종류를 정하는 것은 진폭 가중이지 W 가 아니다.**")

# ============================================================ Q3
print()
print("=" * 96)
print("Q3  W 는 '느린 지수 하나 더' 이상인가 — 같은 시정수 쌍의 단순 합과 대조")
print("=" * 96)
tau_s = np.exp(np.linspace(np.log(1.0), np.log(1e5), 128))
mult = 8.0
tau_w = tau_s * mult
C = cascade(tau_s, mult)                              # 진짜 캐스케이드
l, m = np.exp(-1 / tau_s), np.exp(-1 / tau_w)
# 부분분수: (m^{D+1} − l^{D+1})/(m−l) = [m·m^D − l·l^D]/(m−l)  → 지수 두 개의 가중합
Eq = (m[None, :] * m[None, :] ** D[:, None] - l[None, :] * l[None, :] ** D[:, None]) / (m - l)[None, :]
print(f"  캐스케이드 vs 지수2개 가중합  max|차이| = {np.abs(C - Eq).max():.3e}")
print("  → 항등식이다. 캐스케이드는 **지수 두 개의 특정 가중합**일 뿐이다.")
print()
peak = D[np.argmax(C.sum(1))]
print(f"  캐스케이드 합의 피크 lag = {peak}   (S 만이면 피크는 lag 0)")
print("  → W 가 실제로 주는 것: **상승상**. 꼬리의 모양이 아니라 머리의 모양이다.")

# ============================================================ 실제 규모
print()
print("=" * 96)
print("실제 규모에서 — p=64, 레포의 두 초기화")
print("=" * 96)
D2 = np.unique(np.round(np.logspace(0, 3.5, 200)).astype(int))
for name, t in (("uniform_r (기본)", -1 / np.log(np.linspace(0.90, 0.999, 64))),
                ("logtau", np.exp(np.linspace(np.log(2), np.log(2000), 64)))):
    E2 = np.exp(-D2[:, None] / t[None, :])
    for b, tag in ((0.0, "균등"), (0.5, "τ^-0.5")):
        K = (E2 * t ** -b).sum(1)
        s = slope(D2, K)
        w = (D2 >= 5 * t.min()) & (D2 <= 0.2 * t.max())
        if w.sum() > 3:
            print(f"  {name:18s} {tag:8s}  창 Δ∈[{int(5*t.min())},{int(0.2*t.max())}] "
                  f"기울기 {s[w].mean():6.3f} ± {s[w].std():.3f}")
