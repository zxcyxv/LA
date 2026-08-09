"""Model 0 검증 — 학습 전에 구조가 의도대로 되어 있는지만 확인한다.

  1. parallel ≡ recurrent          (두 계산 경로가 같은 f 를 내는가)
  2. T=1 에서 정확한 3차 동차성      (f(c·h) = c³ f(h))
  3. 감쇠가 실제로 λ^Δ 로 작동하는가  (1/t 제거 후 시간불변인가)
  4. 정렬 O(τ) vs 비상관 O(√τ) 선택성이 살아 있는가
  5. ‖f‖ 이 α 범위에 무관하게 O(‖h‖) 인가 (γ 스케일의 역할)
  6. 역전파가 네 파라미터 전부에 흐르는가
  7. 재귀 깊이에 따른 E_D, ρ, d_t 궤적이 관측되는가
"""

import math

import torch

from model0 import (
    ComplexLinearAttention,
    Model0,
    Model0Config,
    dirichlet_energy,
    self_gain,
    spectral_gain,
)

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def banner(msg):
    print(f"\n{'─' * 68}\n{msg}\n{'─' * 68}")


# ---------------------------------------------------------------------- #
banner("1. parallel ≡ recurrent")
cfg = Model0Config(d=64, p=32, R=3, chunk_p=8)
blk = ComplexLinearAttention(cfg).to(DEV).double()
h = torch.randn(3, 40, cfg.d, device=DEV, dtype=torch.float64)

f_par = blk(h, mode="parallel")
f_rec = blk(h, mode="recurrent")
err = (f_par - f_rec).abs().max().item()
print(f"max |f_parallel - f_recurrent| = {err:.3e}   (scale ‖f‖∞ = {f_par.abs().max():.3f})")
assert err < 1e-10, "두 경로 불일치"

# step() 스트리밍도 같은 답을 내는지
S = None
f_step = []
for t in range(h.shape[1]):
    f_t, S = blk.step(h[:, t], S)
    f_step.append(f_t)
f_step = torch.stack(f_step, 1)
print(f"max |f_parallel - f_step|     = {(f_par - f_step).abs().max().item():.3e}")


# ---------------------------------------------------------------------- #
banner("2. T=1 에서 정확한 3차 동차성  f(c·h) = c³ f(h)")
h1 = torch.randn(4, 1, cfg.d, device=DEV, dtype=torch.float64)
f1 = blk(h1)
for c in (0.5, 2.0, 3.0):
    fc = blk(c * h1)
    ratio = (fc / f1).flatten()
    print(f"  c={c:>4}:  f(c·h)/f(h) = {ratio.mean():.6f} ± {ratio.std():.2e}   (c³ = {c**3:.6f})")
    assert torch.allclose(fc, c**3 * f1, rtol=1e-9), "3차 동차성 위배"

# 자기항 계수와 일치하는지: f_1 = a_11 · h_1
a11 = self_gain(blk, h1)[:, 0]
print(f"  ‖f_1 - a_11·h_1‖∞ = {(f1[:, 0] - a11[:, None] * h1[:, 0]).abs().max().item():.3e}")


# ---------------------------------------------------------------------- #
banner("3. 감쇠 λ^Δ 확인 + 시간불변성 (1/t 제거의 효과)")
cfg2 = Model0Config(d=32, p=8, R=1, r_min=0.9, r_max=0.9, theta_max=0.0)
b2 = ComplexLinearAttention(cfg2).to(DEV).double()
T = 60
# 모든 위치에 같은 벡터를 넣으면 a_tn 은 Δ 에만 의존한다 (θ=0 이므로 순수 감쇠)
v = torch.randn(cfg2.d, device=DEV, dtype=torch.float64)
h2 = v.expand(1, T, cfg2.d).contiguous()
A = b2.attention_matrix(h2)[0]
lam_mag = b2.lam.abs()[0].item()
print(f"  |λ| = {lam_mag:.4f} (τ = {1/b2.alpha[0].item():.1f})")
print(f"  a_{{t,t-Δ}} 의 인접비 a(Δ+1)/a(Δ)   (이론값 |λ| = {lam_mag:.4f}):")
row = A[T - 1]
for dl in (0, 1, 5, 10, 20, 40):
    print(f"    Δ={dl:>3}:  {(row[T - 1 - dl - 1] / row[T - 1 - dl]).item():.6f}")

# 같은 국소 패턴이 시퀀스 앞/뒤 어디에 있어도 같은 이득을 갖는가 (1/t 였다면 깨진다)
pat = torch.randn(1, 8, cfg2.d, device=DEV, dtype=torch.float64)
h_a = torch.zeros(1, T, cfg2.d, device=DEV, dtype=torch.float64)
h_b = torch.zeros_like(h_a)
h_a[:, 2:10] = pat
h_b[:, 40:48] = pat
fa = b2(h_a)[0, 2:10]
fb = b2(h_b)[0, 40:48]
print(f"\n  동일 패턴 위치 t=2 vs t=40 의 f 차이: {(fa - fb).abs().max().item():.3e}  (시간불변)")


# ---------------------------------------------------------------------- #
banner("4. 위상 정렬 선택성:  정렬 O(τ)  vs  비상관 O(√τ)")
print("  1/t 대신 감쇠가 선택성을 만든다면, 정렬/비상관 이득비가 √((1+|λ|)/(1-|λ|)) 로 커져야 한다.")
print(f"\n  {'|λ|':>8} {'τ':>8} {'정렬 ‖f‖':>12} {'비상관 ‖f‖':>12} {'실측비':>9} {'예측비':>9} {'실측/예측':>10}")
base = None
for lam_mag in (0.8, 0.9, 0.95, 0.99, 0.995):
    c3 = Model0Config(d=64, p=32, r_min=lam_mag, r_max=lam_mag, theta_max=0.0, psi_init="zero")
    b3 = ComplexLinearAttention(c3).to(DEV).double()
    tau = b3.tau[0].item()
    Tl = int(min(2000, max(200, 8 * tau)))
    v = torch.randn(c3.d, device=DEV, dtype=torch.float64)
    h_al = v.expand(1, Tl, c3.d).contiguous()  # 완전 위상 정렬
    h_rd = torch.randn(1, Tl, c3.d, device=DEV, dtype=torch.float64)  # 비상관
    h_rd *= h_al.norm() / h_rd.norm()  # 진폭을 맞춰 위상 효과만 남김
    with torch.no_grad():
        na = b3(h_al, mode="recurrent")[0, -1].norm().item()
        nr = b3(h_rd, mode="recurrent")[0, -1].norm().item()
    pred = math.sqrt((1 + lam_mag) / (1 - lam_mag))
    ratio = na / nr
    base = ratio / pred if base is None else base
    print(
        f"  {lam_mag:>8.3f} {tau:>8.1f} {na:>12.2f} {nr:>12.2f} "
        f"{ratio:>9.2f} {pred:>9.2f} {ratio / pred:>10.2f}"
    )
print("  → 마지막 열이 대략 일정하면 선택성이 √τ 스케일을 따른다는 뜻.")


# ---------------------------------------------------------------------- #
banner("5. γ 스케일: ‖f‖/‖h‖ 이 감쇠 범위에 무관한가")
Tg = 2048  # γ 의 정규화는 T ≫ τ 일 때만 정확하다 (짧은 시퀀스에서는 기억이 덜 참)
print(f"  T = {Tg}, 비상관 입력, 마지막 토큰 기준")
print(f"  {'|λ| 범위':>18} {'τ_max':>8} {'use_gamma=True':>16} {'use_gamma=False':>18}")
for rmin, rmax in [(0.5, 0.6), (0.9, 0.95), (0.97, 0.98), (0.995, 0.997)]:
    row = []
    for ug in (True, False):
        c = Model0Config(d=128, p=64, r_min=rmin, r_max=rmax, use_gamma=ug)
        b = ComplexLinearAttention(c).to(DEV)
        hh = torch.randn(1, Tg, c.d, device=DEV)
        with torch.no_grad():
            f = b(hh, mode="recurrent")
        row.append((f[:, -1].norm() / hh[:, -1].norm()).item())
    tau_max = -1.0 / math.log(rmax)
    print(f"  [{rmin:.3f},{rmax:.3f}]  {tau_max:>8.1f} {row[0]:>16.3f} {row[1]:>18.3f}")


# ---------------------------------------------------------------------- #
banner("6. 역전파 — 네 파라미터 전부에 그래디언트가 흐르는가")
cfg4 = Model0Config(d=128, p=64, R=4, vocab_size=256)
model = Model0(cfg4).to(DEV)
tok = torch.randint(0, cfg4.vocab_size, (4, 128), device=DEV)
logits = model(tok)
loss = logits.float().pow(2).mean()  # 임의 스칼라 (테스크·손실은 아직 미정)
loss.backward()
for n, prm in model.named_parameters():
    g = prm.grad
    ok = g is not None and torch.isfinite(g).all()
    print(f"  {n:<20} shape={str(tuple(prm.shape)):<12} ‖g‖={g.norm().item():>10.4f}  finite={ok}")
    assert ok
print(f"\n  파라미터 수: {model.num_parameters()}   (d² = {cfg4.d**2})")


# ---------------------------------------------------------------------- #
banner("7. 재귀 깊이에 따른 진단량 궤적")
with torch.no_grad():
    _, trace = model(tok, return_trace=True)
print(
    f"  {'r':>3} {'E_D':>12} {'ρ=‖HAH‖₂':>12} {'ρ_R(Euler)':>12} "
    f"{'mean d_t':>12} {'max|a_tt|':>12}"
)
for r, s in enumerate(trace):
    if "A" in s:
        att = torch.diagonal(s["A"], dim1=-2, dim2=-1).abs().max().item()
        print(
            f"  {r:>3} {s['E_D'].mean():>12.2f} {s['rho'].mean():>12.4f} "
            f"{s['rho_R'].mean():>12.4f} {s['d_t'].mean():>12.4f} {att:>12.4f}"
        )
    else:
        print(f"  {r:>3} {s['E_D'].mean():>12.2f} {'—':>12} {'—':>12} {'—':>12} {'—':>12}")
print("\n  ρ_R > 1 인 스텝에서 E_D 가 증가해야 한다 (임계 판정량 일치 확인).")

banner("모든 검증 통과")
