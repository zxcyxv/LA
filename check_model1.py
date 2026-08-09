"""Model 1 검증.

  1. parallel ≡ recurrent
  2. Δθ ≡ 0 이면 Model 0 과 정확히 동일        ← 통제 비교의 전제
  3. 누적 위상이 실제로 '내용 가중 거리' 인가   (§4 유도의 수치 확인)
  4. 게이트가 열리면 위상차가 순서에 상수로 잠기는가 (4c 예측)
  5. 역전파가 새 파라미터에도 흐르는가
"""

import math

import torch

from model0 import ComplexLinearAttention, Model0Config
from model1 import DrivenComplexLinearAttention, Model1, Model1Config

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def banner(m):
    print(f"\n{'─' * 70}\n{m}\n{'─' * 70}")


# ------------------------------------------------------------------ #
banner("1. parallel ≡ recurrent")
cfg = Model1Config(d=64, p=32, R=2, chunk_p=8)
blk = DrivenComplexLinearAttention(cfg).to(DEV).double()
h = torch.randn(3, 40, cfg.d, device=DEV, dtype=torch.float64)
fp = blk(h, mode="parallel")
fr = blk(h, mode="recurrent")
print(f"max |f_par - f_rec| = {(fp - fr).abs().max().item():.3e}   (‖f‖∞ = {fp.abs().max():.3f})")
assert (fp - fr).abs().max() < 1e-9


# ------------------------------------------------------------------ #
banner("2. Δθ ≡ 0 이면 Model 0 과 정확히 동일한가")
c0 = Model0Config(d=64, p=32, R=2)
c1 = Model1Config(**{k: getattr(c0, k) for k in c0.__dataclass_fields__}, driven=False)
torch.manual_seed(7)
b0 = ComplexLinearAttention(c0).to(DEV).double()
torch.manual_seed(7)
b1 = DrivenComplexLinearAttention(c1).to(DEV).double()
# 공유 파라미터를 동일하게 맞춘다 (b1 은 뒤에 W_theta 등을 더 뽑으므로 앞부분만 같음)
for name in ("W_re", "W_im", "alpha_log", "theta", "psi"):
    getattr(b1, name).data.copy_(getattr(b0, name).data)
hh = torch.randn(2, 32, c0.d, device=DEV, dtype=torch.float64)
d0, d1 = b0(hh), b1(hh)
print(f"max |Model0 - Model1(driven=False)| = {(d0 - d1).abs().max().item():.3e}")
assert (d0 - d1).abs().max() < 1e-10, "Model 0 으로 환원되지 않음"

b1.cfg.driven = True
b1.s_raw.data.zero_()  # s=0 → Δθ=0 (게이트가 열려 있어도)
d1b = b1(hh)
print(f"max |Model0 - Model1(s=0)|          = {(d0 - d1b).abs().max().item():.3e}")
assert (d0 - d1b).abs().max() < 1e-10


# ------------------------------------------------------------------ #
banner("3. 누적 위상이 '내용 가중 거리' 인가")
# 채널 1개, 감쇠 없음에 가깝게. 게이트를 손으로 세팅해 blank=0, data=c 를 만든다.
c2 = Model1Config(d=8, p=1, R=1, r_min=0.999, r_max=0.999, theta_max=0.0)
b2 = DrivenComplexLinearAttention(c2).to(DEV).double()
with torch.no_grad():
    b2.theta.zero_()
    b2.s_raw.fill_(math.pi / 3)  # c = 60°
    b2.gate_bias.fill_(-30.0)  # 기본은 닫힘
    b2.W_theta.zero_()
    b2.W_theta[0, 0] = 60.0  # h 의 0번 성분이 1 이면 게이트가 열림

T = 12
hseq = torch.zeros(1, T, c2.d, device=DEV, dtype=torch.float64)
hseq[0, :, 1] = 1.0  # 공통 성분
data_at = [2, 5, 6, 9]
for t in data_at:
    hseq[0, t, 0] = 1.0  # '셀' 토큰 표시

g = b2.gate(hseq)[0, :, 0]
Phi = b2.cumulative_phase(hseq)[0, :, 0]
print("  t   :  " + " ".join(f"{t:>5}" for t in range(T)))
print("  gate:  " + " ".join(f"{v:>5.2f}" for v in g.tolist()))
print("  Φ/c :  " + " ".join(f"{v / (math.pi / 3):>5.2f}" for v in Phi.tolist()))
print(f"\n  '셀' 토큰 위치 = {data_at}")
print("  → Φ 가 그 위치에서만 정확히 1c 씩 오르면 위상이 순서 카운터가 된 것")
steps = (Phi[1:] - Phi[:-1]) / (math.pi / 3)
risen = [t + 1 for t, s in enumerate(steps.tolist()) if s > 0.5]
print(f"  실제로 오른 위치 = {risen}   (일치: {risen == data_at})")


# ------------------------------------------------------------------ #
banner("4. selective copy 배치에서 위상차가 순서에 상수로 잠기는가 (4c 예측)")
# blank=0, data=c, marker=c 로 세팅하고 §4 유도를 직접 수치 확인
n_mem, l_noise = 4, 24
c3 = Model1Config(d=8, p=1, R=1, theta_max=0.0)
b3 = DrivenComplexLinearAttention(c3).to(DEV).double()
cval = math.pi / 3
with torch.no_grad():
    b3.theta.zero_()
    b3.s_raw.fill_(cval)
    b3.gate_bias.fill_(-30.0)
    b3.W_theta.zero_()
    b3.W_theta[0, 0] = 60.0

for trial, pos in enumerate([[1, 4, 9, 17], [0, 11, 12, 23], [5, 6, 7, 8]]):
    T = l_noise + n_mem
    hh = torch.zeros(1, T, c3.d, device=DEV, dtype=torch.float64)
    for pp in pos:
        hh[0, pp, 0] = 1.0  # 데이터 토큰: 센다
    for i in range(n_mem):
        hh[0, l_noise + i, 0] = 1.0  # 마커도 센다 (c_M = c)
    Phi = b3.cumulative_phase(hh)[0, :, 0]
    raw = [(Phi[l_noise + i] - Phi[pos[i]]).item() for i in range(n_mem)]
    # 위상은 mod 2π 에서만 의미가 있다. c=π/3 이면 2π=6c 이므로 -2c ≡ 4c.
    gaps = [(v / cval) % (2 * math.pi / cval) for v in raw]
    cosv = [math.cos(v) for v in raw]
    print(
        f"  배치 {trial + 1}  데이터 위치 {str(pos):<18} → "
        f"위상차/c (mod 2π) = [{', '.join(f'{v:.2f}' for v in gaps)}]   "
        f"cos = [{', '.join(f'{v:+.3f}' for v in cosv)}]"
    )
    assert all(abs(v - 4.0) < 1e-9 for v in gaps), "4c 예측 실패"
    assert max(cosv) - min(cosv) < 1e-9, "마커별 커널값이 달라짐"
print("  → 위치가 어떻게 흩어져도 전부 4.00, cos 도 전부 동일 → §4 유도 확인")
print(f"  주의: 2π = {2 * math.pi / cval:.0f}c 이므로 순서 {2 * math.pi / cval:.0f} 이상은 되감긴다.")


# ------------------------------------------------------------------ #
banner("5. 역전파")
cfg5 = Model1Config(d=128, p=64, R=4, vocab_size=18)
m = Model1(cfg5).to(DEV)
tok = torch.randint(0, 18, (4, 36), device=DEV)
m(tok).float().pow(2).mean().backward()
for n, prm in m.named_parameters():
    ok = prm.grad is not None and torch.isfinite(prm.grad).all()
    print(f"  {n:<20} {str(tuple(prm.shape)):<12} ‖g‖={prm.grad.norm().item():>10.4f}  {ok}")
    assert ok
print(f"\n  파라미터: {m.num_parameters()}  (Model 0 대비 +{cfg5.p * cfg5.d + 2 * cfg5.p})")

banner("모든 검증 통과")
