"""Mamba-3 순수 PyTorch 재현의 핵심 불변량 검사.

등가성만으로는 병렬·순차 경로가 함께 틀리는 버그를 잡지 못하므로 다음을 별도로 본다.

1. A 는 입력 의존이고 항상 음수다.
2. causal 감쇠 행렬은 극단적인 누적 감쇠에서도 유한하고 상삼각이 정확히 0 이다.
3. 안정적인 하삼각 재귀가 안전한 범위에서 누적합 식과 일치한다.
4. quadratic 경로와 recurrent 경로가 일치한다.
5. 역전파가 유한하다.
"""

import torch
import torch.nn.functional as F

from mamba3_ref import Mamba3Config, Mamba3Layer, _causal_decay_matrix, heavy_tail


torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def banner(text: str):
    print(f"\n{'─' * 70}\n{text}\n{'─' * 70}")


banner("1. A 입력 의존성·음수 불변량")
cfg = Mamba3Config(d_model=64, d_state=16, headdim=16, n_layer=1)
layer = Mamba3Layer(cfg).to(DEV)
u = torch.randn(3, 48, cfg.d_model, device=DEV)
parts = layer.in_proj(u).split(
    [layer.d_inner, layer.d_inner, cfg.d_state, cfg.d_state,
     layer.nheads, layer.nheads, layer.nheads, layer.n_ang],
    dim=-1,
)
dd_dt, dd_A = parts[4], parts[5]
A = (-heavy_tail(dd_A.float())).clamp(max=-cfg.A_floor)
DT = F.softplus(dd_dt + layer.dt_bias)
print(f"A range=[{A.min().item():.4g}, {A.max().item():.4g}]  unique={A.unique().numel()}")
print(f"exp(A·DT) max={torch.exp(A * DT).max().item():.9f}")
assert bool((A < 0).all())
assert A.unique().numel() > 1
assert torch.exp(A * DT).max() < 1


banner("2. 극단 감쇠에서도 overflow/NaN 없음")
# 누적합 차 방식은 미래 영역에서 exp(255)를 계산해 float32 overflow 한다.
extreme = -torch.ones(2, 256, 3, device=DEV)
dec_extreme = _causal_decay_matrix(extreme)
upper = torch.ones(256, 256, device=DEV, dtype=torch.bool).triu(1)
print(f"finite={bool(torch.isfinite(dec_extreme).all())}  "
      f"max={dec_extreme.max().item():.1f}  upper max={dec_extreme[..., upper].abs().max().item():.1f}")
assert bool(torch.isfinite(dec_extreme).all())
assert dec_extreme.max() <= 1
assert dec_extreme[..., upper].abs().max() == 0


banner("3. 누적합 닫힌 형태와 일치")
adt = -0.2 * torch.rand(2, 24, 4, device=DEV, dtype=torch.float64)
stable = _causal_decay_matrix(adt)
cum = torch.cumsum(adt, dim=1).transpose(1, 2)
log_ref = cum.unsqueeze(-1) - cum.unsqueeze(-2)
mask = torch.ones(24, 24, device=DEV, dtype=torch.bool).tril()
ref = torch.exp(log_ref.masked_fill(~mask, float("-inf")))
err = (stable - ref).abs().max().item()
print(f"max |stable-reference| = {err:.3e}")
assert err < 1e-12


banner("4. quadratic ≡ recurrent")
layer.eval()
u = torch.randn(2, 40, cfg.d_model, device=DEV)
layer.use_quadratic = True
yq = layer(u)
layer.use_quadratic = False
yr = layer(u)
rel = (yq - yr).norm() / yr.norm().clamp_min(1e-12)
print(f"relative error = {rel.item():.3e}")
assert rel < 2e-6


banner("5. 역전파 유한성")
layer.use_quadratic = True
u = torch.randn(2, 64, cfg.d_model, device=DEV, requires_grad=True)
loss = layer(u).square().mean()
loss.backward()
assert torch.isfinite(u.grad).all()
for name, param in layer.named_parameters():
    if param.grad is not None:
        assert torch.isfinite(param.grad).all(), name
print(f"loss={loss.item():.6f}  input grad norm={u.grad.norm().item():.6f}")

banner("모든 검증 통과")
