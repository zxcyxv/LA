"""Mamba-3 SISO 레이어의 순수 PyTorch 재현 (공식 커널은 Triton/CuteDSL 이라 대체).

공식 구현(state-spaces/mamba @ mamba_ssm/modules/mamba3.py)에서 확인한 파라미터화를
그대로 따른다:

    in_proj  → [z, x, B, C, dd_dt, dd_A, trap, angles]
    _A       = -heavy_tail(dd_A),  clamp(max=-A_floor)
    DT       = softplus(dd_dt + dt_bias)
    ADT      = _A * DT
    angle_inc = tanh(angle_proj) * DT * π          ← mamba3_mimo_rotary_step.py:75
    Φ        = cumsum(angle_inc) mod 2π            ← mamba3_siso_combined.py:315
    B, C     = RMSNorm(B) + B_bias,  RMSNorm(C) + C_bias
    RoPE trick: 누적 회전 Φ 를 B(key) 와 C(query) 에 적용 (논문 Prop 3)

우리 Model 1 과의 구조적 차이(이 실험이 겨냥하는 지점):
    · 증분이 tanh → 부호 있음. 위상이 뒤로 갈 수 있어 '단조 시계'가 아니다.
      우리는 s·sigmoid → [0, s] 비음수 단조.
    · 증분이 Δ_t(입력 의존 스텝)와 곱해진다. 두 입력 의존량의 곱.
    · rope_fraction=0.5 → 상태 차원의 절반만 회전한다. 나머지 절반은 무회전.
    · 감쇠 ADT 도 입력 의존. 우리는 고정 α.
    · Llama 식으로 SwiGLU 블록과 교대 배치 + pre-norm (논문 §3.4).

단순화(명시):
    · exponential-Euler (Prop 2/3) 를 쓴다. 논문 본체의 exponential-trapezoidal
      3항 재귀는 생략 — 위상 메커니즘 자체는 동일하고 trap 항은 입력 컨볼루션에
      해당한다.
    · MIMO 미사용 (논문 기본값도 SISO).
    · 청크 커널 대신 시퀀스 루프 (T ≈ 40 이라 무의미한 비용).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def heavy_tail(x: Tensor) -> Tensor:
    """f(x) = 1+x (x≥0), 1/(1-x) (x<0). 공식 구현 그대로."""
    return x.clamp_min(0) + torch.reciprocal(1 - x.clamp_max(0))


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


@dataclass
class Mamba3Config:
    d_model: int = 128
    d_state: int = 32
    expand: int = 1
    headdim: int = 32
    n_layer: int = 4
    vocab_size: int = 18
    rope_fraction: float = 0.5
    A_floor: float = 1e-4
    use_quadratic: bool = True
    dt_min: float = 1e-3
    dt_max: float = 1e-1


class Mamba3Layer(nn.Module):
    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.cfg = cfg
        d_inner = cfg.d_model * cfg.expand
        self.d_inner = d_inner
        self.nheads = d_inner // cfg.headdim
        self.headdim = cfg.headdim
        self.d_state = cfg.d_state

        # rope_fraction: 상태 차원의 일부만 회전한다
        split = int(cfg.d_state * cfg.rope_fraction)
        split -= split % 2
        self.n_ang = split // 2
        self.rot_dim = split  # 앞쪽 rot_dim 차원만 회전

        d_in_proj = 2 * d_inner + 2 * cfg.d_state + 3 * self.nheads + self.n_ang
        self.in_proj = nn.Linear(cfg.d_model, d_in_proj, bias=False)
        self.out_proj = nn.Linear(d_inner, cfg.d_model, bias=False)

        _dt = torch.exp(
            torch.rand(self.nheads) * (math.log(cfg.dt_max) - math.log(cfg.dt_min))
            + math.log(cfg.dt_min)
        )
        self.dt_bias = nn.Parameter(_dt + torch.log(-torch.expm1(-_dt)))
        self.B_bias = nn.Parameter(torch.ones(cfg.d_state))
        self.C_bias = nn.Parameter(torch.ones(cfg.d_state))
        self.B_norm = RMSNorm(cfg.d_state)
        self.C_norm = RMSNorm(cfg.d_state)
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = RMSNorm(d_inner)
        self.use_quadratic = cfg.use_quadratic

    def _rope(self, v: Tensor, phi: Tensor) -> Tensor:
        """v: (B,T,d_state), phi: (B,T,n_ang) → 앞쪽 rot_dim 차원에 2D 회전 적용."""
        r, rest = v[..., : self.rot_dim], v[..., self.rot_dim :]
        r = r.reshape(*r.shape[:-1], self.n_ang, 2)
        c, s = torch.cos(phi).unsqueeze(-1), torch.sin(phi).unsqueeze(-1)
        x0, x1 = r[..., 0:1], r[..., 1:2]
        rot = torch.cat([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1)
        return torch.cat([rot.reshape(*v.shape[:-1], self.rot_dim), rest], dim=-1)

    def forward(self, u: Tensor, return_phase: bool = False):
        B_, T, _ = u.shape
        cfg = self.cfg
        parts = self.in_proj(u).split(
            [self.d_inner, self.d_inner, cfg.d_state, cfg.d_state,
             self.nheads, self.nheads, self.nheads, self.n_ang],
            dim=-1,
        )
        z, x, Bp, Cp, dd_dt, dd_A, _trap, ang = parts

        # 부호를 **먼저** 뒤집고 그 다음에 clamp 한다. 공식 구현이 두 문장으로 나눠
        # 쓴 이유가 이것이다 (mamba_ssm/modules/mamba3.py):
        #     _A = -heavy_tail_activation(dd_A.to(torch.float32))
        #     _A = torch.clamp(_A, max=-self.A_floor)
        # 한 줄로 `-heavy_tail(x).clamp(max=-A_floor)` 라 쓰면 파이썬 우선순위상
        # clamp 가 먼저 걸린다. heavy_tail 은 항상 양수이므로 전 원소가 -A_floor 로
        # 뭉개지고, 부호를 뒤집으면 A ≡ +A_floor 상수가 된다 — 입력 의존성이 사라지고
        # ADT>0 이 되어 감쇠가 아니라 증폭이 된다. 괄호가 없으면 안 된다.
        _A = (-heavy_tail(dd_A.float())).clamp(max=-cfg.A_floor)  # (B,T,H)
        DT = F.softplus(dd_dt + self.dt_bias)  # (B,T,H)
        ADT = _A * DT

        # 위상: tanh 로 부호 있는 증분, DT 와 곱한 뒤 누적, mod 2π
        # DT 는 헤드별이지만 angle 은 헤드 공유이므로 헤드 평균 DT 를 쓴다
        inc = torch.tanh(ang) * DT.mean(-1, keepdim=True) * math.pi  # (B,T,n_ang)
        phi = torch.remainder(torch.cumsum(inc.double(), dim=1), 2 * math.pi).float()

        Bv = self.B_norm(Bp) + self.B_bias
        Cv = self.C_norm(Cp) + self.C_bias
        Bv = self._rope(Bv, phi)  # key
        Cv = self._rope(Cv, phi)  # query

        xh = x.view(B_, T, self.nheads, self.headdim)

        if self.use_quadratic:
            # Mamba-2 식 이차 dual form. ADT<0 이므로 cumA 는 단조 감소하고
            # exp(cumA[t]-cumA[n]) ≤ 1 (t≥n) 이라 수치적으로 안전하다.
            cumA = torch.cumsum(ADT, dim=1)  # (B,T,H)
            dec = torch.exp(cumA.transpose(1, 2).unsqueeze(-1)
                            - cumA.transpose(1, 2).unsqueeze(-2))  # (B,H,T,T)
            score = torch.einsum("btn,bsn->bts", Cv, Bv)  # (B,T,T) : query t · key s
            mask = torch.ones(T, T, device=u.device, dtype=torch.bool).tril()
            w = dec * score.unsqueeze(1) * DT.transpose(1, 2).unsqueeze(-2)
            w = w * mask
            y = torch.einsum("bhts,bshp->bthp", w, xh)
            y = y + self.D[None, None, :, None] * xh
        else:
            decay = torch.exp(ADT)
            h = x.new_zeros(B_, self.nheads, self.headdim, cfg.d_state)
            ys = []
            for t in range(T):
                h = decay[:, t, :, None, None] * h + (
                    DT[:, t, :, None, None]
                    * xh[:, t, :, :, None]
                    * Bv[:, t, None, None, :]
                )
                ys.append((h * Cv[:, t, None, None, :]).sum(-1)
                          + self.D[None, :, None] * xh[:, t])
            y = torch.stack(ys, 1)
        y = y.reshape(B_, T, self.d_inner)
        y = self.norm(y * F.silu(z))
        out = self.out_proj(y)
        return (out, phi) if return_phase else out


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: int = 2):
        super().__init__()
        hid = d * mult
        self.w1 = nn.Linear(d, hid, bias=False)
        self.w2 = nn.Linear(d, hid, bias=False)
        self.w3 = nn.Linear(hid, d, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Mamba3Model(nn.Module):
    """Llama 식: [pre-norm → Mamba3 → residual] + [pre-norm → SwiGLU → residual] 반복."""

    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList([RMSNorm(cfg.d_model), Mamba3Layer(cfg),
                               RMSNorm(cfg.d_model), SwiGLU(cfg.d_model)])
                for _ in range(cfg.n_layer)
            ]
        )
        self.norm_f = RMSNorm(cfg.d_model)

    def forward(self, tokens: Tensor, return_phase: bool = False):
        h = self.embed(tokens)
        phases = []
        for n1, mix, n2, mlp in self.layers:
            if return_phase:
                o, phi = mix(n1(h), return_phase=True)
                phases.append(phi)
            else:
                o = mix(n1(h))
            h = h + o
            h = h + mlp(n2(h))
        logits = self.norm_f(h) @ self.embed.weight.t()
        return (logits, phases) if return_phase else logits

    def num_parameters(self):
        return {"total": sum(p.numel() for p in self.parameters())}
