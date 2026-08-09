"""Model 1 — 구동 해밀토니안(입력 의존 위상)을 넣은 복소 선형어텐션.

Model 0 과의 차이는 전이 연산자 한 줄뿐이다.

    Model 0:   U   = exp(-α + i θ)                 (입력 무관, LTI)
    Model 1:   U_t = exp(-α + i ω_t),  ω_t = θ + Δθ(h_t)

    Δθ_j(h) = s_j · gate(w_j·h + b_j)              gate = sigmoid | softplus

|U_t| = e^{-α} 는 그대로이므로 **입력이 조절하는 것은 감쇠가 아니라 위상 속도**다.
Mamba/GLA/RWKV 계열이 |λ| 를 입력 의존으로 만들어 '선택적 망각'을 얻는 것과 달리,
여기서는 arg λ 만 입력 의존이라 정보가 지워지지 않고 '선택적 위상 색인'만 생긴다.

핵심 귀결. 누적 위상을 Φ_t = Σ_{m≤t} ω_m 이라 하면

    Π_{m=n+1}^{t} U_m = exp(-α(t-n)) · exp(i(Φ_t - Φ_n))

즉 Model 0 의 시간 거리 (t-n)θ 자리에 **내용 가중 거리 Φ_t - Φ_n** 이 들어온다.
Δθ 가 어떤 토큰에는 크고 어떤 토큰에는 0 이면, Φ_t - Φ_n 은 그 사이에 '셀 만한'
토큰이 몇 개 있었는지를 센 값이 된다 → 위상 자체가 순서 카운터가 된다.

게이트를 sigmoid 로 두는 이유:
    유도가 요구하는 해는 Δθ(blank)=0, Δθ(data)=Δθ(marker)=c 라는 이산적 게이트다.
    softplus 는 위로 무제한이라 한 토큰이 위상을 임의로 돌릴 수 있고 정확히 0 을
    내려면 로짓이 -∞ 여야 한다. s·sigmoid 는 [0, s] 로 묶여 '셀 것인가' 게이트가 된다.

초기화:
    gate_bias = -2 → sigmoid ≈ 0.12 이므로 Δθ ≈ 0.12·s 로 작게 출발한다.
    즉 학습 시작 시점에서 Model 1 ≈ Model 0 이고, 개선이 있다면 그것은 다른 초기화가
    아니라 게이트를 학습한 결과라고 말할 수 있다. s=0 으로 두면 정확히 Model 0 이다.

수치적 이점:
    |exp(-iΦ)| = 1 이므로 q, k 를 각각 e^{-iΦ} 로 회전시키는 것만으로 병렬 전개가
    정확하게 된다. 입력 의존 *감쇠* 였다면 exp(+αn) 이 폭발해 이 트릭을 못 쓴다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from model0 import Model0Config, dirichlet_energy, euler_gain, spectral_gain  # noqa: F401


@dataclass
class Model1Config(Model0Config):
    phase_gate: str = "sigmoid"  # "sigmoid" | "softplus"
    s_max: float = math.pi  # 채널별 순서 위상 스케일 s_j 의 초기 상한
    gate_bias_init: float = -2.0  # 초기 게이트 개방도 (≈0.12)
    driven: bool = True  # False 면 Δθ≡0 → Model 0 과 정확히 동일
    tied: bool = True  # False 면 R 개 블록을 독립 파라미터로 (깊이 용량 분리)
    use_wv: bool = False  # value 사영 W_V. span{h} 제약을 풀고 '재표현'을 가능하게 한다
    use_silu_wo: bool = False  # A·V 뒤에 SiLU + W_O. 사이에 비선형이 있어야 W_O 가
    # W_V 와 합쳐지지 않는다 (없으면 f = A h (W_V W_O) 로 d² 자유도에 2d² 파라미터)
    gamma_split: bool = True  # γ 를 q,k 에 √γ 씩 배분 → (q,k) 가 라그랑지안 그래프.
    # 커널은 곱 q̄k 만 보므로 함수는 불변(실측 1.7e-16). 기하학적 정합성만 얻는다.


class DrivenComplexLinearAttention(nn.Module):
    """h ∈ R^{B,T,d} → f ∈ R^{B,T,d}. 전이 위상이 입력 의존."""

    def __init__(self, cfg: Model1Config):
        super().__init__()
        self.cfg = cfg
        d, p = cfg.d, cfg.p

        std = cfg.w_scale / math.sqrt(2.0 * d)
        self.W_re = nn.Parameter(torch.randn(p, d) * std)
        self.W_im = nn.Parameter(torch.randn(p, d) * std)

        mag = torch.empty(p).uniform_(cfg.r_min, cfg.r_max)
        self.alpha_log = nn.Parameter(torch.log(-torch.log(mag)))
        self.theta = nn.Parameter(torch.empty(p).uniform_(0.0, cfg.theta_max))

        psi = (
            torch.zeros(p)
            if cfg.psi_init == "zero"
            else torch.empty(p).uniform_(-math.pi, math.pi)
        )
        self.psi = nn.Parameter(psi)

        # --- 구동 항 (Model 0 대비 추가되는 전부) ---
        self.W_theta = nn.Parameter(torch.randn(p, d) / math.sqrt(d))
        self.s_raw = nn.Parameter(torch.empty(p).uniform_(0.0, cfg.s_max))
        self.gate_bias = nn.Parameter(torch.full((p,), cfg.gate_bias_init))

        # value 사영. 없으면 블록 출력이 span{h_1..h_t} 안에 갇힌다 (대화내역2 §8).
        self.W_V = (
            nn.Parameter(torch.randn(d, d) / math.sqrt(d)) if cfg.use_wv else None
        )
        self.W_O = (
            nn.Parameter(torch.randn(d, d) / math.sqrt(d)) if cfg.use_silu_wo else None
        )

    # ------------------------------------------------------------------ #
    @property
    def alpha(self) -> Tensor:
        return torch.exp(self.alpha_log)

    @property
    def gamma(self) -> Tensor:
        if not self.cfg.use_gamma:
            return torch.ones_like(self.alpha)
        return torch.sqrt(-torch.expm1(-2.0 * self.alpha))

    @property
    def tau(self) -> Tensor:
        return 1.0 / self.alpha

    def gate(self, h: Tensor) -> Tensor:
        """g ∈ [0,1]^{B,T,p} — 이 토큰을 위상 축에서 '셀' 것인가."""
        logit = torch.einsum("btd,pd->btp", h, self.W_theta) + self.gate_bias
        if self.cfg.phase_gate == "softplus":
            return torch.nn.functional.softplus(logit)
        return torch.sigmoid(logit)

    def phase_increment(self, h: Tensor) -> Tensor:
        """ω_t = θ + Δθ(h_t) ∈ R^{B,T,p} — 이 스텝의 회전각."""
        if not self.cfg.driven:
            return self.theta.expand(h.shape[0], h.shape[1], -1)
        return self.theta + self.s_raw * self.gate(h)

    def cumulative_phase(self, h: Tensor) -> Tensor:
        """Φ_t = Σ_{m≤t} ω_m. float64 로 누적한 뒤 2π 로 감아 정밀도를 지킨다."""
        omega = self.phase_increment(h)
        Phi = torch.cumsum(omega.double(), dim=1)
        return torch.remainder(Phi, 2 * math.pi).to(omega.dtype)

    def project(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        W = torch.complex(self.W_re, self.W_im)
        z = torch.einsum("btd,pd->btp", h.to(W.dtype), W)
        half = 0.5 * self.psi
        rot = torch.polar(torch.ones_like(half), half)
        if self.cfg.gamma_split:
            # γ 를 √γ 씩 나눠 걸면 (q,k) 가 심플렉토모피즘의 그래프가 되어
            # (ω⊖ω) 가 정확히 0 이 된다. 커널은 곱 q̄k 만 보므로 함수는 불변.
            g = torch.sqrt(self.gamma).to(z.dtype)
            q = z * rot * g
            k = z * rot.conj() * g
        else:
            q = z * rot
            k = z * rot.conj() * self.gamma.to(z.dtype)
        return z, q, k

    # ------------------------------------------------------------------ #
    def forward(self, h: Tensor, mode: str = "parallel", return_A: bool = False):
        if mode == "parallel":
            A = self.attention_matrix(h)
            v = h if self.W_V is None else h @ self.W_V
            f = torch.einsum("btn,bnd->btd", A, v)
            if self.W_O is not None:
                f = torch.nn.functional.silu(f) @ self.W_O
            return (f, A) if return_A else f
        if mode == "recurrent":
            if return_A:
                raise ValueError("recurrent 경로는 A 를 만들지 않는다.")
            return self._forward_recurrent(h)
        raise ValueError(f"unknown mode: {mode}")

    def attention_matrix(self, h: Tensor) -> Tensor:
        """a_tn = p^{-1/2} Σ_j e^{-α_j(t-n)} · Re[ conj(q'_{t,j}) k'_{n,j} ],  q' = q e^{-iΦ}

        Φ 를 q, k 에 미리 흡수시키면 남는 커널이 실수 감쇠뿐이라 전 과정이 실수 연산이 된다.
        |e^{-iΦ}|=1 이라 이 인수분해는 수치적으로 정확하다 (감쇠였다면 폭발한다).
        """
        _, q, k = self.project(h)
        Phi = self.cumulative_phase(h)  # (B,T,p)
        rot = torch.polar(torch.ones_like(Phi), -Phi)  # e^{-iΦ}
        qs, ks = q * rot, k * rot

        qr, qi = qs.real, qs.imag
        kr, ki = ks.real, ks.imag

        B, T, p = qr.shape
        idx = torch.arange(T, device=qr.device)
        delta = idx[:, None] - idx[None, :]
        causal = delta >= 0
        dvec = delta.clamp(min=0).to(self.alpha.dtype)  # (T,T)

        A = qr.new_zeros(B, T, T)
        for j0 in range(0, p, self.cfg.chunk_p):
            j1 = min(j0 + self.cfg.chunk_p, p)
            decay = torch.exp(-dvec[:, :, None] * self.alpha[None, None, j0:j1])  # (T,T,c)
            inner = (
                qr[:, :, None, j0:j1] * kr[:, None, :, j0:j1]
                + qi[:, :, None, j0:j1] * ki[:, None, :, j0:j1]
            )  # (B,T,T,c)
            A = A + (inner * decay).sum(-1)

        return A * causal.to(A.dtype) / math.sqrt(p)

    def _forward_recurrent(self, h: Tensor) -> Tensor:
        _, q, k = self.project(h)
        omega = self.phase_increment(h)  # (B,T,p)
        B, T, p = q.shape
        d = h.shape[-1]

        hc = h.to(q.dtype)
        S = q.new_zeros(B, p, d)
        alpha = self.alpha[None, :, None]
        out = []
        for t in range(T):
            U = torch.polar(torch.exp(-alpha).expand(B, p, 1), omega[:, t].unsqueeze(-1))
            S = U * S + k[:, t].unsqueeze(-1) * hc[:, t].unsqueeze(1)
            out.append((q[:, t].conj().unsqueeze(-1) * S).sum(1).real)
        return torch.stack(out, dim=1) / math.sqrt(p)


class Model1(nn.Module):
    """embedding → 같은 구동 블록 R 번 Euler 적분 → tied head."""

    def __init__(self, cfg: Model1Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d)
        nn.init.normal_(self.embed.weight, std=1.0)
        if cfg.tied:
            self.block = DrivenComplexLinearAttention(cfg)
            self.blocks = None
        else:
            self.blocks = nn.ModuleList(
                [DrivenComplexLinearAttention(cfg) for _ in range(cfg.R)]
            )
            self.block = self.blocks[0]
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / cfg.d))

    def forward(self, tokens: Tensor, mode: str = "parallel", return_trace: bool = False):
        h = self.embed(tokens)
        trace = []
        for _r in range(self.cfg.R):
            blk = self.block if self.blocks is None else self.blocks[_r]
            if return_trace:
                f, A = blk(h, mode="parallel", return_A=True)
                trace.append(
                    {
                        "h": h.detach(),
                        "A": A.detach(),
                        "d_t": A.detach().sum(-1),
                        "E_D": dirichlet_energy(h.detach()),
                        "rho": spectral_gain(A.detach()),
                        "rho_R": euler_gain(A.detach(), self.cfg.R),
                        "dtheta": (blk.phase_increment(h) - blk.theta).detach(),
                    }
                )
            else:
                f = blk(h, mode=mode)
            h = h + f / self.cfg.R
        if return_trace:
            trace.append({"h": h.detach(), "E_D": dirichlet_energy(h.detach())})
        logits = self.logit_scale * (h @ self.embed.weight.t())
        return (logits, trace) if return_trace else logits

    def num_parameters(self) -> dict[str, int]:
        mods = [self.block] if self.blocks is None else list(self.blocks)
        blk = sum(x.numel() for m in mods for x in m.parameters())
        emb = self.embed.weight.numel()
        return {"block": blk, "embedding": emb, "total": blk + emb}
