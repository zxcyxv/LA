"""Model 0 — 복소 선형어텐션 단일 블록의 재귀 전개.

    z_t = W_C h_t                        (C^p)
    q_t = e^{+i psi/2} ⊙ z_t             (C^p)
    k_t = e^{-i psi/2} ⊙ z_t             (C^p)
    S_t = Λ S_{t-1} + (γ ⊙ k_t) h_t^T    (C^{p×d}),  Λ = diag(λ),  λ_j = e^{-α_j + i θ_j}
    f_t = p^{-1/2} · Re[ q_t^† S_t ]     (R^d)
    h^{(r+1)} = h^{(r)} + f^{(r)} / R    (weight-tied, r = 0..R-1)

원 설계(대화내역2 §12)와 달라진 두 가지:

  1. 시간 스케일 1/t 제거.
  2. 유니터리 전이 e^{-iθ} → 감쇠 전이 e^{-α+iθ}, α = exp(alpha_log) > 0.

     |λ| < 1 이므로 상태가 유한한 유효 기억 길이 τ_j = 1/α_j 를 갖는다.
     이제 "비정렬 잡음은 사라지고 정렬된 패턴만 남는" 성질은 t 에 의존하는
     측도 정규화가 아니라 채널별 감쇠에서 나온다:

         비상관 위상  →  Σ_n λ^Δ (·)  의 크기 ~ O( (1-|λ|²)^{-1/2} )
         정렬된 위상  →                       ~ O( (1-|λ|)^{-1}   )

     두 스케일의 비가 (1-|λ|)^{-1/2} = O(√τ) 이므로, 선택성은 유지되면서
     t 에 대해 시간불변(time-invariant)이 된다. 1/t 를 쓸 때와 달리
     시퀀스 위치에 따라 이득이 변하지 않으므로 학습·추론이 일관된다.

  γ_j = sqrt(1 - |λ_j|²) 는 별도의 정규화 레이어가 아니라 위 감쇠와 짝이 되는
  입력 스케일이다 (LRU 계열의 표준 파라미터화). 이것이 있어야 α 가 채널마다
  달라도 E‖S_{t,j}‖ 가 α 에 무관하게 O(1) 로 유지된다. use_gamma=False 로 끌 수 있다.

MLP / softmax / norm / gate / bias / positional embedding 없음.
학습 파라미터는 {W_C, α, θ, ψ} 뿐이고, 재귀 깊이 R 을 늘려도 늘지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class Model0Config:
    d: int = 128  # 실수 hidden dimension
    p: int = 64  # 복소 채널 수 (기본 d/2)
    R: int = 4  # 재귀 깊이 (weight-tied)
    vocab_size: int = 256

    # λ_j = exp(-α_j + i θ_j) 초기화
    r_min: float = 0.90  # |λ| 초기 하한  (짧은 기억, τ ≈ 9.5)
    r_max: float = 0.999  # |λ| 초기 상한  (긴 기억,   τ ≈ 1000)
    theta_max: float = 2 * math.pi  # θ 초기 범위 [0, theta_max)

    # ψ_j: 채널의 흥분/억제 성질. cos ψ_j > 0 이면 진폭 의존적 흥분,
    # cos ψ_j < 0 이면 진폭 의존적 억제 (자기항 a_tt 의 부호).
    psi_init: str = "uniform"  # "uniform" | "zero"

    use_gamma: bool = True  # γ = sqrt(1-|λ|²) 입력 스케일

    # W_C 초기 이득 배율. norm 이 없고 자기항이 3차이므로 초기화에 임계 스케일이 있다.
    # d=128,p=64,R=4,T=128 에서 실측한 순전파 1회의 디리클레 에너지 증폭률:
    #     0.25 → ×1.03    0.5 → ×1.18    0.71 → ×1.56    1.0 → ×5.1    1.5 → ×7e5 (발산)
    # 기본값은 임계점 아래(약한 소산~중립)에서 출발하도록 잡는다. 흥분 전환은
    # 초기화가 아니라 학습으로 얻어야 관측이 의미 있다.
    w_scale: float = 0.5

    chunk_p: int = 16  # parallel 경로의 채널 청크 (메모리 상한 조절)


class ComplexLinearAttention(nn.Module):
    """단일 블록. h ∈ R^{B,T,d} → f ∈ R^{B,T,d}."""

    def __init__(self, cfg: Model0Config):
        super().__init__()
        self.cfg = cfg
        d, p = cfg.d, cfg.p

        # W_C ∈ C^{p×d}. 복소 파라미터를 직접 두지 않고 실수 두 장으로 유지한다
        # (옵티마이저/초기화/그래디언트 클리핑이 전부 실수 경로로 돌게).
        std = cfg.w_scale / math.sqrt(2.0 * d)  # E|z_j|² ≈ w_scale²·‖h‖²/d
        self.W_re = nn.Parameter(torch.randn(p, d) * std)
        self.W_im = nn.Parameter(torch.randn(p, d) * std)

        # α_j = exp(alpha_log_j) > 0,  |λ_j| = exp(-α_j)
        mag = torch.empty(p).uniform_(cfg.r_min, cfg.r_max)
        alpha = -torch.log(mag)
        self.alpha_log = nn.Parameter(torch.log(alpha))

        self.theta = nn.Parameter(torch.empty(p).uniform_(0.0, cfg.theta_max))

        if cfg.psi_init == "zero":
            psi = torch.zeros(p)
        else:  # 절반은 흥분성, 절반은 억제성으로 출발
            psi = torch.empty(p).uniform_(-math.pi, math.pi)
        self.psi = nn.Parameter(psi)

    # ------------------------------------------------------------------ #
    # 파라미터에서 유도되는 값들
    # ------------------------------------------------------------------ #
    @property
    def alpha(self) -> Tensor:  # (p,) 양수 감쇠율
        return torch.exp(self.alpha_log)

    @property
    def log_lam(self) -> Tensor:  # (p,) 복소, λ = exp(log_lam)
        return torch.complex(-self.alpha, self.theta)

    @property
    def lam(self) -> Tensor:  # (p,) 복소 전이 고유값
        return torch.exp(self.log_lam)

    @property
    def gamma(self) -> Tensor:  # (p,) 실수 입력 스케일
        if not self.cfg.use_gamma:
            return torch.ones_like(self.alpha)
        return torch.sqrt(-torch.expm1(-2.0 * self.alpha))  # sqrt(1 - e^{-2α})

    @property
    def tau(self) -> Tensor:  # (p,) 채널별 유효 기억 길이 1/α
        return 1.0 / self.alpha

    def project(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """h → (z, q, k). z = W_C h, q = e^{+iψ/2}z, k = γ ⊙ e^{-iψ/2}z."""
        W = torch.complex(self.W_re, self.W_im)  # (p,d)
        z = torch.einsum("btd,pd->btp", h.to(W.dtype), W)  # (B,T,p)
        half = 0.5 * self.psi
        rot = torch.polar(torch.ones_like(half), half)  # e^{+iψ/2}
        q = z * rot
        k = z * rot.conj() * self.gamma.to(z.dtype)
        return z, q, k

    # ------------------------------------------------------------------ #
    # 두 가지 계산 경로 (수학적으로 동일)
    # ------------------------------------------------------------------ #
    def forward(
        self,
        h: Tensor,
        mode: str = "parallel",
        return_A: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """h: (B,T,d) → f: (B,T,d).

        mode="parallel"   : A ∈ R^{B,T,T} 를 명시적으로 만들고 f = A h. O(T²p).
        mode="recurrent"  : S_t 를 직접 굴린다. O(T·p·d), 시퀀스 루프.
        """
        if mode == "parallel":
            A = self.attention_matrix(h)
            f = torch.einsum("btn,bnd->btd", A, h)
            return (f, A) if return_A else f
        if mode == "recurrent":
            if return_A:
                raise ValueError("recurrent 경로는 A 를 만들지 않는다. mode='parallel' 사용.")
            return self._forward_recurrent(h)
        raise ValueError(f"unknown mode: {mode}")

    def attention_matrix(self, h: Tensor) -> Tensor:
        """A ∈ R^{B,T,T}, 인과 마스킹된 signed edge weight.

            a_tn = p^{-1/2} Σ_j γ_j |z_tj||z_nj| e^{-α_j Δ}
                              cos( arg z_nj - arg z_tj - ψ_j - Δ θ_j ),   Δ = t-n ≥ 0
        """
        _, q, k = self.project(h)
        B, T, p = q.shape
        dev = q.device

        idx = torch.arange(T, device=dev)
        delta = idx[:, None] - idx[None, :]  # (T,T)
        causal = delta >= 0
        delta_c = delta.clamp(min=0)

        dvec = idx.to(self.alpha.dtype)  # (T,)
        A = q.new_zeros(B, T, T, dtype=q.real.dtype)

        # 채널을 청크로 잘라 (B,T,T,chunk) 이상은 절대 만들지 않는다.
        for j0 in range(0, p, self.cfg.chunk_p):
            j1 = min(j0 + self.cfg.chunk_p, p)
            # λ_j^Δ  for Δ = 0..T-1  → (T, c);  α>0 이라 큰 Δ 에서 0 으로 언더플로만 함
            lam_pow = torch.exp(dvec[:, None] * self.log_lam[None, j0:j1])
            W = lam_pow[delta_c]  # (T,T,c)
            term = q[:, :, None, j0:j1].conj() * k[:, None, :, j0:j1] * W
            A = A + term.real.sum(-1)

        A = A * causal.to(A.dtype)
        return A / math.sqrt(p)

    def _forward_recurrent(self, h: Tensor) -> Tensor:
        _, q, k = self.project(h)
        B, T, p = q.shape
        d = h.shape[-1]

        lam = self.lam.view(1, p, 1)  # (1,p,1)
        hc = h.to(q.dtype)  # value 는 h 자신 (W_V 없음)
        S = q.new_zeros(B, p, d)
        out = []
        for t in range(T):
            S = lam * S + k[:, t].unsqueeze(-1) * hc[:, t].unsqueeze(1)  # (B,p,d)
            f_t = (q[:, t].conj().unsqueeze(-1) * S).sum(1).real
            out.append(f_t)
        return torch.stack(out, dim=1) / math.sqrt(p)

    @torch.no_grad()
    def step(self, h_t: Tensor, S: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """추론용 1-스텝. h_t: (B,d), S: (B,p,d) 복소 → (f_t, S_new)."""
        _, q, k = self.project(h_t.unsqueeze(1))
        q, k = q[:, 0], k[:, 0]
        if S is None:
            S = q.new_zeros(h_t.shape[0], self.cfg.p, h_t.shape[-1])
        S = self.lam.view(1, -1, 1) * S + k.unsqueeze(-1) * h_t.to(q.dtype).unsqueeze(1)
        f_t = (q.conj().unsqueeze(-1) * S).sum(1).real / math.sqrt(self.cfg.p)
        return f_t, S


class Model0(nn.Module):
    """embedding → 같은 블록 R 번 Euler 적분 → tied head.

    손실·테스크·정규화는 여기 넣지 않는다. forward 는 logits 만 낸다.
    """

    def __init__(self, cfg: Model0Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d)
        nn.init.normal_(self.embed.weight, std=1.0)  # ‖h‖ ≈ √d
        self.block = ComplexLinearAttention(cfg)
        # tied readout 의 스칼라 스케일. ‖h‖≈√d, ‖E_j‖≈√d 이므로 h·E_j 의 표준편차가
        # √d 가 되어 그대로 쓰면 softmax 가 초기부터 포화한다 (초기 loss ≈ 120 ≫ ln V).
        # 1/d 로 시작해 학습이 confidence 를 스스로 올리게 둔다. 정규화 레이어가 아니라
        # 출력 사영의 스케일이므로 파라미터 하나로만 둔다.
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / cfg.d))

    def forward(
        self,
        tokens: Tensor,
        mode: str = "parallel",
        return_trace: bool = False,
    ):
        """tokens: (B,T) long → logits: (B,T,vocab)."""
        h = self.embed(tokens)
        trace = []
        for _ in range(self.cfg.R):
            if return_trace:
                f, A = self.block(h, mode="parallel", return_A=True)
                trace.append(
                    {
                        "h": h.detach(),
                        "A": A.detach(),
                        "d_t": A.detach().sum(-1),  # 토큰별 순간 흥분도
                        "E_D": dirichlet_energy(h.detach()),
                        "rho": spectral_gain(A.detach()),
                        "rho_R": euler_gain(A.detach(), self.cfg.R),
                    }
                )
            else:
                f = self.block(h, mode=mode)
            h = h + f / self.cfg.R
        if return_trace:
            trace.append({"h": h.detach(), "E_D": dirichlet_energy(h.detach())})
        logits = self.logit_scale * (h @ self.embed.weight.t())  # tied
        return (logits, trace) if return_trace else logits

    def num_parameters(self) -> dict[str, int]:
        blk = sum(x.numel() for x in self.block.parameters())
        emb = self.embed.weight.numel()
        return {"block": blk, "embedding": emb, "total": blk + emb}


# ---------------------------------------------------------------------- #
# 진단량 (학습 없이도 그대로 관측 가능한 네 가지)
# ---------------------------------------------------------------------- #
def dirichlet_energy(H: Tensor) -> Tensor:
    """E_K(H) = ‖ (I - 11^T/T) H ‖_F²  — 완전그래프 디리클레 에너지. (B,T,d) → (B,)"""
    Hc = H - H.mean(dim=1, keepdim=True)
    return (Hc**2).sum(dim=(1, 2))


def spectral_gain(A: Tensor) -> Tensor:
    """ρ = ‖ H A H ‖₂ — 어텐션 혼합 자체의 이득. (B,T,T) → (B,)

    주의: 대화내역2 §14 의 "ρ<1 소산 / ρ>1 증폭" 기준은 출력이 O = A V 일 때의
    것이다. Model 0 은 residual Euler 적분 h ← h + f/R 이므로, 디리클레 에너지가
    실제로 늘어나는지를 결정하는 것은 아래 euler_gain 쪽이다.
    """
    T = A.shape[-1]
    C = torch.eye(T, device=A.device, dtype=A.dtype) - 1.0 / T
    return torch.linalg.matrix_norm(C @ A @ C, ord=2)


def euler_gain(A: Tensor, R: int) -> Tensor:
    """ρ_R = ‖ H (I + A/R) H ‖₂ — 한 재귀 스텝의 실제 임계 판정량.

    <1 이면 그 스텝에서 토큰 간 대비가 줄고(소산), >1 이면 늘어난다(선택적 증폭).
    "발화 문턱"에 대응하는 사건은 ρ_R 이 1 을 가로지르는 순간이다.
    """
    T = A.shape[-1]
    I = torch.eye(T, device=A.device, dtype=A.dtype)
    C = I - 1.0 / T
    return torch.linalg.matrix_norm(C @ (I + A / R) @ C, ord=2)


def self_gain(block: ComplexLinearAttention, h: Tensor) -> Tensor:
    """a_tt = p^{-1/2} Σ_j γ_j cos(ψ_j) |z_tj|²  — factorized cubic 의 계수. (B,T,d) → (B,T)"""
    z, _, _ = block.project(h)
    w = block.gamma * torch.cos(block.psi)
    return torch.einsum("btp,p->bt", z.abs() ** 2, w) / math.sqrt(block.cfg.p)


def edge_channel_decomposition(block: ComplexLinearAttention, h: Tensor) -> Tensor:
    """a_tn,j 전체. (B,T,T,p) — 메모리 큼. 소규모 분석용."""
    _, q, k = block.project(h)
    T = q.shape[1]
    idx = torch.arange(T, device=q.device)
    delta = (idx[:, None] - idx[None, :]).clamp(min=0)
    lam_pow = torch.exp(idx.to(block.alpha.dtype)[:, None] * block.log_lam[None, :])
    W = lam_pow[delta]  # (T,T,p)
    a = (q[:, :, None, :].conj() * k[:, None, :, :] * W).real
    causal = (idx[:, None] - idx[None, :]) >= 0
    return a * causal[None, :, :, None] / math.sqrt(block.cfg.p)
