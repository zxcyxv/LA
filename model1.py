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
    phase_gate: str = "sigmoid"  # "sigmoid" | "softplus" | "hard"
    # "hard" = clamp(logit,0,1). 로짓 ≤0 에서 **정확히 0** 이므로 ω(blank)=0 이
    # 유한 파라미터로 도달 가능하다. sigmoid 는 -∞ 가 필요해 잔차가 남는다.
    ordinal_only: bool = False  # θ:=0 고정. 위치 위상을 없애고 순수 '내용 주소' 만 남긴다.
    # ω = s·gate(h) 이므로 Φ 는 '지금까지 센 토큰 수' 가 되고 주소가 길이에 불변이 된다.
    decay_init: str = "uniform"  # "uniform" = 크기 r~U(r_min,r_max) | "logtau" = log α 균등
    s_max: float = math.pi  # 채널별 순서 위상 스케일 s_j 의 초기 상한
    gate_bias_init: float = -2.0  # 초기 게이트 개방도 (≈0.12)
    driven: bool = True  # False 면 Δθ≡0 → Model 0 과 정확히 동일
    tied: bool = True  # False 면 R 개 블록을 독립 파라미터로 (깊이 용량 분리)
    use_wv: bool = False  # value 사영 W_V. span{h} 제약을 풀고 '재표현'을 가능하게 한다
    use_silu_wo: bool = False  # A·V 뒤에 SiLU + W_O. 사이에 비선형이 있어야 W_O 가
    # W_V 와 합쳐지지 않는다 (없으면 f = A h (W_V W_O) 로 d² 자유도에 2d² 파라미터)
    use_delta: bool = False  # 델타 규칙: S ← ΛS − β k̂(k̂†ΛS) + β k̂ v†.
    # 키 방향으로 선택적 치환. 대각성이 깨지므로 recurrent 경로 전용.
    polar: bool = False  # 극좌표 사영: z = (W_R h) ⊙ e^{i W_φ h}, W_R·W_φ 는 실수.
    # 진폭이 실수이므로 고정 토큰의 사영 상 {a⊙e^{iφ}: a∈R^p} 가 라그랑지안 부분공간이
    # 된다 (<q,q'> = Σ a_j a'_j 실수 → ω≡0). 초기구상의 구조를 되살리되 φ 를 입력
    # 의존으로 두어 토큰별 라그랑지안을 갖는다. 파라미터 수는 복소 W_C 와 동일(2pd).
    unitary: bool = False  # α=0, γ=1 → |U_t|=1. 감쇠 없는 무손실 중첩(초기구상 QM§2,3)
    read_norm: bool = False  # 읽기 시점에 √(누적 쓰기 에너지) 로 나눈다.
    # ‖S_t‖²_F 의 대각 성분 Σ_{n≤t}‖k_n‖²‖h_n‖² 는 정확히 cumsum 이라 스캔 선형성을
    # 깨지 않는다. 비대각(간섭)항은 무상관 입력에서 0 으로 평균된다.
    # 정렬 입력 O(t)/O(√t)=O(√t), 비상관 O(√t)/O(√t)=O(1) → 선택성 유지.
    use_bias: bool = False  # z 에 복소 바이어스 b. 커널에 상수·1차 경로가 생겨
    # 3차 동차성이 깨진다 (Yu & Erichson 2025: B 바이어스가 보편근사를 준다는 결과를
    # Mamba-3 §3.4 가 인용). q,k 가 여전히 e^{-iψ} 의 그래프라 라그랑지안은 보존된다.
    gamma_split: bool = True  # γ 를 q,k 에 √γ 씩 배분 → (q,k) 가 라그랑지안 그래프.
    # 커널은 곱 q̄k 만 보므로 함수는 불변(실측 1.7e-16). 기하학적 정합성만 얻는다.

    # --- MISALIGNMENT.md §6.2 의 개입 1, 2 (둘 다 파라미터 0개) ---
    psi_inhib: bool = False  # ψ = π/2 + π·sigmoid(ψ_raw) ∈ (π/2, 3π/2) → cos ψ < 0 항등적.
    # FHN 의 -u³/3 은 고정 음수여야 유계성을 보장한다. 자유 ψ 에서는 a_tt 부호가 내용에
    # 따라 뒤집혀(측정 [-0.207,+0.267]) 3차항이 불안정화 항이 되는 입력이 존재한다.
    # cos ψ<0 은 |sin ψ|≈1 과 양립하므로(3사분면) 주소 채널의 반대칭성을 잃지 않는다.
    # psi_init 을 무시한다.
    # --- 개입 3, 4 (깊이축 FHN) — **§9.1 에서 폐기됨.** 코드만 남긴다 ---
    # 깊이축의 총 적분 시간이 항상 1 이라 FHN 사이클(s~1/ε)이 원리적으로 안 들어간다.
    tonic: bool = False  # 개입 3: 긴장성 전류 b_I ∈ R^d. FHN 의 I / 휴지 전위.
    # f 가 정확히 3차 동차라 Df(0)=0 이고 깊이 사상의 야코비안이 원점에서 정확히 I 다
    # (완전 퇴화). 즉 선형 영역 자체가 없다. degree-0 항이 원점을 떼어내야 문턱에
    # 절대적 척도가 생긴다. init 0 → 학습 시작 시점에는 기존과 동일.
    recovery: bool = False  # 개입 4: 회복변수 v ∈ R^{B,T,d} 를 **깊이축**에 둔다.
    #     h^(r+1) = h^(r) + ( f + b_I − c⊙v^(r) ) / R
    #     v^(r+1) = v^(r) + eps ⊙ ( h^(r) − b_v⊙v^(r) ) / R
    # 우리에게 없는 것은 문턱이 아니라 **복귀**다 — u=|a_tt|/R 의 깊이 사상
    # u←u(1−u)² 이 u=2 에 불안정 고정점을 이미 갖는다(§8.6.1). 넘어간 뒤 무한으로
    # 탈출하는 것을 유한한 여행 + 복귀로 바꾸는 것이 −c⊙v 의 역할이다.
    # v 는 깊이축 상태이므로 t 축 스캔을 건드리지 않는다 → 병렬성 보존.
    t_end: float = 1.0  # 깊이축 **총 적분 시간**. 스텝 크기는 dt = t_end/R.
    # 기존 `h ← h + f/R` 을 R 번 = `dh/ds = f(h)` 를 s∈[0,1] 에서 적분하는 것이다.
    # 즉 R 은 **해상도**이고 총 시간은 언제나 1 로 암묵적으로 고정돼 있었다.
    # FHN 이 여행 한 번 완성하려면 s ~ 1/ε 이 필요하므로(ε=0.08 → 12) 총 시간 1 로는
    # 원리적으로 흥분 사이클이 불가능하다. t_end 를 분리해 R(정확도)과 지속(시간)을 가른다.
    # t_end=1.0 이면 기존과 정확히 동일. 대가: dt=t_end/R 이 커지면 u=|a_tt|·dt 가
    # §8.6.1 의 분리선 u=2 로 다가가므로 **개입 2(self_exp)가 여기서 호출된다.**
    eps_max: float = 0.5  # eps = eps_max·sigmoid(·). ε≪1 이 FHN 의 정의 조건이라 유계화한다
    eps_init: float = 0.08
    b_v_init: float = 1.0  # b_v = softplus(·) > 0. v 가 안정값을 가져야 한다
    c_init: float = 0.5  # c = softplus(·) ≥ 0. 결합이 **감산**이어야 FHN 이다
    v_init: str = "nullcline"  # "nullcline" (v=h/b_v, v 에 대해 정지) | "zero"

    # --- 갈래 C (§9): t 축 선형 막전위 + 점별 문턱 + **깊이축 리셋** ---
    # 깊이축을 '시간'으로 쓰려던 것이 실수였다. 깊이축의 총 적분 시간은 1 이므로
    # FHN 사이클(s~1/ε)이 원리적으로 안 들어간다. 깊이축은 시간이 아니라
    # §6.1 이 지적한 **되먹임 통로**다 — r 단계에서 h^(r-1) 전체가 이미 있으므로
    # 그것으로 계산한 무엇이든 r 단계 t축 스캔에게는 입력이다.
    #
    #   m_t^(r) = λ_m(1 − ρ·s_{t-1}^(r-1))·m_{t-1}^(r) + w_m⊙f_t^(r)   ← 게이트된 선형 스캔
    #   s_t^(r) = sigmoid( (m_t^(r) − θ) / τ_s )                        ← 점별 비선형
    #
    # 문턱하 누적이 **t 축**에서 일어나고(갈래 A 가 못 하는 것), 리셋(=상태 되먹임)이
    # 한 깊이 단계 늦게 적용되어 병렬 스캔이 보존된다. R 번의 (적분→검출→리셋→재적분)
    # 완화 반복이 자기정합적 스파이크열로 수렴하는지가 이 설계의 사활이다 (진단 C4).
    membrane: bool = False
    lam_m_init: float = 0.9  # λ_m = sigmoid(·) ∈ (0,1) : 막 누설
    w_m_init: float = 1.0  # 시냅스 전류 f 를 막전위로 넣는 이득
    theta_init: float = 1.0  # 발화 문턱 (채널별)
    tau_s_init: float = 0.3  # 문턱 급준도. 작을수록 전부아니면전무에 가깝다
    # --- 불응기: 진단 C3 이 실패해서 추가. 원인이 둘이었다 ---
    #  (a) 리셋이 유입 m_{t-1} 만 게이트하고 **입력 w_m⊙f_t 는 안 게이트**했다.
    #      자극이 강하면 입력 단독으로 문턱을 넘어 리셋이 무의미해진다. 생물의 불응기는
    #      '새 입력에도 반응하지 않는' 상태인데 누적분만 지우고 있었다.
    #  (b) s_{t-1} 만 봤으므로 **불응 창이 정확히 1토큰**이었다 (Δt≥2 에서 억제 0).
    # 수정: 발화 흔적 a_t = μ·a_{t-1} + s_t 를 두고 exp(−ρ·a) 로 **둘 다** 억제한다.
    # SRM(Spike Response Model)의 곱셈형 불응 커널이며, PLASTICITY.md §6.4 의
    # (a) 기존 억제 + (b) 쓰기 억제 와 **정확히 같은 수학**이다. s^(r-1) 이 상수라
    # a 도 미리 정해지므로 게이트된 선형 스캔이 유지된다 (병렬성 보존).
    mu_a_init: float = 0.7  # μ = sigmoid(·) : 발화 흔적 감쇠 → 불응 창 폭
    rho_state_init: float = 2.0  # ρ₁ = softplus(·) ≥ 0 : 유입 억제 (기존 리셋)
    rho_in_init: float = 2.0  # ρ₂ = softplus(·) ≥ 0 : **입력** 억제 (새로 추가)
    gate_write: bool = False  # 링크 3: 스파이크로 S 쓰기를 게이트. 구조 검증 뒤에 켠다
    g_floor_init: float = 0.5  # 기저 방출. g = g_floor + (1−g_floor)·s.
    # 0 이면 게이트가 완전히 닫힐 수 있어 f=0 → m=0 → 영구 침묵의 흡수 상태가 생긴다.

    self_exp: bool = False  # 자기항 a_tt 를 지수적으로 적분:
    #     h ← h·exp(a_tt/R) + (f − a_tt·h)/R          (1차까지 오일러와 동일)
    # 명시적 오일러 h ← h(1 − c‖h‖²/R) 는 로지스틱 사상이라 큰 ‖h‖ 에서 오버슈트한다.
    # 즉 연속 FHN 에서 유계성을 보장하는 항이 이산화에서는 발산을 **가속**한다.
    # Λ = e^{-α+iθ} 에 이미 지수적 처리를 쓰고 있으므로 일관성도 회복된다.
    # psi_inhib 와 **함께** 써야 한다 — a_tt>0 이면 exp(+a_tt/R) 가 증폭한다.


class DrivenComplexLinearAttention(nn.Module):
    """h ∈ R^{B,T,d} → f ∈ R^{B,T,d}. 전이 위상이 입력 의존."""

    def __init__(self, cfg: Model1Config):
        super().__init__()
        self.cfg = cfg
        d, p = cfg.d, cfg.p

        std = cfg.w_scale / math.sqrt(2.0 * d)
        if cfg.polar:
            # 진폭(실수) 과 위상을 분리. 파라미터 수는 복소 W_C 와 같다.
            self.W_R = nn.Parameter(torch.randn(p, d) * (std * math.sqrt(2.0)))
            self.W_phi = nn.Parameter(torch.randn(p, d) / math.sqrt(d))
            self.W_re = self.W_im = None
        else:
            self.W_re = nn.Parameter(torch.randn(p, d) * std)
            self.W_im = nn.Parameter(torch.randn(p, d) * std)
            self.W_R = self.W_phi = None

        if getattr(cfg, "decay_init", "uniform") == "logtau":
            # 시간척도를 로그 균등으로. 크기 r 을 균등 추출하면 반감기 ln2/(-ln r) 이
            # r→1 에서 폭발해 채널 대부분이 짧은 척도에 몰린다 (U(0.9,0.999) 는
            # 범위 6.6~693 인데 중앙값이 14). log α 를 균등하게 뿌리면 반감기가
            # 자릿수마다 고르게 배치된다.
            a_hi, a_lo = -math.log(cfg.r_min), -math.log(cfg.r_max)
            self.alpha_log = nn.Parameter(
                torch.empty(p).uniform_(math.log(a_lo), math.log(a_hi))
            )
        else:
            mag = torch.empty(p).uniform_(cfg.r_min, cfg.r_max)
            self.alpha_log = nn.Parameter(torch.log(-torch.log(mag)))
        if getattr(cfg, "ordinal_only", False):
            self.theta = nn.Parameter(torch.zeros(p), requires_grad=False)
        else:
            self.theta = nn.Parameter(torch.empty(p).uniform_(0.0, cfg.theta_max))

        if cfg.psi_inhib:
            # ψ = π/2 + π·sigmoid(raw). 초기값을 (π/2, 3π/2) 균등에 대응시킨다.
            # 파라미터는 psi_raw 지만 읽을 때는 property `psi` 가 실효값을 돌려주므로
            # blk.psi 를 보는 기존 분석 코드(analyze_excitability, self_gain 등)가 그대로 맞다.
            u = torch.empty(p).uniform_(0.02, 0.98)
            self.psi_raw = nn.Parameter(torch.log(u) - torch.log1p(-u))
        else:
            self.psi_raw = nn.Parameter(
                torch.zeros(p)
                if cfg.psi_init == "zero"
                else torch.empty(p).uniform_(-math.pi, math.pi)
            )

        # --- 구동 항 (Model 0 대비 추가되는 전부) ---
        self.W_theta = nn.Parameter(torch.randn(p, d) / math.sqrt(d))
        # read_gain: 초기값을 t=1 의 기대 read_scale 로 두어 무정규화 모델과 스케일 일치
        self.read_gain = nn.Parameter(
            torch.tensor(cfg.w_scale * math.sqrt(p * d)) if cfg.read_norm
            else torch.tensor(1.0)
        )
        self.s_raw = nn.Parameter(torch.empty(p).uniform_(0.0, cfg.s_max))
        self.gate_bias = nn.Parameter(torch.full((p,), cfg.gate_bias_init))

        # 델타 규칙의 쓰기 강도 β ∈ (0,2). 2 면 하우스홀더 반사(유니터리).
        self.W_beta = nn.Parameter(torch.randn(d) / math.sqrt(d)) if cfg.use_delta else None

        # z 바이어스. 0 초기화 → 학습 시작 시점에는 바이어스 없는 모델과 정확히 동일.
        if cfg.use_bias:
            self.b_re = nn.Parameter(torch.zeros(p))
            self.b_im = nn.Parameter(torch.zeros(p))
        else:
            self.b_re = self.b_im = None

        # value 사영. 없으면 블록 출력이 span{h_1..h_t} 안에 갇힌다 (대화내역2 §8).
        self.W_V = (
            nn.Parameter(torch.randn(d, d) / math.sqrt(d)) if cfg.use_wv else None
        )
        self.W_O = (
            nn.Parameter(torch.randn(d, d) / math.sqrt(d)) if cfg.use_silu_wo else None
        )

    # ------------------------------------------------------------------ #
    @property
    def psi(self) -> Tensor:
        """(p,) 실효 ψ. psi_inhib 면 (π/2, 3π/2) 로 묶여 cos ψ < 0 이 항등적이다."""
        if self.cfg.psi_inhib:
            return math.pi / 2 + math.pi * torch.sigmoid(self.psi_raw)
        return self.psi_raw

    @property
    def alpha(self) -> Tensor:
        if self.cfg.unitary:
            return torch.zeros_like(self.alpha_log)
        return torch.exp(self.alpha_log)

    @property
    def gamma(self) -> Tensor:
        if self.cfg.unitary or not self.cfg.use_gamma:
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
        if self.cfg.phase_gate == "hard":
            return logit.clamp(0.0, 1.0)
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

    def read_scale(self, h: Tensor, k: Tensor) -> Tensor:
        """√(eps + Σ_{n≤t} ‖k_n‖²‖h_n‖²) — (B,T,1). 상태에 써넣은 누적 에너지."""
        e = (k.abs() ** 2).sum(-1) * (h ** 2).sum(-1)  # (B,T)
        return torch.sqrt(1e-8 + torch.cumsum(e, dim=1)).unsqueeze(-1)

    def project(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.cfg.polar:
            amp = torch.einsum("btd,pd->btp", h, self.W_R)          # 실수 진폭
            phi = torch.einsum("btd,pd->btp", h, self.W_phi)        # 입력 의존 위상
            z = amp.to(torch.complex64 if h.dtype == torch.float32 else torch.complex128)
            z = z * torch.polar(torch.ones_like(phi), phi)
        else:
            W = torch.complex(self.W_re, self.W_im)
            z = torch.einsum("btd,pd->btp", h.to(W.dtype), W)
        if self.b_re is not None:
            z = z + torch.complex(self.b_re, self.b_im)
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
    def forward(self, h: Tensor, mode: str = "parallel", return_A: bool = False,
                gate: Tensor | None = None):
        """gate: (B,T,d) 쓰기 게이트. 갈래 C 의 스파이크가 여기로 들어온다.

        `S_t = Λ S_{t-1} + k_t (g_t ⊙ h_t)^T` — 게이트가 **value(post) 쪽**에 걸린다.
        `s` 가 d 차원이므로 차원이 맞고, 그 결과가 `Δw ∝ pre × post_출력` 이다.
        PLASTICITY.md §6 이 "빠진 것은 post 항" 이라 지목한 자리이며, 어긋남 `#1`
        (post 가 출력이 아니라 입력이었다)이 여기서 고쳐진다.
        게이트는 이전 깊이 단계에서 계산되므로 `A` 와 스캔에는 영향이 없다 → 병렬성 보존.
        """
        if self.cfg.use_delta:
            mode = "recurrent"
        if mode == "parallel":
            A = self.attention_matrix(h)
            v = h if self.W_V is None else h @ self.W_V
            if gate is not None:
                v = gate * v
            f = torch.einsum("btn,bnd->btd", A, v)
            if self.W_O is not None:
                f = torch.nn.functional.silu(f) @ self.W_O
            return (f, A) if return_A else f
        if mode == "recurrent":
            if return_A:
                raise ValueError("recurrent 경로는 A 를 만들지 않는다.")
            return self._forward_recurrent(h, gate=gate)
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

        A = A * causal.to(A.dtype) / math.sqrt(p)
        if self.cfg.read_norm:
            A = A * (self.read_gain / self.read_scale(h, k))
        return A

    def _forward_recurrent(self, h: Tensor, return_norm: bool = False,
                           gate: Tensor | None = None) -> Tensor:
        _, q, k = self.project(h)
        omega = self.phase_increment(h)  # (B,T,p)
        B, T, p = q.shape

        v = h if self.W_V is None else h @ self.W_V
        if gate is not None:
            v = gate * v
        hc = v.to(q.dtype)
        S = q.new_zeros(B, p, v.shape[-1])
        alpha = self.alpha[None, :, None]
        beta_all = (
            2.0 * torch.sigmoid(h @ self.W_beta) if self.W_beta is not None else None
        )
        out, norms = [], []
        for t in range(T):
            U = torch.polar(torch.exp(-alpha).expand(B, p, 1), omega[:, t].unsqueeze(-1))
            S = U * S
            if self.cfg.use_delta:
                kh = k[:, t] / (k[:, t].norm(dim=-1, keepdim=True) + 1e-6)   # (B,p)
                u = (kh.conj().unsqueeze(-1) * S).sum(1)                      # (B,d) 저장값
                bt = beta_all[:, t].unsqueeze(-1).unsqueeze(-1).to(S.dtype)
                S = S + bt * kh.unsqueeze(-1) * (hc[:, t] - u).unsqueeze(1)   # 오차만 기록
            else:
                S = S + k[:, t].unsqueeze(-1) * hc[:, t].unsqueeze(1)
            out.append((q[:, t].conj().unsqueeze(-1) * S).sum(1).real)
            if return_norm:
                norms.append(S.norm(dim=(1, 2)).mean().item())
        f = torch.stack(out, dim=1) / math.sqrt(p)
        if self.cfg.read_norm:
            f = f * (self.read_gain / self.read_scale(h, k))
        if self.W_O is not None:
            f = torch.nn.functional.silu(f) @ self.W_O
        return (f, norms) if return_norm else f


class Model1(nn.Module):
    """embedding → 같은 구동 블록 R 번 Euler 적분 → tied head."""

    def __init__(self, cfg: Model1Config):
        super().__init__()
        self.cfg = cfg
        if cfg.self_exp and (cfg.use_wv or cfg.use_silu_wo or cfg.use_delta):
            # self_exp 는 f 의 자기항이 정확히 a_tt·h_t 라는 분해에 의존한다.
            # use_wv 면 자기항이 a_tt·(h_t W_V) 이고, use_silu_wo/use_delta 면 A 가 없다.
            raise ValueError(
                "self_exp 는 value=h 인 스칼라 자기항 a_tt·h_t 에만 정의된다. "
                "use_wv 면 자기항이 a_tt·(h_t W_V) 인 행렬 동역학이므로 함께 쓸 수 없다. "
                "W_V 실험은 self_exp 를 빼고 psi_inhib + use_wv 로 실행하라. "
                "use_silu_wo / use_delta 역시 현재 분해와 호환되지 않는다."
            )
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

        # --- 깊이축 FHN 파라미터. 블록이 아니라 모델에 둔다 (막의 성질이고 tied 여부와 무관) ---
        inv_sp = lambda y: math.log(math.expm1(y))  # softplus^{-1}
        self.b_I = nn.Parameter(torch.zeros(cfg.d)) if cfg.tonic else None
        if cfg.recovery:
            self.c_raw = nn.Parameter(torch.full((cfg.d,), inv_sp(cfg.c_init)))
            self.b_v_raw = nn.Parameter(torch.full((cfg.d,), inv_sp(cfg.b_v_init)))
            q = cfg.eps_init / cfg.eps_max
            self.eps_raw = nn.Parameter(torch.full((cfg.d,), math.log(q / (1 - q))))
        else:
            self.c_raw = self.b_v_raw = self.eps_raw = None

        if cfg.membrane:
            logit = lambda y: math.log(y / (1 - y))
            self.lam_m_raw = nn.Parameter(torch.full((cfg.d,), logit(cfg.lam_m_init)))
            self.mu_a_raw = nn.Parameter(torch.full((cfg.d,), logit(cfg.mu_a_init)))
            self.rho_state_raw = nn.Parameter(torch.full((cfg.d,), inv_sp(cfg.rho_state_init)))
            self.rho_in_raw = nn.Parameter(torch.full((cfg.d,), inv_sp(cfg.rho_in_init)))
            self.w_m = nn.Parameter(torch.full((cfg.d,), cfg.w_m_init))
            self.theta = nn.Parameter(torch.full((cfg.d,), cfg.theta_init))
            self.tau_s_raw = nn.Parameter(torch.full((cfg.d,), inv_sp(cfg.tau_s_init)))
            self.g_floor_raw = nn.Parameter(torch.full((cfg.d,), logit(cfg.g_floor_init)))
        else:
            self.g_floor_raw = None
            self.lam_m_raw = self.mu_a_raw = self.w_m = None
            self.rho_state_raw = self.rho_in_raw = None
            self.theta = self.tau_s_raw = None

    @property
    def lam_m(self) -> Tensor:  # (0,1) : 막 누설
        return torch.sigmoid(self.lam_m_raw)

    @property
    def mu_a(self) -> Tensor:  # (0,1) : 발화 흔적 감쇠 = 불응 창 폭
        return torch.sigmoid(self.mu_a_raw)

    @property
    def rho_state(self) -> Tensor:  # ≥ 0 : 유입 억제
        return torch.nn.functional.softplus(self.rho_state_raw)

    @property
    def rho_in(self) -> Tensor:  # ≥ 0 : 입력 억제
        return torch.nn.functional.softplus(self.rho_in_raw)

    @property
    def tau_s(self) -> Tensor:  # > 0 : 문턱 급준도
        return torch.nn.functional.softplus(self.tau_s_raw)

    def membrane_scan(self, f: Tensor, s_prev: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """발화 흔적을 가진 막전위 스캔.

            a_t = μ⊙a_{t-1} + s_t^(r-1)                          발화 흔적 (SRM)
            m_t = λ_m·e^{−ρ₁ a_{t-1}}·m_{t-1}
                       + e^{−ρ₂ a_{t-1}}·w_m⊙f_t                 유입·입력 **둘 다** 억제
            s_t = sigmoid( (m_t − θ) / τ_s )

        `s_prev` 는 **이전 깊이 단계**의 값이라 이 스캔에게는 상수다. 따라서 `a` 도
        미리 정해지고, 계수가 전부 사전 결정된 **게이트된 선형 스캔**이 되어 결합법칙이
        성립한다 (Mamba selectivity 와 같은 형태). T 가 작아 여기서는 루프로 쓴다.

        `a_{t-1}` 을 쓰는 이유: 자기 발화가 자신을 억제하면 순환이 된다.
        `exp(−ρa)` 를 쓰는 이유: `a` 가 1 을 넘을 수 있어 `1−ρa` 는 음수가 될 수 있다.
        """
        T = f.shape[1]
        # 발화 흔적 (s_prev 가 상수이므로 이것도 상수)
        a_t = torch.zeros_like(s_prev[:, 0])
        traces = []
        for t in range(T):
            a_t = self.mu_a * a_t + s_prev[:, t]
            traces.append(a_t)
        a = torch.stack(traces, dim=1)  # (B,T,d)
        a_sh = torch.cat([torch.zeros_like(a[:, :1]), a[:, :-1]], dim=1)  # a_{t-1}

        g_state = self.lam_m * torch.exp(-self.rho_state * a_sh)
        g_in = torch.exp(-self.rho_in * a_sh)
        b = g_in * (self.w_m * f)

        m_t = torch.zeros_like(f[:, 0])
        out = []
        for t in range(T):
            m_t = g_state[:, t] * m_t + b[:, t]
            out.append(m_t)
        m = torch.stack(out, dim=1)
        return m, torch.sigmoid((m - self.theta) / self.tau_s), a

    # ---- 이론이 요구하는 부호/유계성을 파라미터화로 강제한다 (개입 1 의 교훈) ----
    @property
    def c(self) -> Tensor:  # ≥ 0 : 감산 결합
        return torch.nn.functional.softplus(self.c_raw)

    @property
    def b_v(self) -> Tensor:  # > 0 : v 가 안정값을 갖는다
        return torch.nn.functional.softplus(self.b_v_raw)

    @property
    def eps(self) -> Tensor:  # (0, eps_max) : ε ≪ 1
        return self.cfg.eps_max * torch.sigmoid(self.eps_raw)

    def forward(self, tokens: Tensor, mode: str = "parallel", return_trace: bool = False,
                h0: Tensor | None = None, pulse: dict | None = None):
        """pulse: {r: 그 깊이 스텝에서 h 에 더할 (B,T,d) 텐서} — 구조 진단용 자극 주입."""
        h = self.embed(tokens) if h0 is None else h0
        v = None
        if self.cfg.recovery:
            v = h / self.b_v if self.cfg.v_init == "nullcline" else torch.zeros_like(h)
        # 갈래 C: 스파이크는 **이전** 깊이 단계 값을 쓴다. 첫 단계는 정보가 없다.
        # 불응 흔적에는 s_prev=0 (직전 발화 없음)이 맞지만, 쓰기 게이트에는 0 이면
        # 아무것도 안 써서 f=0 → m=0 → 영구 침묵이 된다 (죽은 시작). 두 용도의
        # 초기값이 다르므로 첫 단계는 None 으로 표시하고 게이트만 1 로 둔다.
        s_prev = None
        trace = []
        need_A = return_trace or self.cfg.self_exp
        for _r in range(self.cfg.R):
            if pulse is not None and _r in pulse:
                h = h + pulse[_r]
            blk = self.block if self.blocks is None else self.blocks[_r]
            # 쓰기 게이트 g = g_floor + (1−g_floor)·s.  g_floor 는 문턱 아래에서도
            # 남는 기저 방출(spontaneous release)에 해당하며, 학습이 게이트를 완전히
            # 닫아 기울기를 죽이는 것을 막는다. gate_write=False 면 g=None 으로
            # 기존과 정확히 동일하다.
            g = None
            if self.cfg.gate_write:
                if s_prev is None:
                    g = torch.ones_like(h)
                else:
                    gf = torch.sigmoid(self.g_floor_raw)
                    g = gf + (1.0 - gf) * s_prev
            A = None
            if need_A:
                f, A = blk(h, mode="parallel", return_A=True, gate=g)
            else:
                f = blk(h, mode=mode, gate=g)
            if return_trace:
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
            # 갈래 C: f 는 시냅스 전류다. 막전위는 그것을 t 축으로 누적한다.
            if self.cfg.membrane:
                sp = torch.zeros_like(h) if s_prev is None else s_prev
                m, s_now, a_tr = self.membrane_scan(f, sp)
                if return_trace:
                    trace[-1]["m"] = m.detach()
                    trace[-1]["s"] = s_now.detach()
                    trace[-1]["a_ref"] = a_tr.detach()
                s_prev = s_now  # 다음 깊이 단계가 이것으로 리셋한다

            # 깊이축 구동항: f + b_I − c⊙v
            drive = f
            if self.b_I is not None:
                drive = drive + self.b_I
            if v is not None:
                drive = drive - self.c * v
            dt = self.cfg.t_end / self.cfg.R
            if self.cfg.self_exp:
                # f = Σ_n a_tn (g_n ⊙ h_n) 에서 자기항만 분리해 지수적으로 적분한다.
                # a_tt<0 이면 스텝 크기와 무관하게 수축이므로 오버슈트가 없다.
                a_tt = torch.diagonal(A, dim1=-2, dim2=-1).unsqueeze(-1)  # (B,T,1)
                if g is not None:
                    # **자기 계수는 A_tt 가 아니라 A_tt·g_t 다.** 게이트가 value 에
                    # 걸리므로 n=t 항도 g_t 배가 된다. 이걸 빼먹으면 drive 에서
                    # A_tt·h 를 과잉 감산하고(= −A_tt(1−g)h 증폭항이 생긴다) 동시에
                    # 게이트 안 걸린 full 감쇠 exp(A_tt·dt) 를 적용해 분해가 어긋난다.
                    a_tt = a_tt * g  # (B,T,d)
                h_new = h * torch.exp(a_tt * dt) + (drive - a_tt * h) * dt
            else:
                h_new = h + drive * dt
            if v is not None:  # 동시 갱신 (2D 계의 명시적 오일러)
                v = v + self.eps * (h - self.b_v * v) * dt
                if return_trace:
                    trace[-1]["v"] = v.detach()
            h = h_new
        if return_trace:
            trace.append({"h": h.detach(), "E_D": dirichlet_energy(h.detach())})
        logits = self.logit_scale * (h @ self.embed.weight.t())
        return (logits, trace) if return_trace else logits

    @torch.no_grad()
    def calibrate_threshold(self, tokens: Tensor, q: float = 0.995, sharp: float = 0.05):
        """θ·τ_s 를 무자극 `m` 분포로 보정한다 (데이터 의존 초기화).

        `theta_init` 고정값은 `m` 의 절대 스케일을 모르는 상태의 추측이다. 빗나가면
        항상 발화 또는 영구 침묵이 되어 학습이 시작조차 못 한다. S4/Mamba 가 `dt` 를
        세심히 초기화하는 것과 같은 이유이고, `CAPACITY.md §4.2` 는 그런 축이 학습으로
        **교정되지 않는다**는 것을 `τ` 에서 실측했다 — 초기화로 맞춰야 한다.

        q: 기저 발화율 목표의 여집합. 0.995 면 자극 없이 0.5% 만 문턱 위.
        """
        if not self.cfg.membrane:
            return None
        tr = self(tokens, return_trace=True)[1]
        mm = torch.cat([x["m"] for x in tr if "m" in x], dim=0)
        flat = mm.reshape(-1, mm.shape[-1]).float()
        self.theta.copy_(torch.quantile(flat, q, dim=0).to(self.theta.dtype))
        self.tau_s_raw.fill_(math.log(math.expm1(max(sharp * mm.std().item(), 1e-4))))
        return {"m_std": mm.std().item(), "theta_mean": self.theta.mean().item()}

    def num_parameters(self) -> dict[str, int]:
        mods = [self.block] if self.blocks is None else list(self.blocks)
        blk = sum(x.numel() for m in mods for x in m.parameters())
        emb = self.embed.weight.numel()
        fhn = sum(
            x.numel() for x in (self.b_I, self.c_raw, self.b_v_raw, self.eps_raw)
            if x is not None
        )
        mem = sum(
            x.numel() for x in (self.lam_m_raw, self.mu_a_raw, self.rho_state_raw,
                                self.rho_in_raw, self.w_m, self.theta, self.tau_s_raw,
                                self.g_floor_raw)
            if x is not None
        )
        out = {"block": blk, "embedding": emb, "total": blk + emb + fhn + mem + 1}
        if fhn:
            out["fhn"] = fhn
        if mem:
            out["membrane"] = mem
        return out
