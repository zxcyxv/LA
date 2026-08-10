"""생물학적 층위 정합성 진단 — MISALIGNMENT.md 의 [A]~[E] 를 재현한다.

이 스크립트는 학습을 하지 않는다. 전부 **초기화 시점의 구조적 성질**만 잰다.
그것으로 충분한 주장들만 여기 모았다 (model0.py §1.3 의 w_scale 표도 같은 층위다).

    [A] f(0) = 0 인가 — 고유 스케일(=안정 정지점, 휴지 전위)의 존재 여부
    [B] a_tt 의 부호가 내용에 따라 뒤집히는가 — FHN ① 포화 보장 여부
    [C] 3차 포화항이 유계성을 보장하는가 — 명시적 오일러 vs 자기항 지수 적분
    [D] 흔적 커널 모양 — 단일 지수x회전 vs 이중 지수(캐스케이드)
    [E] 회복변수 S 가 깊이축에서 유지되는가

사용:
    python diag_misalign.py            # 전부
    python diag_misalign.py --only C   # 한 절만
"""

from __future__ import annotations

import argparse
import math
import sys

import torch

from model0 import (
    ComplexLinearAttention,
    Model0,
    Model0Config,
    dirichlet_energy,
)
from model1 import DrivenComplexLinearAttention, Model1Config

D_MODEL, P, T, B = 128, 64, 36, 4
INHIB_LO, INHIB_HI = math.pi / 2 + 0.05, 3 * math.pi / 2 - 0.05  # cos psi < 0 구간


def _banner(s: str) -> None:
    print("=" * 74)
    print(s)
    print("=" * 74)


# --------------------------------------------------------------------------- #
# [A] 고유 스케일 — 문턱의 필요조건
# --------------------------------------------------------------------------- #
def check_A() -> None:
    _banner("[A] f(0)=0 인가 — 안정 정지점(휴지 전위)이 원점 밖에 있는가")
    print("  value 경로가 h (또는 h W_V) 이므로 A 에 무엇을 넣어도 f = A·0 = 0 이다.")
    print("  즉 h=0 이 항상 평형이고, 그 평형은 max Re lambda(M) > 0 으로 불안정하다.")
    print("  → FHN 의 '안정 정지점 + 불안정 중간가지' 구조가 성립할 자리가 없다.\n")

    flags = [
        {},
        {"use_bias": True},
        {"use_wv": True},
        {"use_silu_wo": True},
        {"polar": True},
        {"read_norm": True},
        {"unitary": True},
        {"use_bias": True, "use_wv": True, "use_silu_wo": True},
    ]
    h0 = torch.zeros(2, T, D_MODEL)
    for fl in flags:
        torch.manual_seed(0)
        cfg = Model1Config(d=D_MODEL, p=P, vocab_size=18, **fl)
        blk = DrivenComplexLinearAttention(cfg)
        with torch.no_grad():
            # use_bias 는 0 초기화라 학습된 상황을 흉내내어 강제로 채운다.
            # 바이어스가 있어도 f(0)=0 이 유지되는 것이 요점이다.
            if blk.b_re is not None:
                blk.b_re.fill_(0.7)
                blk.b_im.fill_(-0.4)
            f = blk(h0, mode="parallel")
        print(f"  {str(fl)[:50]:<52} max|f(0)| = {f.abs().max().item():.3e}")


# --------------------------------------------------------------------------- #
# [B] FHN ① 포화항의 부호
# --------------------------------------------------------------------------- #
def _att(blk: ComplexLinearAttention, h: torch.Tensor) -> torch.Tensor:
    z, _, _ = blk.project(h)
    w = blk.gamma * torch.cos(blk.psi)
    return torch.einsum("btp,p->bt", z.abs() ** 2, w) / math.sqrt(P)


def check_B(n_seed: int = 6) -> None:
    _banner("[B] a_tt 의 부호 — FHN 의 -u^3/3 은 고정 음수여야 한다")
    print("  a_tt = p^-1/2 sum_j gamma_j cos(psi_j) |z_tj|^2.")
    print("  cos psi_j < 0 을 전 채널에 강제하면 입력과 무관하게 a_tt < 0 이 보장된다.\n")
    for mode in ("free", "inhib"):
        lo, hi = [], []
        for s in range(n_seed):
            torch.manual_seed(s)
            blk = ComplexLinearAttention(Model0Config(d=D_MODEL, p=P, vocab_size=18))
            with torch.no_grad():
                if mode == "inhib":
                    blk.psi.uniform_(INHIB_LO, INHIB_HI)
                a = _att(blk, torch.randn(B, T, D_MODEL))
            lo.append(a.min().item())
            hi.append(a.max().item())
        verdict = "부호 뒤집힘 — 포화 보장 없음" if max(hi) > 0 else "항상 음수 — 포화 보장"
        print(f"  psi={mode:<6} a_tt 범위 [{min(lo):+.3f}, {max(hi):+.3f}]   → {verdict}")


# --------------------------------------------------------------------------- #
# [C] 유계성 — 명시적 오일러 vs 자기항 지수 적분
# --------------------------------------------------------------------------- #
def _run_depth(w_scale: float, R: int, psi_mode: str, integ: str, seed: int = 0):
    """순전파 1회의 디리클레 에너지 증폭률과 max|a_tt| 를 돌려준다."""
    torch.manual_seed(seed)
    cfg = Model0Config(d=D_MODEL, p=P, R=R, vocab_size=18, w_scale=w_scale)
    blk = ComplexLinearAttention(cfg)
    with torch.no_grad():
        if psi_mode == "inhib":
            blk.psi.uniform_(INHIB_LO, INHIB_HI)
        emb = torch.randn(18, D_MODEL)
        h = emb[torch.randint(0, 18, (B, T))]
        e0 = dirichlet_energy(h).mean().item()
        a_max = 0.0
        for _ in range(R):
            f, A = blk(h, mode="parallel", return_A=True)
            dg = torch.diagonal(A, dim1=-2, dim2=-1)  # a_tt, (B,T)
            a_max = max(a_max, dg.abs().max().item())
            if integ == "euler":
                h = h + f / R
            else:
                # 자기항만 지수적으로. 1차까지 오일러와 동일하고, a_tt<0 이면
                # 스텝 크기와 무관하게 수축이다 (오버슈트가 없다).
                h = h * torch.exp(dg.unsqueeze(-1) / R) + (f - dg.unsqueeze(-1) * h) / R
            if not torch.isfinite(h).all():
                return float("inf"), a_max
        return dirichlet_energy(h).mean().item() / e0, a_max


def check_C() -> None:
    _banner("[C] 3차 포화항이 유계성을 보장하는가")
    print("  h <- h(1 - c|h|^2/R) 는 로지스틱 사상이다. 명시적 오일러에서 3차 감쇠는")
    print("  큰 |h| 에서 오버슈트하여 발산을 **가속**한다. 부호만 고치면 더 나빠진다.")
    print("  자기항을 지수적으로 적분하면(exp 열) 사라진다.\n")
    fmt = lambda v: "발산" if v == float("inf") else f"x{v:,.2f}"
    hdr = (f"  {'w_scale':>7} {'R':>2} | {'오일러/자유':>14} {'오일러/cos<0':>14} "
           f"{'지수/cos<0':>14} | {'max|a_tt|':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for R in (1, 4):
        for ws in (0.5, 1.0, 1.5, 2.0):
            r1, am = _run_depth(ws, R, "free", "euler")
            r2, _ = _run_depth(ws, R, "inhib", "euler")
            r3, _ = _run_depth(ws, R, "inhib", "exp")
            print(f"  {ws:>7} {R:>2} | {fmt(r1):>14} {fmt(r2):>14} {fmt(r3):>14} | {am:>9.2f}")
    print("\n  주의: 초기화 시점, 시드 1개, 미학습. 강성(stiffness) 주장으로는 충분하나")
    print("        '그래도 학습이 되는가' 는 이 표로 답할 수 없다.")


# --------------------------------------------------------------------------- #
# [D] 흔적 커널 모양
# --------------------------------------------------------------------------- #
def check_D() -> None:
    _banner("[D] 흔적 커널 — 단일 지수x회전 vs 이중 지수(캐스케이드)")
    print("  생물학의 가소성 창은 상승상 + 감쇠상(2차 동역학)이라 피크가 lag>0 에 있다.")
    print("  현재 커널은 1차 + 회전이므로 lag 0 에서 최대이고 진동으로 창을 흉내낸다.\n")
    lag = torch.arange(0, 40).double()

    # EXTRAPOLATION.md 실측 중앙값: theta 2.93 rad/스텝, 반감기 중앙 11.5 토큰
    theta, alpha = 2.93, 1.0 / 11.5
    cur = torch.exp(-alpha * lag) * torch.cos(theta * lag)

    # 캐스케이드: 빠른 극점 -> 느린 극점. 닫힌 형태가 (lam^(D+1)-mu^(D+1))/(lam-mu).
    lam, mu = math.exp(-1 / 8), math.exp(-1 / 60)
    cas = (lam ** (lag + 1) - mu ** (lag + 1)) / (lam - mu)

    for name, w in (("현재 e^-aD·cos(thD)", cur), ("캐스케이드 alpha-func", cas)):
        flips = int(((w[:-1] * w[1:]) < 0).sum())
        peak = int(w.abs().argmax())
        ratio = abs(w[0].item()) / w.abs().max().item()
        print(f"  {name:<22} 피크 lag={peak:>2}  부호반전={flips:>2}회  lag0/피크={ratio:.3f}")
    print(f"\n  현재      앞 8개: {[round(v, 2) for v in cur[:8].tolist()]}")
    print(f"  캐스케이드 앞 8개: {[round(v, 2) for v in cas[:8].tolist()]}   <- 상승상 존재")
    print("\n  부호 반전 0회 = 실극점 = 원리적으로 앨리어싱 없음.")
    print("  EXTRAPOLATION.md §6 이 지목한 절벽의 원인(전 채널 Nyquist 위)과 직결된다.")


# --------------------------------------------------------------------------- #
# [E] 회복변수의 깊이축 지속성
# --------------------------------------------------------------------------- #
def check_E() -> None:
    _banner("[E] 회복변수 S 가 깊이축에서 유지되는가")
    print("  FHN 은 u 와 v 가 같은 축에서 결합되어야 성립한다.")
    print("  S 가 매 깊이 스텝마다 0 으로 재생성되면 S^(r) 는 h^(r) 만의 함수이고,")
    print("  (u,v) 위상평면 궤적이 곡선 위에 갇힌다 — 면적이 없으니 고리도 없다.\n")

    count = {"n": 0}
    orig = ComplexLinearAttention._forward_recurrent

    def patched(self, h):
        count["n"] += 1
        return orig(self, h)

    ComplexLinearAttention._forward_recurrent = patched
    try:
        torch.manual_seed(0)
        R = 4
        m = Model0(Model0Config(d=64, p=32, R=R, vocab_size=18))
        with torch.no_grad():
            m(torch.randint(0, 18, (2, 16)), mode="recurrent")
    finally:
        ComplexLinearAttention._forward_recurrent = orig

    print(f"  R={R} 순전파 1회 → S 를 0 에서 새로 만든 횟수: {count['n']}  (= R)")
    print("  → 깊이축 기억 없음. ③ 느린 회복변수는 t 축에만, ① 막전위 적분은 r 축에만 있다.")
    print("     ④ 시간척도 분리 eps = (1/alpha) / (1/R) 은 분모와 분자가 다른 축이라")
    print("     비 자체가 정의되지 않는다.")


# --------------------------------------------------------------------------- #
CHECKS = {"A": check_A, "B": check_B, "C": check_C, "D": check_D, "E": check_E}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(CHECKS), help="한 절만 실행")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    keys = [args.only] if args.only else sorted(CHECKS)
    for i, k in enumerate(keys):
        if i:
            print()
        CHECKS[k]()


if __name__ == "__main__":
    sys.exit(main())
