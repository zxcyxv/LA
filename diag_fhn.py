"""깊이축 FHN 구조 진단 — 개입 3(긴장성 전류) + 4(회복변수)가 구조를 만드는가.

**학습을 하지 않는다.** 손으로 FHN 영역에 놓은 파라미터로 순전파만 돌린다.
여기서 구조가 안 나타나면 학습으로도 안 나온다 (MISALIGNMENT.md §8.4 의 교훈:
동역학 주장을 태스크 정확도로 재면 부호가 뒤집힌다).

**핵심 축이 둘이다.** `h ← h + f·dt` 를 R 번, `dt = t_end/R`.
  · `R`      = 적분 **해상도**
  · `t_end`  = 적분 **총 시간**  ← 기존 코드는 이것이 1 로 암묵 고정돼 있었다
FHN 은 여행 한 번에 `s ~ 1/ε` 가 필요하므로 `t_end` 가 진짜 변수다.

    [T0] 통제 환원 — 플래그 off / t_end=1 에서 기존과 비트 단위 동일한가
    [T1] 원점 밖 안정 정지점 — ‖h‖ 가 0 도 ∞ 도 아닌 값으로 수렴하는가
    [T2] 유계 여행 + 복귀 — ‖h‖ 가 **비단조**인가 (내부 극대 후 하강). 단조면 여행이 없다
    [T3] 문턱 — 증폭률 (응답/자극) 이 자극 진폭에 대해 급변하는가

사용:
    python diag_fhn.py
    python diag_fhn.py --R 8 16 32 --t-end 1 4 12 32
"""

from __future__ import annotations

import argparse
import math

import torch

from model1 import Model1, Model1Config

BASE = dict(d=128, p=64, vocab_size=18, psi_inhib=True, self_exp=True)  # 개입 1+2 를 표준으로
B_I = 0.3  # 구조 시험용 긴장성 전류 (학습 초기값은 0 이라 손으로 켠다)


def build(R, fhn: bool, t_end=1.0, seed=0, **over):
    torch.manual_seed(seed)
    kw = dict(BASE, R=R, t_end=t_end, **over)
    if fhn:
        kw.update(tonic=True, recovery=True)
    m = Model1(Model1Config(**kw)).eval()
    if fhn:
        with torch.no_grad():
            m.b_I.fill_(B_I)
    return m


def trace_of(m, tok, pulse=None):
    with torch.no_grad():
        tr = m(tok, return_trace=True, pulse=pulse)[1]
    hs = [s["h"] for s in tr]
    vs = [s["v"] for s in tr if "v" in s]
    us = [torch.diagonal(s["A"], dim1=-2, dim2=-1).abs().max().item()
          * (m.cfg.t_end / m.cfg.R) for s in tr if "A" in s]
    return hs, vs, us


# --------------------------------------------------------------------------- #
def t0(_):
    print("=" * 78)
    print("[T0] 통제 환원")
    print("=" * 78)
    tok = torch.randint(0, 18, (3, 24))
    for R in (4, 8):
        a, b = build(R, False, seed=1), build(R, False, seed=1)
        c = build(R, True, seed=1, c_init=1e-9)
        base = build(R, False, seed=1)
        with torch.no_grad():
            d0 = (a(tok) - b(tok)).abs().max().item()
            d1 = (c(tok) - base(tok)).abs().max().item()
        print(f"  R={R}  플래그 off {d0:.3e}   |  recovery on + c≈1e-9 (b_I=0.3 포함) {d1:.3e}")
    print("  두 번째 열은 0 이 아니어야 정상이다 — b_I 를 0.3 으로 켰으므로.")
    print("  정확한 환원은 첫 열(플래그 off)이 담당한다.\n")


def t1(Rs, Ts):
    print("=" * 78)
    print("[T1] 원점 밖 안정 정지점 — ‖h‖ 최종/초기, 그리고 분리선 u=2 까지의 여유")
    print("=" * 78)
    tok = torch.randint(0, 18, (4, 24))
    print(f"  {'R':>4} {'t_end':>6} {'구성':>11} {'‖h‖최종/초기':>13} {'max u':>8}  판정")
    for te in Ts:
        for R in Rs:
            for name, fhn in (("기준(1+2)", False), ("+3,4", True)):
                m = build(R, fhn, t_end=te)
                hs, _, us = trace_of(m, tok)
                n0 = hs[0].norm(dim=-1).mean().item()
                nf = hs[-1].norm(dim=-1).mean().item()
                r = nf / n0
                mu = max(us) if us else float("nan")
                verdict = ("발산" if not math.isfinite(r) or r > 50 else
                           "0 으로 붕괴" if r < 0.02 else "유한값 유지")
                warn = "  ⚠u>2" if mu > 2 else ""
                print(f"  {R:>4} {te:>6.1f} {name:>11} {r:>13.4f} {mu:>8.3f}  {verdict}{warn}")
        print()


def t2(Rs, Ts):
    print("=" * 78)
    print("[T2] 유계 여행 + 복귀 — ‖h‖ 가 비단조인가 (내부 극대 후 하강)")
    print("=" * 78)
    print("  단조면 여행이 없다. 면적 지표는 단조 궤적에서 무의미하므로 먼저 단조성을 본다.\n")
    tok = torch.randint(0, 18, (4, 24))
    print(f"  {'R':>4} {'t_end':>6} {'피크 위치':>10} {'피크/시작':>10} {'끝/피크':>9}  판정")
    for te in Ts:
        for R in Rs:
            m = build(R, True, t_end=te)
            hs, _, _ = trace_of(m, tok)
            n = [x.norm(dim=-1).mean().item() for x in hs]
            pk = max(range(len(n)), key=lambda i: n[i])
            interior = 0 < pk < len(n) - 1
            print(f"  {R:>4} {te:>6.1f} {f'{pk}/{len(n)-1}':>10} "
                  f"{n[pk] / n[0]:>10.4f} {n[-1] / n[pk]:>9.4f}  "
                  f"{'**내부 극대 = 여행 있음**' if interior else '단조 — 여행 없음'}")
        print()


def t3(Rs, Ts):
    print("=" * 78)
    print("[T3] 문턱 — 증폭률(응답/자극)이 자극 진폭에 대해 급변하는가")
    print("=" * 78)
    print("  응답 = max_{r>=1} ‖h_자극 − h_무자극‖ (자극 준 토큰에서). 자극 = ‖pulse‖.")
    print("  귀무가설: 증폭률이 A 에 대해 매끄럽게 단조. 문턱: 특정 A 에서 급증 후 포화.\n")
    tok = torch.randint(0, 18, (2, 24))
    amps = torch.logspace(-1.0, 1.5, 11)
    t0i = 12
    for te in Ts:
        for R in Rs:
            for name, fhn in (("기준(1+2)", False), ("+3,4", True)):
                m = build(R, fhn, t_end=te)
                base_hs, _, _ = trace_of(m, tok)
                direction = torch.randn(tok.shape[0], 1, m.cfg.d)
                direction = direction / direction.norm(dim=-1, keepdim=True)
                gains = []
                for A in amps:
                    pulse = torch.zeros_like(base_hs[0])
                    pulse[:, t0i : t0i + 1, :] = A * direction * math.sqrt(m.cfg.d)
                    hs, _, _ = trace_of(m, tok, pulse={0: pulse})
                    dev = max((hs[r][:, t0i, :] - base_hs[r][:, t0i, :]).norm(dim=-1).mean().item()
                              for r in range(1, len(hs)))
                    gains.append(dev / pulse.norm(dim=-1).max().item())
                g = torch.tensor(gains)
                ratio = (g.max() / g.min()).item() if g.min() > 0 else float("inf")
                jump = (g[1:] / g[:-1].clamp_min(1e-12)).max().item()
                print(f"  R={R:>3} t_end={te:>5.1f} {name:>11}  증폭률 "
                      f"{g.min():.3f}~{g.max():.3f}  최대/최소 {ratio:>7.2f}  "
                      f"인접 최대 급증 {jump:>6.2f}")
            print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--t-end", type=float, nargs="+", default=[1.0, 4.0, 12.0, 32.0])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    for name, fn in (("T0", t0), ("T1", t1), ("T2", t2), ("T3", t3)):
        if args.only and name not in args.only:
            continue
        fn(args.R) if name == "T0" else fn(args.R, args.t_end)


if __name__ == "__main__":
    main()
