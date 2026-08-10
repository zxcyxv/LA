"""갈래 C 구조 진단 — t축 선형 막전위 + 점별 문턱 + 깊이축 리셋.

**학습하지 않는다.** `gate_write=False` 이므로 막전위는 **순수 관측자**이고 `h` 동역학을
전혀 바꾸지 않는다. 따라서 여기서 재는 것은 순수하게 "구조가 생겼는가" 다.

    [C0] 통제 환원 — membrane off 시 기존과 비트 단위 동일 / on 이어도 h 가 불변인가
    [C4] **완화 반복 수렴** — s^(r) 가 깊이에 대해 수렴하는가. 설계의 사활
    [C1] 문턱하 누적 — 약한 자극을 k 번 주면 k 번째에 발화하는가 (갈래 A 가 못 하는 것)
    [C2] 전부아니면전무 — s 분포가 이봉인가, 자극-발화 전이가 급한가
    [C3] 불응기 — 발화 직후 재발화가 억제되는가 (깊이축 리셋이 실제로 작동하는가)

사용:
    python diag_membrane_c.py
    python diag_membrane_c.py --only C4 C1
"""

from __future__ import annotations

import argparse
import math

import torch

from model1 import Model1, Model1Config

BASE = dict(d=128, p=64, vocab_size=18, psi_inhib=True, self_exp=True)  # 개입 1+2 표준
T_SEQ = 40
R_CONV = 16   # C4 결과: 완화 반복 수렴에 R~16 필요 (R=4 는 미수렴)


def build(R=4, membrane=True, seed=0, **over):
    torch.manual_seed(seed)
    kw = dict(BASE, R=R, **over)
    if membrane:
        kw["membrane"] = True
    m = Model1(Model1Config(**kw)).eval()
    return m


def run(m, tok, pulse=None):
    with torch.no_grad():
        return m(tok, return_trace=True, pulse=pulse)[1]


def calibrate(m, tok, q=0.995, sharp=0.05):
    """theta 를 무자극 m 분포의 **높은** 분위로 맞춘다.

    90분위로 잡으면 기저 발화율이 10% 여서 '항상 켜진 채널' 이 생기고, 자극이 문턱을
    넘는지를 볼 수 없다 (자극 없이 이미 넘어 있으므로). 문턱 시험은 기저가 문턱 **아래**
    여야 성립한다. tau_s 도 m 의 스케일에 맞춰 좁힌다.
    """
    tr = run(m, tok)
    mm = torch.cat([s["m"] for s in tr if "m" in s], dim=0)
    flat = mm.reshape(-1, mm.shape[-1]).float()
    with torch.no_grad():
        m.theta.copy_(torch.quantile(flat, q, dim=0))
        m.tau_s_raw.fill_(math.log(math.expm1(sharp * mm.std().item())))
    return mm.std().item()


def unit_dir(d, seed=3):
    torch.manual_seed(seed)
    v = torch.randn(1, 1, d)
    return v / v.norm(dim=-1, keepdim=True)


def stim(m, tok, positions, A, direction):
    """자극 유무 두 런의 마지막 깊이 단계 (m, s) 를 돌려준다. **차이**로 재기 위한 것."""
    base = run(m, tok)
    p = torch.zeros_like(base[0]["h"])
    for t0 in positions:
        p[:, t0 : t0 + 1, :] = A * direction * math.sqrt(m.cfg.d)
    tr = run(m, tok, pulse={0: p})
    last = lambda t, k: [x[k] for x in t if k in x][-1]
    return (last(base, "m"), last(base, "s"), last(tr, "m"), last(tr, "s"))


# --------------------------------------------------------------------------- #
def c0(_):
    print("=" * 78)
    print("[C0] 통제 환원 — membrane 은 gate_write=False 에서 순수 관측자여야 한다")
    print("=" * 78)
    tok = torch.randint(0, 18, (3, T_SEQ))
    for R in (4, 8):
        off1, off2 = build(R, False, seed=1), build(R, False, seed=1)
        on = build(R, True, seed=1)
        with torch.no_grad():
            d0 = (off1(tok) - off2(tok)).abs().max().item()
            d1 = (on(tok) - off1(tok)).abs().max().item()
        print(f"  R={R}  off 재현성 {d0:.3e}   |  membrane on 시 logits 변화 {d1:.3e}")
    print("  두 열 모두 0 이어야 한다 — 관측자는 아무것도 바꾸지 않는다.\n")


def c4(Rs):
    print("=" * 78)
    print("[C4] 완화 반복 수렴 — 설계 C 의 사활")
    print("=" * 78)
    print("  s^(r) 는 s^(r-1) 로 리셋한 스캔의 결과다. 이 반복이 자기정합적 스파이크열로")
    print("  수렴하지 않으면(진동/발산) 설계 C 는 성립하지 않는다.\n")
    tok = torch.randint(0, 18, (4, T_SEQ))
    for R in Rs:
        m = build(R)
        sd = calibrate(m, tok)
        tr = run(m, tok)
        ss = [s["s"] for s in tr if "s" in s]
        diffs = [(ss[i] - ss[i - 1]).abs().mean().item() for i in range(1, len(ss))]
        rate = [ss[i].mean().item() for i in range(len(ss))]
        print(f"  R={R:>2}  (m 의 std={sd:.3g})")
        print(f"      발화율 s.mean() : {' '.join(f'{x:.4f}' for x in rate)}")
        print(f"      |s^r − s^(r-1)|: {' '.join(f'{x:.4f}' for x in diffs)}")
        if len(diffs) >= 2:
            trend = "수렴" if diffs[-1] < diffs[0] else "**비수렴/진동**"
            print(f"      → {trend}  (첫 {diffs[0]:.4f} → 끝 {diffs[-1]:.4f})")
        print()


def c1(_):
    print("=" * 78)
    print("[C1] 문턱하 누적 — 약한 자극 k 번 뒤에 발화하는가")
    print("=" * 78)
    print("  갈래 A(깊이축 FHN)가 원리적으로 못 하는 성질. t 축에 막이 있어야 생긴다.")
    print("  자극 위치 5,10,15,20,25 에 **같은 크기**를 반복 주입하고, 자극에 의한")
    print("  변화량 Δm 이 위치를 따라 **누적**되는지, 그리고 몇 번째에서 문턱을 넘는지 본다.\n")
    tok = torch.randint(0, 18, (2, T_SEQ))
    m = build(R_CONV)
    sd = calibrate(m, tok)
    pos = [5, 10, 15, 20, 25]
    dirv = unit_dir(m.cfg.d)
    print(f"  (무자극 m 의 std = {sd:.3g},  기저 발화율 = 0.5% 로 보정)\n")
    print(f"  {'A':>6} {'Δm at 5':>9} {'@10':>8} {'@15':>8} {'@20':>8} {'@25':>8} "
          f"{'첫 발화':>8}  누적")
    for A in (0.25, 0.5, 1.0, 2.0, 4.0):
        mb, sb, mp, sp = stim(m, tok, pos, A, dirv)
        dm, ds = mp - mb, sp - sb
        ch = ds[:, pos, :].abs().mean(dim=(0, 1)).argmax().item()  # 자극이 가장 움직인 채널
        vals = [dm[:, t, ch].mean().item() for t in pos]
        fired = [i for i, t in enumerate(pos)
                 if sp[:, t, ch].mean() > 0.5 and sb[:, t, ch].mean() < 0.5]
        fst = f"{pos[fired[0]]}" if fired else "-"
        inc = sum(abs(vals[i + 1]) > abs(vals[i]) for i in range(4))
        print(f"  {A:>6.2f} " + " ".join(f"{v:>8.4f}" for v in vals) +
              f" {fst:>8}  단조증가 {inc}/4")
    print("\n  누적이 있으면: |Δm| 이 위치를 따라 커지고, A 가 작을수록 첫 발화가 늦어진다.\n")


def c2(_):
    print("=" * 78)
    print("[C2] 전부아니면전무 — 자극받은 토큰에서 Δs 가 급전이하는가")
    print("=" * 78)
    print("  전체 평균이 아니라 **자극 토큰의 가장 영향받은 채널**에서 잰다.\n")
    tok = torch.randint(0, 18, (2, T_SEQ))
    m = build(R_CONV)
    calibrate(m, tok)
    dirv = unit_dir(m.cfg.d)
    t0 = 20
    # 채널을 **한 번** 고정한다. A 마다 argmax 를 다시 고르면 행마다 다른 채널을 재게
    # 되어 기저값이 0.37/0.00 으로 튀고 비교가 무의미해진다 (실제로 그렇게 나왔다).
    _, sb0, _, sp0 = stim(m, tok, [t0], 3.0, dirv)
    ch = (sp0 - sb0)[:, t0, :].abs().mean(0).argmax().item()
    print(f"  A=3.0 에서 가장 크게 움직인 채널 {ch} 로 **고정**한다.\n")
    print(f"  {'A':>7} {'Δm':>9} {'s(무자극)':>10} {'s(자극)':>9} {'Δs':>8}  기울기")
    rows = []
    for A in torch.logspace(-1.0, 1.2, 10):
        mb, sb, mp, sp = stim(m, tok, [t0], float(A), dirv)
        rows.append((float(A), (mp - mb)[:, t0, ch].mean().item(),
                     sb[:, t0, ch].mean().item(), sp[:, t0, ch].mean().item()))
    for i, (A, dm, s0, s1) in enumerate(rows):
        sl = ""
        if i:
            dA = math.log(A) - math.log(rows[i - 1][0])
            sl = f"{(s1 - rows[i-1][3]) / dA:>7.3f}"
        print(f"  {A:>7.3f} {dm:>9.4f} {s0:>10.4f} {s1:>9.4f} {s1-s0:>8.4f}  {sl}")
    print()


def c3(_):
    print("=" * 78)
    print("[C3] 불응기 — 발화 직후 재발화가 억제되는가 (깊이축 리셋의 작동 확인)")
    print("=" * 78)
    tok = torch.randint(0, 18, (2, T_SEQ))
    m = build(R_CONV)
    calibrate(m, tok)
    dirv = unit_dir(m.cfg.d)
    A = 6.0
    # 채널을 **단일 펄스**로 한 번 고정한다. Δt 마다 argmax 를 다시 고르면 1차 응답
    # 자체가 행마다 0.50/0.55/0.84 로 달라져 비율이 무의미해진다 (실제로 그렇게 나왔다).
    _, sb0, _, sp0 = stim(m, tok, [10], A, dirv)
    ch = (sp0 - sb0)[:, 10, :].abs().mean(0).argmax().item()
    solo = (sp0 - sb0)[:, 10, ch].mean().item()
    print(f"  채널 {ch} 고정.  펄스 하나만 줬을 때의 Δs = {solo:.4f} (= 기준)\n")
    print(f"  {'Δt':>4} {'Δs(1차)':>9} {'Δs(2차)':>9} {'2차/1차':>9} {'2차/단독':>9}")
    for dtk in (1, 2, 3, 4, 5, 8, 12, 20):
        mb, sb, mp, sp = stim(m, tok, [10, 10 + dtk], A, dirv)
        ds = sp - sb
        r1 = ds[:, 10, ch].mean().item()
        r2 = ds[:, 10 + dtk, ch].mean().item()
        print(f"  {dtk:>4} {r1:>9.4f} {r2:>9.4f} {r2 / (abs(r1) + 1e-9):>9.4f} "
              f"{r2 / (abs(solo) + 1e-9):>9.4f}")
    print("\n  불응기가 있으면 Δt 가 작을 때 비율이 낮고 Δt 가 커지며 1 로 회복.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    for name, fn, needs_R in (("C0", c0, True), ("C4", c4, True), ("C1", c1, False),
                              ("C2", c2, False), ("C3", c3, False)):
        if a.only and name not in a.only:
            continue
        fn(a.R)


if __name__ == "__main__":
    main()
