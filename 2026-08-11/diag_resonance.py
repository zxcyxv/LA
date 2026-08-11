"""보강간섭이 스파이킹을 일으키는가 — 두 가지로 가른다.

[R1] 결맞음(coherence) 시험
     같은 에너지, 같은 펄스 개수. 부호만 (a) 전부 +v  (b) 무작위 ±v.
     보강간섭이 작동하면 (a) 가 (b) 보다 √n 배 크게 쌓여야 한다.

[R2] 주파수 응답 (resonate-and-fire)
     펄스 **개수를 고정**하고 간격만 바꾼다 (에너지 통제). 채널별 회전 θ_j 와
     자극 주기가 맞으면 공진해야 한다. 공진 봉우리가 있으면 R&F 뉴런이다.
"""
import numpy as np
import torch

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 레포 루트

from model1 import Model1, Model1Config

torch.manual_seed(0)
D_MODEL, P, R = 128, 64, 4
T = 256
NP = 8  # 펄스 개수 — 두 시험 모두에서 고정한다

cfg = Model1Config(d=D_MODEL, p=P, R=R, vocab_size=32,
                   membrane=True, psi_inhib=True, self_exp=True)
model = Model1(cfg).eval()

g = torch.Generator().manual_seed(0)
bg = torch.randint(0, 32, (1, T), generator=g)  # 배경 토큰
with torch.no_grad():
    model.calibrate_threshold(bg, q=0.995)


@torch.no_grad()
def run(pulse_pos, signs, amp, vec):
    """지정 위치에 amp·sign·vec 을 주입하고 (m, s) 를 반환."""
    pv = torch.zeros(1, T, D_MODEL)
    for pos, sg in zip(pulse_pos, signs):
        pv[0, pos] = amp * sg * vec
    tr = model(bg, return_trace=True, pulse={0: pv})[1]
    last = [x for x in tr if "m" in x][-1]
    return last["m"][0], last["s"][0]          # (T,d)


with torch.no_grad():
    base_m, base_s = run([], [], 0.0, torch.zeros(D_MODEL))

vec = torch.randn(D_MODEL, generator=g)
vec = vec / vec.norm()

# ---- 응답 채널을 **한 번** 고정 (§9.5 의 교훈 3) ----
with torch.no_grad():
    ref_pos = list(range(0, NP * 5, 5))
    m_ref, _ = run(ref_pos, [1.0] * NP, 3.0, vec)
CH = int((m_ref[ref_pos[-1]] - base_m[ref_pos[-1]]).abs().argmax())
print(f"응답 채널 ch={CH} 로 고정.  무자극 발화율 = {base_s.mean():.4f}")
print(f"θ 분포: 주기 2π/θ 의 [10,50,90]분위 = "
      f"{np.percentile((2*np.pi/model.block.theta.detach().abs().clamp(min=1e-6)).numpy(), [10,50,90]).round(2)}")

# =============================================================== R1
print()
print("=" * 84)
print("[R1] 결맞음 — 같은 에너지, 부호만 다르게")
print("=" * 84)
print("  보강간섭이 작동하면 결맞은 쪽이 크게 쌓인다. 무작위 부호는 √n 로만 자란다.")
print()
print(f"  {'진폭':>6s}  {'Δm 결맞음':>10s} {'Δm 무작위':>10s} {'비':>6s}   "
       f"{'s 결맞음':>9s} {'s 무작위':>9s}")
print("-" * 84)
pos = list(range(0, NP * 5, 5))
for amp in (0.5, 1.0, 2.0, 4.0, 8.0):
    with torch.no_grad():
        m_c, s_c = run(pos, [1.0] * NP, amp, vec)
        dm_c = (m_c[pos[-1], CH] - base_m[pos[-1], CH]).item()
        sc = s_c[pos[-1], CH].item()
        # 무작위 부호 8 회 평균
        dms, ss = [], []
        for seed in range(8):
            gg = torch.Generator().manual_seed(100 + seed)
            sg = (torch.randint(0, 2, (NP,), generator=gg) * 2 - 1).float().tolist()
            m_r, s_r = run(pos, sg, amp, vec)
            dms.append((m_r[pos[-1], CH] - base_m[pos[-1], CH]).item())
            ss.append(s_r[pos[-1], CH].item())
    ratio = abs(dm_c) / (np.mean(np.abs(dms)) + 1e-12)
    print(f"  {amp:6.1f}  {dm_c:10.4f} {np.mean(dms):10.4f} {ratio:6.2f}   "
          f"{sc:9.4f} {np.mean(ss):9.4f}")
print()
print(f"  참고: 완전 결맞음/무작위의 이론 비 ≈ √{NP} = {np.sqrt(NP):.2f}")

# =============================================================== R2
print()
print("=" * 84)
print("[R2] 주파수 응답 — 펄스 개수 고정, 간격만 스윕 (에너지 통제)")
print("=" * 84)
print(f"  펄스 {NP}개, 진폭 고정. 간격 T_stim 만 바꾼다.")
print("  공진하면 특정 간격에서 Δm·s 가 봉우리를 만든다.")
print()
print(f"  {'T_stim':>7s}  {'Δm':>9s}  {'s':>8s}   {'전체 발화율':>10s}")
print("-" * 84)
rows = []
for ts in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24):
    p2 = list(range(0, NP * ts, ts))
    if p2[-1] >= T:
        continue
    with torch.no_grad():
        m_x, s_x = run(p2, [1.0] * NP, 3.0, vec)
    dm = (m_x[p2[-1], CH] - base_m[p2[-1], CH]).item()
    rows.append((ts, dm, s_x[p2[-1], CH].item(), s_x.mean().item()))
    print(f"  {ts:7d}  {dm:9.4f}  {rows[-1][2]:8.4f}   {rows[-1][3]:10.4f}")

dms = np.array([r[1] for r in rows])
best = rows[int(np.argmax(np.abs(dms)))]
print()
print(f"  |Δm| 최대 = T_stim {best[0]}  ({abs(best[1]):.4f}),  "
      f"최소 = T_stim {rows[int(np.argmin(np.abs(dms)))][0]} "
      f"({abs(dms).min():.4f}),  최대/최소 = {abs(dms).max()/max(abs(dms).min(),1e-12):.2f}")
