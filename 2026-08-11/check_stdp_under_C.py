"""갈래 C(gate_write)가 PLASTICITY.md §5.3 의 위상-STDP 해석을 깨는가.

§5.3 의 주장 셋을 하나씩 건다.
  (1) 역할(pre/post)은 토큰 순서가 정한다
  (2) 부호(LTP/LTD)는 위상이 정한다  — 시간이 아니라
  (3) ψ 가 헤비안(대칭 g) ↔ STDP형(반대칭 ω) 다이얼이다  (§3)
"""
import torch

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 레포 루트

from model1 import Model1, Model1Config
from selective_copy import TaskConfig, make_batch

torch.manual_seed(0)

tcfg = TaskConfig(l_noise=32, n_mem=4, n_data=16)
x, y = make_batch(tcfg, batch=8, generator=torch.Generator().manual_seed(0))

base = dict(d=128, p=64, R=4, vocab_size=tcfg.vocab_size,
            psi_inhib=True, self_exp=True)  # 개입 1+2 = 갈래 C 의 표준 바닥


def build(gate_write):
    torch.manual_seed(0)
    m = Model1(Model1Config(membrane=True, gate_write=gate_write, **base))
    m.eval()
    with torch.no_grad():
        m.calibrate_threshold(x)
    return m


gw = build(True)
ob = build(False)

print("=" * 78)
print("[1] 커널 A 는 게이트를 받는가 — §5.3 의 부호가 사는 곳")
print("=" * 78)
with torch.no_grad():
    h = gw.embed(x)
    A_nogate = gw.block(h, mode="parallel", return_A=True, gate=None)[1]
    g_fake = torch.rand_like(h)
    A_gated = gw.block(h, mode="parallel", return_A=True, gate=g_fake)[1]
print(f"  같은 h, 게이트만 바꿈:  max|A_gated − A_nogate| = {(A_gated - A_nogate).abs().max():.3e}")
print("  → A 는 h 만의 함수다. 게이트는 value 경로에만 걸린다.")

print()
print("=" * 78)
print("[2] 게이트 g 의 부호 — 양수면 부호를 뒤집을 수 없다")
print("=" * 78)
with torch.no_grad():
    _, tr = gw(x, return_trace=True)
    gf = torch.sigmoid(gw.g_floor_raw)
    s_all = torch.cat([t["s"].reshape(-1) for t in tr if "s" in t])
    g_all = gf.repeat(s_all.numel() // gf.numel() + 1)[: s_all.numel()]
    g_eff = torch.cat([(gf + (1 - gf) * t["s"]).reshape(-1) for t in tr if "s" in t])
print(f"  g = g_floor + (1−g_floor)·s")
print(f"  g_floor 범위 : [{gf.min():.4f}, {gf.max():.4f}]")
print(f"  s      범위 : [{s_all.min():.4f}, {s_all.max():.4f}]")
print(f"  g      범위 : [{g_eff.min():.6f}, {g_eff.max():.6f}]   음수 개수 = {(g_eff <= 0).sum().item()}")

print()
print("=" * 78)
print("[3] §5.3 의 부호 분포 — lag 별 개별 기여의 LTP/LTD 비율")
print("=" * 78)
print("  기여 = a_tn · v_{n,e}.  게이트가 걸리면 v = g⊙h 다.")


def sign_table(model, tag):
    with torch.no_grad():
        h = model.embed(x)
        _, tr = model(x, return_trace=True)
        A = model.block(h, mode="parallel", return_A=True, gate=None)[1]
        # 첫 깊이 단계의 게이트는 항상 1 이므로 두 번째 단계(r=1)의 s 를 쓴다
        s_prev = tr[0]["s"]
        gf = torch.sigmoid(model.g_floor_raw)
        g = gf + (1 - gf) * s_prev if model.cfg.gate_write else torch.ones_like(h)
        v = g * h
        rows = []
        for lag in (1, 3, 5, 10, 20):
            t_idx = torch.arange(lag, A.shape[1])
            a = A[:, t_idx, t_idx - lag]              # (B, T')
            contrib = a.unsqueeze(-1) * v[:, t_idx - lag]  # (B, T', d)
            pos = (contrib > 0).float().mean().item()
            rows.append((lag, pos * 100, (1 - pos) * 100))
    print(f"\n  --- {tag} ---")
    print("   Δ    양(LTP)   음(LTD)")
    for lag, p, n in rows:
        print(f"  {lag:2d}    {p:6.2f}%   {n:6.2f}%")
    return rows


r_ob = sign_table(ob, "gate_write=False (관측자)")
r_gw = sign_table(gw, "gate_write=True  (되먹임)")
print()
print("  차이:", " ".join(f"Δ{l}:{abs(a[1]-b[1]):.4f}%p" for l, a, b in
                          [(r[0], r, s) for r, s in zip(r_ob, r_gw)]))

print()
print("=" * 78)
print("[4] 부호 보존 정리 — 같은 h 에서 게이트 on/off 의 기여 부호 대조")
print("=" * 78)
with torch.no_grad():
    h = gw.embed(x)
    _, tr = gw(x, return_trace=True)
    gf = torch.sigmoid(gw.g_floor_raw)
    g = gf + (1 - gf) * tr[0]["s"]
    A = gw.block(h, mode="parallel", return_A=True, gate=None)[1]
    c_plain = torch.einsum("btn,bne->btne", A, h)
    c_gated = torch.einsum("btn,bne->btne", A, g * h)
    same = (torch.sign(c_plain) == torch.sign(c_gated)).float().mean().item()
    ratio = (c_gated.abs() + 1e-30) / (c_plain.abs() + 1e-30)
print(f"  부호가 같은 원소 비율 : {same*100:.4f}%   (전체 {c_plain.numel():,} 개)")
print(f"  크기 비 |gated|/|plain| : [{ratio.min():.4f}, {ratio.max():.4f}]")
print("  → 게이트는 크기만 줄인다. 부호는 여전히 위상이 정한다.")

print()
print("=" * 78)
print("[5] §3 의 ψ 다이얼 — 대칭(헤비안) / 반대칭(STDP형) 분해")
print("=" * 78)
print("  §3.4 는 커널 K 를 분해했다. 게이트가 걸린 유효 연산자는 K·g 다.")
with torch.no_grad():
    h = gw.embed(x)
    _, tr = gw(x, return_trace=True)
    gf = torch.sigmoid(gw.g_floor_raw)
    g = gf + (1 - gf) * tr[0]["s"]
    K = gw.block.attention_matrix(h)          # 인과 마스크 포함
    Kf = K + K.transpose(1, 2) * 0            # 그대로
    gs = g.mean(-1)                            # (B,T) 채널 평균 게이트

    def decomp(M, tag):
        S = 0.5 * (M + M.transpose(1, 2))
        Aa = 0.5 * (M - M.transpose(1, 2))
        ns, na = S.norm().item(), Aa.norm().item()
        print(f"  {tag:26s} ‖S‖={ns:8.4f}  ‖A‖={na:8.4f}  반대칭 {100*na/(ns+na):5.1f}%")

    decomp(Kf, "K (커널)")
    decomp(Kf * gs.unsqueeze(1), "K·g (게이트된 유효)")

print()
print("=" * 78)
print("[6] post 항 — 스캔 내부 vs 깊이축")
print("=" * 78)
print("  PLASTICITY.md §6.1: 상태 S 를 흔들어도 쓰기 항이 0 만큼 변하면 post 가 없다.")
with torch.no_grad():
    h = gw.embed(x)
    # (a) 스캔 내부: 같은 깊이 단계에서 g 는 이미 상수다
    tr = gw(x, return_trace=True)[1]
    g1 = torch.sigmoid(gw.g_floor_raw)
    g1 = g1 + (1 - g1) * tr[0]["s"]
    g2 = torch.sigmoid(gw.g_floor_raw)
    g2 = g2 + (1 - g2) * tr[0]["s"]
    # 주의: 이 줄은 같은 계산을 두 번 해서 비교하므로 **순환**이다 (측정이 아님).
    # 스캔 내부에 되먹임이 없다는 것은 코드 구조가 보장한다 — forward() 에서
    # `s_prev = s_now` 가 블록 호출 **뒤에** 대입되므로 r 단계 스캔 동안 g 는 고정이다.
    print(f"  스캔 내부: (구조적 사실) s_prev 는 스캔 전 확정 → g 고정. 재계산 차이 "
          f"{(g1-g2).abs().max():.3e}  → 병렬 유지")
    # (b) 깊이축: r 단계 h 를 흔들면 r+1 의 g 가 변하는가
    hpert = h * 1.01
    m0, s0, _ = gw.membrane_scan(gw.block(h, gate=None), torch.zeros_like(h))
    m1, s1, _ = gw.membrane_scan(gw.block(hpert, gate=None), torch.zeros_like(h))
    print(f"  깊이축  : h^(r) 를 1% 흔들면 s^(r) 가 변하는가 → "
          f"max|Δs| = {(s1-s0).abs().max():.4e}   (≠0 → post 항 존재)")
