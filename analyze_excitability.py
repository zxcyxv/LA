"""흥분성 측정 — f_t 의 '기억 의존 선형 이득' M(S_{t-1}) 의 스펙트럼.

이 스크립트가 존재하는 이유:
    f_t 를 h_t 에 대해 갈랐을 때 나오는 선형항을 놓치기가 매우 쉽다.
    놓치면 "이 구조에는 FHN 의 +u 에 해당하는 요소가 없다"는 틀린 결론에 도달한다.
    자세한 함정은 EXCITABILITY.md 참조.

분해:
    S_t = Λ S_{t-1} + k_t h_tᵀ
    f_t = p^{-1/2} Re[q_t† S_t]
        = p^{-1/2} Re[q_t† Λ S_{t-1}]  +  p^{-1/2} Re[q_t† k_t h_tᵀ]
        = M(S_{t-1}) h_t               +  a_tt h_t
          └─ h_t 에 선형 ─┘               └─ h_t 에 3차 ─┘

    q_t 가 h_t 에 선형이고 S_{t-1} 은 과거만의 함수이므로 첫 항은 정확히 선형이다.
    그 계수 행렬이 대화내역2 가 유도한 λ(S_{t-1}) 이다.

    M[d,e] = p^{-1/2} Re[ Σ_j conj(rot_j) · conj(W_C[j,e]) · λ_j · S_{t-1}[j,d] ]
             rot_j = e^{iψ_j/2} (gamma_split 이면 ×√γ_j)

흥분성 판정:
    max Re λ(M) > 0  →  그 방향의 섭동이 스스로 커진다 = 원점 불안정 = 흥분 가능
"""

from __future__ import annotations

import argparse
import math

import torch

from model1 import Model1, Model1Config
from selective_copy import BLANK, DATA_OFFSET, MARKER, TaskConfig, make_batch


def load(ckpt: str, dev: str):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    old = ck["mcfg"]
    keep = ("d", "p", "R", "vocab_size", "w_scale", "r_min", "r_max")
    cfg = Model1Config(**{k: getattr(old, k) for k in keep})
    m = Model1(cfg).to(dev).double()
    m.load_state_dict({k: v.double() for k, v in ck["state"].items()})
    m.eval()
    return m, cfg, ck["task"]


def linear_gain(block, S_prev, W, rot, p):
    """M(S_{t-1}) ∈ R^{B,d,d} — h_t 에 대한 선형 계수 행렬."""
    B = S_prev.shape[0]
    C = rot.conj().unsqueeze(0).expand(B, -1)  # (B,p)
    return torch.einsum("bp,pe,bpd->bde", C, W.conj(), S_prev).real / math.sqrt(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_model1_R4.pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--verify", action="store_true", help="autograd 야코비안과 대조")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, cfg, task = load(args.ckpt, dev)
    b, p = m.block, cfg.p

    torch.manual_seed(5)
    x, _ = make_batch(task, args.batch, dev)

    with torch.no_grad():
        h = m.embed(x)
        _, q, k = b.project(h)
        omega = b.phase_increment(h)
        W = torch.complex(b.W_re, b.W_im)
        rot = torch.polar(torch.ones_like(b.psi), 0.5 * b.psi)
        if getattr(cfg, "gamma_split", False):
            rot = rot * torch.sqrt(b.gamma).to(rot.dtype)

        B, T, d = h.shape
        S = torch.zeros(B, p, d, dtype=q.dtype, device=dev)
        stats = {"blank": [], "data": [], "marker": []}
        norms = []

        for t in range(T):
            lam = torch.polar(torch.exp(-b.alpha), omega[:, t])  # (B,p)
            S_prev = lam.unsqueeze(-1) * S
            M = linear_gain(b, S_prev, W, rot, p)  # (B,d,d)

            if args.verify and t == T - 1:
                hv = h[0, t].clone().requires_grad_(True)

                def past(u):
                    z = u.to(W.dtype) @ W.t()
                    return (( (z * rot).conj().unsqueeze(-1) * S_prev[0]).sum(0)).real / math.sqrt(p)

                J = torch.autograd.functional.jacobian(past, hv)
                print(f"[검증] M 해석식 vs autograd 야코비안: {(J - M[0]).abs().max().item():.3e}\n")

            ev = torch.linalg.eigvals(M).real.max(-1).values  # (B,)
            norms.append(torch.linalg.matrix_norm(M, ord=2))
            for name, msk in [
                ("blank", x[:, t] == BLANK),
                ("data", x[:, t] >= DATA_OFFSET),
                ("marker", x[:, t] == MARKER),
            ]:
                if msk.any():
                    stats[name].append(ev[msk])

            S = S_prev + k[:, t].unsqueeze(-1) * h[:, t].to(q.dtype).unsqueeze(1)

    print(f"체크포인트: {args.ckpt}   (T={task.seq_len}, d={cfg.d}, p={p}, R={cfg.R})\n")
    print("기억 의존 선형 이득 M(S_{t-1}) 의 스펙트럼")
    print(f"  {'토큰':>8} {'max Re λ(M)':>14} {'>0 비율':>10}")
    for name, v in stats.items():
        if not v:
            continue
        v = torch.cat(v)
        print(f"  {name:>8} {v.mean():>14.4f} {(v > 0).double().mean():>10.2%}")
    print("\n  max Re λ(M) > 0  →  원점이 불안정한 방향 존재 = 흥분 가능")
    nn = torch.cat(norms)
    print(f"\n  ‖M‖₂ 평균 {nn.mean():.4f}   >1 비율 {(nn > 1).double().mean():.2%}")


if __name__ == "__main__":
    main()
