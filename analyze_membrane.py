"""학습된 Model 1 을 막전위 동역학으로 읽을 수 있는지 직접 측정한다.

블록 출력은 항등적으로 다음처럼 갈린다.

    f_t = Σ_n a_tn (h_n - h_t)   +   d_t h_t          d_t = Σ_n a_tn
          └── 확산항(누전) ──┘       └─ 반응항(이득) ─┘

질문: "평소에는 의미 없는 입력을 누전시키다가, 중요한 자극이 오면 공명해서 솟는가?"
→ 토큰 종류별로 d_t 와 두 항의 크기를 재면 된다.

예측(막전위 해석이 맞다면):
    blank  : d_t 작음/음수, 확산항 지배  → 누전 모드
    data   : d_t 큼,       반응항 지배  → 공명·증폭 모드
"""

from __future__ import annotations

import argparse

import torch

from model1 import Model1, Model1Config
from selective_copy import BLANK, DATA_OFFSET, MARKER, TaskConfig, make_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_model1_R4.pt")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    task: TaskConfig = ck["task"]
    mcfg: Model1Config = ck["mcfg"]
    model = Model1(mcfg).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()

    torch.manual_seed(77)
    x, _ = make_batch(task, args.batch, dev)
    kinds = {
        "blank": x == BLANK,
        "data": x >= DATA_OFFSET,
        "marker": x == MARKER,
    }

    print(f"체크포인트: {args.ckpt}   R={mcfg.R}\n")
    print("토큰 종류별  d_t = Σ_n a_tn  (반응 이득) 및 확산/반응 항 크기")
    print(f"{'r':>3} {'종류':>8} {'d_t':>10} {'‖확산항‖':>11} {'‖반응항‖':>11} {'반응/확산':>10} {'a_tt':>9}")

    with torch.no_grad():
        h = model.embed(x)
        for r in range(mcfg.R):
            f, A = model.block(h, mode="parallel", return_A=True)
            d = A.sum(-1)  # (B,T)
            att = torch.diagonal(A, dim1=-2, dim2=-1)  # (B,T)
            react = d.unsqueeze(-1) * h  # (B,T,d)
            diff = f - react  # 항등식의 나머지가 확산항
            dn, rn = diff.norm(dim=-1), react.norm(dim=-1)
            keep = torch.ones_like(d, dtype=torch.bool)
            keep[:, 0] = False  # t=0 은 확산항이 정확히 0 (자기 자신뿐) → 비율에서 제외
            for name, m in kinds.items():
                mk = m & keep
                # 비율은 '평균의 비'로 낸다. '비의 평균'은 0 나눗셈에 오염된다.
                print(
                    f"{r:>3} {name:>8} {d[mk].mean():>10.4f} {dn[mk].mean():>11.4f} "
                    f"{rn[mk].mean():>11.4f} {(rn[mk].mean() / dn[mk].mean()):>10.4f} "
                    f"{att[mk].mean():>9.4f}"
                )
            print()
            h = h + f / mcfg.R

    # 스텝별 상태 변화율: 누전 모드면 작고, 공명하면 커야 한다
    print("토큰 종류별  ‖Δh‖ / ‖h‖  (한 스텝에서 상태가 얼마나 움직였나)")
    with torch.no_grad():
        h = model.embed(x)
        print(f"{'r':>3} " + " ".join(f"{k:>10}" for k in kinds))
        for r in range(mcfg.R):
            f = model.block(h, mode="parallel")
            rel = (f / mcfg.R).norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-9)
            print(f"{r:>3} " + " ".join(f"{rel[m].mean():>10.4f}" for m in kinds.values()))
            h = h + f / mcfg.R


    # 가장 직접적인 시험: 의미 없는 입력(전부 빈칸) vs 정상 시퀀스
    print("\n무의미 입력 vs 유의미 입력  — 같은 모델, 같은 길이")
    from model0 import dirichlet_energy, euler_gain

    blankseq = torch.full_like(x, BLANK)
    for name, xx in [("전부 빈칸", blankseq), ("정상 시퀀스", x)]:
        with torch.no_grad():
            h = model.embed(xx)
            E0 = dirichlet_energy(h).mean().item()
            fs, rhos = [], []
            for r in range(mcfg.R):
                f, A = model.block(h, mode="parallel", return_A=True)
                fs.append((f.norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-9)).mean().item())
                rhos.append(euler_gain(A, mcfg.R).mean().item())
                h = h + f / mcfg.R
            E1 = dirichlet_energy(h).mean().item()
        print(
            f"  {name:>10}:  ‖f‖/‖h‖ = [{', '.join(f'{v:.3f}' for v in fs)}]   "
            f"ρ_R = [{', '.join(f'{v:.3f}' for v in rhos)}]   "
            f"E_D {E0:.1f} → {E1:.1f} ({E1 / max(E0, 1e-9):.2f}×)"
        )


if __name__ == "__main__":
    main()
