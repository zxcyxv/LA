"""학습된 Model 1 이 실제로 찾은 위상 불변량을 측정한다.

§4 유도가 예측한 해는 "빈칸 0, 데이터=마커=c → 위상차가 항상 4c" 였다.
그런데 학습된 게이트는 반대로 나왔다 (빈칸이 크게, 데이터/마커가 작게).
따라서 예측을 그대로 확인하는 대신, 모델이 무슨 불변량을 쓰는지 직접 잰다.

측정 항목:
  (A) 마커 i → 정답 데이터 i 의 위상차가 배치에 걸쳐 집중돼 있는가
      (집중돼 있다 = 그 위상차가 '주소'로 쓰인다)
  (B) 정답 데이터와 오답 데이터의 위상차 분포가 분리되는가
  (C) 어텐션이 실제로 정답 데이터에 몰리는가 (R=1 이면 합성 = I + A)
  (D) 커널값 cos(Δφ - ΔΦ) 가 정답에서 높은가
"""

from __future__ import annotations

import argparse
import math

import torch

from model1 import Model1, Model1Config
from selective_copy import DATA_OFFSET, TaskConfig, accuracy, make_batch


def circ_stats(x: torch.Tensor, dim=0):
    """원형 통계: 평균 방향과 집중도 R (0=균등, 1=완전집중)."""
    c, s = torch.cos(x).mean(dim), torch.sin(x).mean(dim)
    return torch.atan2(s, c), torch.sqrt(c**2 + s**2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    task: TaskConfig = ck["task"]
    mcfg: Model1Config = ck["mcfg"]
    model = Model1(mcfg).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    blk = model.block

    torch.manual_seed(4321)
    x, y = make_batch(task, args.batch, dev)
    with torch.no_grad():
        logits, tr = model(x, return_trace=True)
    ta, sa = accuracy(logits, task, y)
    print(f"체크포인트:  R={mcfg.R}  tok_acc={ta:.4f}  seq_acc={sa:.4f}\n")

    B, T = x.shape
    l, n_mem = task.l_noise, task.n_mem
    bidx = torch.arange(B, device=dev)
    is_data = x[:, :l] >= DATA_OFFSET
    data_pos = is_data.float().argsort(dim=-1, descending=True, stable=True)[:, :n_mem]
    data_pos = data_pos.sort(dim=-1).values  # (B,n_mem)

    with torch.no_grad():
        h0 = tr[0]["h"]
        Phi = blk.cumulative_phase(h0)  # (B,T,p)

        # 가장 대비가 큰 '세는 채널' 고르기
        g = blk.gate(h0)
        dth = blk.s_raw * g
        contrast = (dth[x >= DATA_OFFSET].mean(0) - dth[x == 0].mean(0)).abs()

        # 모든 채널의 '정답쌍 위상차 집중도' 를 먼저 재고, 그걸로 정렬한다.
        # 게이트 대비가 크다고 주소를 나르는 것은 아니므로 대비 랭킹은 쓰지 않는다.
        allR = []
        for j in range(mcfg.p):
            gaps = torch.cat(
                [Phi[bidx, l + i, j] - Phi[bidx, data_pos[:, i], j] for i in range(n_mem)]
            )
            allR.append(circ_stats(gaps)[1])
        allR = torch.stack(allR)
        top = allR.argsort(descending=True)[:8]
        print(f"[채널 스캔] 집중도 R > 0.5 인 채널 수 = {int((allR > 0.5).sum())} / {mcfg.p}")
        print(f"            R 상위 5 = {[round(v, 3) for v in allR[top[:5]].tolist()]}")
        print(f"            게이트 대비 상위 5 = {contrast.argsort(descending=True)[:5].tolist()}")
        print(f"            집중도   상위 5 = {top[:5].tolist()}\n")

        print("(A)(B) 위상차 ΔΦ = Φ[marker i] - Φ[data j] 의 배치 내 집중도")
        print("      집중도 R: 1 이면 모든 샘플에서 위상차가 동일 (= 주소로 쓸 수 있음)")
        print(f"\n  {'채널':>5} {'정답 j=i':>22} {'오답 j≠i':>22} {'판별력':>18}")
        print(
            f"  {'':>5} {'집중도 R':>11}{'평균방향':>11} {'집중도 R':>11}{'평균방향':>11}"
            f" {'각분리':>9}{'점수':>9}"
        )
        for j in top.tolist():
            corr, wrong = [], []
            for i in range(n_mem):
                corr.append(Phi[bidx, l + i, j] - Phi[bidx, data_pos[:, i], j])
                for jj in range(n_mem):
                    if jj != i:
                        wrong.append(Phi[bidx, l + i, j] - Phi[bidx, data_pos[:, jj], j])
            cm, cR = circ_stats(torch.cat(corr))
            wm, wR = circ_stats(torch.cat(wrong))
            # 각분리: 정답/오답 평균방향의 원형 거리 (최대 π).
            # 점수: 정답이 뭉쳐 있으면서 오답과 떨어져 있어야 주소로 쓸 수 있다.
            sep = torch.abs(torch.atan2(torch.sin(cm - wm), torch.cos(cm - wm)))
            score = cR * (1 - wR * torch.cos(cm - wm))
            flag = "  ← 주소 채널" if score > 0.8 else ""
            print(
                f"  {j:>5} {cR:>11.4f}{cm:>11.4f} {wR:>11.4f}{wm:>11.4f}"
                f" {sep:>9.4f}{score:>9.4f}{flag}"
            )
        print("  각분리 ≈ π 면 오답이 정답의 반대 위상으로 밀려나 커널에서 부호가 뒤집힌다.")

        # 채널 전체를 합친 커널값 비교
        print("\n(D) 커널 기여  Σ_j |z_t||z_n| cos(argz_n - argz_t - ψ_j - ΔΦ_j) 의 정답/오답 대비")
        z, q, k = blk.project(h0)
        rot = torch.polar(torch.ones_like(Phi), -Phi)
        qs, ks = q * rot, k * rot
        kern_ok, kern_no = [], []
        for i in range(n_mem):
            qt = qs[bidx, l + i]  # (B,p)
            for jj in range(n_mem):
                kn = ks[bidx, data_pos[:, jj]]
                val = (qt.conj() * kn).real.sum(-1) / math.sqrt(mcfg.p)
                (kern_ok if jj == i else kern_no).append(val)
        ok = torch.cat(kern_ok)
        no = torch.cat(kern_no)
        print(f"  정답 데이터: 평균 {ok.mean():+.4f}   표준편차 {ok.std():.4f}")
        print(f"  오답 데이터: 평균 {no.mean():+.4f}   표준편차 {no.std():.4f}")
        d = (ok.mean() - no.mean()).abs() / torch.sqrt(0.5 * (ok.var() + no.var()))
        print(f"  분리도 (Cohen's d) = {d:.3f}   ← 1 이상이면 뚜렷한 분리")

        # (C) 어텐션 국소화
        print("\n(C) 합성 연산자에서의 주소지정 (마커 i 가 데이터 4개에 배분한 질량)")
        R = mcfg.R
        M = torch.eye(T, device=dev).expand(B, T, T).clone()
        for r in range(R):
            M = (torch.eye(T, device=dev) + tr[r]["A"] / R) @ M
        for i in range(n_mem):
            row = M[:, l + i, :l].abs()
            row = row / row.sum(-1, keepdim=True).clamp_min(1e-9)
            per = torch.stack([row[bidx, data_pos[:, jj]] for jj in range(n_mem)], -1)
            share = per / per.sum(-1, keepdim=True).clamp_min(1e-9)
            best = share.mean(0).argmax().item()
            mark = "OK" if best == i else "  "
            print(
                f"  marker {i}: "
                + "  ".join(f"D{jj}:{share[:, jj].mean():.3f}" for jj in range(n_mem))
                + f"   최대=D{best} {mark}"
            )


if __name__ == "__main__":
    main()
