"""학습된 Model 0 가 selective copy 를 *어떤 경로로* 푸는지 분해한다.

핵심 의문: 전이 Λ = diag(e^{-α+iθ}) 는 입력과 무관한 LTI 다. 그런데 데이터 토큰의
위치는 샘플마다 랜덤이므로, "i 번째 데이터 토큰"까지의 시간 지연 Δ 도 랜덤이다.
시간 지연만으로 주소지정이 불가능한데 어떻게 푸는가?

가설: 재귀 깊이 r 이 순서(ordinal) 주소를 만든다.
  r=0 에서 각 위치의 h 는 "그 앞에 데이터 토큰이 몇 개 있었는지"를 흡수한다.
  → r≥1 에서 z_n = W_C h_n 의 위상/진폭이 (내용, 순서)를 함께 인코딩한다.
  → 마커 i 의 q 가 그 순서 성분과 위상 정합해서 꺼낸다.
검증:
  (1) 마커 i 의 어텐션이 i 번째 데이터 위치에 실제로 몰리는가
  (2) 그 정합이 r 이 깊어질수록 선명해지는가
  (3) R=1 로 줄이면 무너지는가  (→ ablate_R.py)
"""

from __future__ import annotations

import argparse

import torch

from model0 import Model0, Model0Config
from selective_copy import DATA_OFFSET, TaskConfig, accuracy, make_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    task: TaskConfig = ck["task"]
    mcfg: Model0Config = ck["mcfg"]
    model = Model0(mcfg).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()

    torch.manual_seed(1234)
    x, y = make_batch(task, args.batch, dev)
    with torch.no_grad():
        logits, tr = model(x, return_trace=True)
    ta, sa = accuracy(logits, task, y)
    print(f"체크포인트 성능:  tok_acc={ta:.4f}  seq_acc={sa:.4f}\n")

    is_data = x[:, : task.l_noise] >= DATA_OFFSET  # (B, l_noise)
    # 각 샘플의 데이터 토큰 위치 (정렬됨)
    data_pos = is_data.float().argsort(dim=-1, descending=True, stable=True)[:, : task.n_mem]
    data_pos = data_pos.sort(dim=-1).values  # (B, n_mem)

    # ---------------------------------------------------------------- #
    print("(1)(2) 마커 i 의 어텐션이 i 번째 데이터 위치에 몰리는가 — 재귀 깊이별")
    print(f"  {'r':>3} {'질량@정답위치':>14} {'질량@데이터전체':>16} {'질량@blank':>12} {'argmax적중':>11}")
    B = x.shape[0]
    bidx = torch.arange(B, device=dev)
    for r, s in enumerate(tr):
        if "A" not in s:
            continue
        A = s["A"]  # (B,T,T)
        rows = []
        for i in range(task.n_mem):
            row = A[:, task.l_noise + i, : task.l_noise].abs()  # (B, l_noise)
            row = row / row.sum(-1, keepdim=True).clamp_min(1e-9)
            on_target = row[bidx, data_pos[:, i]]
            on_data = (row * is_data.float()).sum(-1)
            on_blank = (row * (~is_data).float()).sum(-1)
            hit = (row.argmax(-1) == data_pos[:, i]).float()
            rows.append(torch.stack([on_target, on_data, on_blank, hit]))
        m = torch.stack(rows).mean(dim=(0, 2))
        print(f"  {r:>3} {m[0]:>14.4f} {m[1]:>16.4f} {m[2]:>12.4f} {m[3]:>11.4f}")
    unif = 1.0 / task.l_noise
    print(f"  (균등 기준선: 정답위치 {unif:.4f}, 데이터전체 {task.n_mem / task.l_noise:.4f})")

    # ---------------------------------------------------------------- #
    print("\n(3) 마커별 어텐션 피크가 순서대로 이동하는가")
    A_last = tr[-2]["A"]
    print(f"  {'marker i':>9} {'argmax 위치(평균)':>18} {'정답 위치(평균)':>16} {'상관':>8}")
    for i in range(task.n_mem):
        row = A_last[:, task.l_noise + i, : task.l_noise].abs()
        am = row.argmax(-1).float()
        tgt = data_pos[:, i].float()
        corr = torch.corrcoef(torch.stack([am, tgt]))[0, 1].item()
        print(f"  {i:>9} {am.mean():>18.2f} {tgt.mean():>16.2f} {corr:>8.4f}")

    # ---------------------------------------------------------------- #
    print("\n(4) blank 위치의 h 가 '앞에 나온 데이터 개수'를 인코딩하는가 (순서 주소의 근거)")
    #  각 위치까지의 누적 데이터 개수
    cum = is_data.float().cumsum(-1)  # (B, l_noise)
    for r, s in enumerate(tr):
        h = s["h"][:, : task.l_noise]  # (B, l_noise, d)
        # blank 위치만 골라 h 를 선형 프로브 (closed form ridge)
        mask = (~is_data).reshape(-1)
        H = h.reshape(-1, h.shape[-1])[mask]
        c = cum.reshape(-1)[mask]
        if H.std(0).max() < 1e-8:
            print(f"  r={r}:  모든 blank 의 h 가 동일 — 순서 정보 없음 (임베딩 그대로)")
            continue
        H = torch.cat([H, torch.ones_like(H[:, :1])], -1)
        w = torch.linalg.lstsq(H.double(), c.double().unsqueeze(-1)).solution
        pred = (H.double() @ w).squeeze(-1)
        r2 = 1 - (pred - c.double()).pow(2).sum() / (c.double() - c.double().mean()).pow(2).sum()
        print(f"  r={r}:  blank 위치 h 에서 누적 데이터 개수 선형복원 R² = {r2.item():.4f}")

    composed_analysis(model, tr, task, x, data_pos, is_data)


def composed_analysis(model, tr, task, x, data_pos, is_data):
    """(5) 합성 연산자로 다시 측정.

    h^(r+1) = h^(r) + A^(r) h^(r) / R  이므로 입력 토큰장에서 출력까지의 실제
    토큰 혼합은 개별 A^(r) 이 아니라  M = Π_r (I + A^(r)/R)  이다.
    단일 A^(r) 만 보면 residual 로 흘러가는 항등 경로를 통째로 무시하게 된다.
    """
    R = model.cfg.R
    A0 = tr[0]["A"]
    B, T, _ = A0.shape
    dev = A0.device
    M = torch.eye(T, device=dev).expand(B, T, T).clone()
    for r in range(R):
        M = (torch.eye(T, device=dev) + tr[r]["A"] / R) @ M

    bidx = torch.arange(B, device=dev)
    l = task.l_noise
    print("\n(5) 합성 연산자 M = Π_r (I + A^(r)/R) 로 본 주소지정")
    print(
        f"  {'marker i':>9} {'질량@정답':>10} {'질량@타데이터':>13} {'질량@blank':>11} "
        f"{'질량@마커':>10} {'argmax적중':>11} {'상관':>8}"
    )
    tot_hit = []
    for i in range(task.n_mem):
        row = M[:, l + i, :].abs()
        row = row / row.sum(-1, keepdim=True).clamp_min(1e-9)
        noise = row[:, :l]
        on_target = noise[bidx, data_pos[:, i]]
        on_data_all = (noise * is_data.float()).sum(-1)
        on_blank = (noise * (~is_data).float()).sum(-1)
        on_marker = row[:, l:].sum(-1)
        hit = (noise.argmax(-1) == data_pos[:, i]).float()
        am = noise.argmax(-1).float()
        corr = torch.corrcoef(torch.stack([am, data_pos[:, i].float()]))[0, 1]
        tot_hit.append(hit.mean().item())
        print(
            f"  {i:>9} {on_target.mean():>10.4f} {(on_data_all - on_target).mean():>13.4f} "
            f"{on_blank.mean():>11.4f} {on_marker.mean():>10.4f} {hit.mean():>11.4f} "
            f"{corr:>8.4f}"
        )
    print(f"  (균등 기준선: 정답 {1 / T:.4f})   평균 argmax 적중 = {sum(tot_hit) / len(tot_hit):.4f}")

    print("\n(6) 정답 데이터 위치 vs 같은 샘플의 다른 데이터 위치 — 질량비")
    for i in range(task.n_mem):
        row = M[:, l + i, :l].abs()
        row = row / row.sum(-1, keepdim=True).clamp_min(1e-9)
        per = torch.stack([row[bidx, data_pos[:, j]] for j in range(task.n_mem)], -1)
        share = per / per.sum(-1, keepdim=True).clamp_min(1e-9)
        print(
            f"  marker {i}: 데이터 4개에 대한 질량 배분 = "
            + "  ".join(f"D{j}:{share[:, j].mean():.3f}" for j in range(task.n_mem))
            + f"   ← D{i} 가 최대여야 함"
        )


if __name__ == "__main__":
    main()
