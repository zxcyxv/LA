"""Mamba-3 스타일 레이어를 selective copy 로 학습하고, 우리 위상 분석을 그대로 건다.

묻는 것: 데이터 의존 회전 SSM 이 §4 에서 유도한 것과 같은 '순서 위상 코드'를
        실제로 학습하는가?

  같은 코드가 나온다  → 우리 유도가 SOTA 계열 모델의 작동 원리를 설명한 것
  안 나온다           → SwiGLU/BCNorm 이 그 일을 대신한다는 뜻이고,
                        "MLP 없이 위상만으로 된다"는 우리 결과가 더 강해진다
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from mamba3_ref import Mamba3Config, Mamba3Model
from selective_copy import DATA_OFFSET, TaskConfig, accuracy, loss_fn, make_batch


def circ_stats(x, dim=0):
    c, s = torch.cos(x).mean(dim), torch.sin(x).mean(dim)
    return torch.atan2(s, c), torch.sqrt(c**2 + s**2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n-mem", type=int, default=4)
    ap.add_argument("--l-noise", type=int, default=32)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    task = TaskConfig(n_mem=args.n_mem, l_noise=args.l_noise)
    cfg = Mamba3Config(vocab_size=task.vocab_size, n_layer=args.n_layer)
    model = Mamba3Model(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    print(
        f"Mamba-3(ref)  n_layer={args.n_layer}  T={task.seq_len}  "
        f"params={model.num_parameters()}  각도채널={model.layers[0][1].n_ang}"
    )
    print(f"{'step':>7} {'loss':>8} {'tok':>7} {'seq':>7} {'sec':>7}")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = make_batch(task, args.batch, dev)
        loss = loss_fn(model(x), task, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                xe, ye = make_batch(task, 512, dev)
                le = model(xe)
                ta, sa = accuracy(le, task, ye)
                el = loss_fn(le, task, ye).item()
            model.train()
            print(f"{step:>7} {el:>8.4f} {ta:>7.4f} {sa:>7.4f} {time.time() - t0:>7.1f}")

    # ---------------- 위상 코드 측정 ----------------
    model.eval()
    torch.manual_seed(4321)
    with torch.no_grad():
        x, y = make_batch(task, 512, dev)
        logits, phases = model(x, return_phase=True)
    ta, sa = accuracy(logits, task, y)
    print(f"\n최종  tok={ta:.4f}  seq={sa:.4f}")

    B, T = x.shape
    l, n_mem = task.l_noise, task.n_mem
    bidx = torch.arange(B, device=dev)
    is_data = x[:, :l] >= DATA_OFFSET
    dpos = is_data.float().argsort(dim=-1, descending=True, stable=True)[:, :n_mem]
    dpos = dpos.sort(dim=-1).values

    print("\n레이어·각도채널별  ΔΦ = Φ[marker i] − Φ[data j] 통계")
    print(f"  {'layer':>5} {'ch':>4} {'정답 R':>9} {'정답 방향':>10} {'오답 R':>9} "
          f"{'각분리':>8} {'점수':>8}")
    best = []
    for li, phi in enumerate(phases):
        n_ang = phi.shape[-1]
        for j in range(n_ang):
            corr, wrong = [], []
            for i in range(n_mem):
                corr.append(phi[bidx, l + i, j] - phi[bidx, dpos[:, i], j])
                for jj in range(n_mem):
                    if jj != i:
                        wrong.append(phi[bidx, l + i, j] - phi[bidx, dpos[:, jj], j])
            cm, cR = circ_stats(torch.cat(corr))
            wm, wR = circ_stats(torch.cat(wrong))
            sep = torch.abs(torch.atan2(torch.sin(cm - wm), torch.cos(cm - wm)))
            score = cR * (1 - wR * torch.cos(cm - wm))
            best.append((score.item(), li, j, cR.item(), cm.item(), wR.item(), sep.item()))
    best.sort(reverse=True)
    for sc, li, j, cR, cm, wR, sep in best[:10]:
        flag = "  ← 주소 채널" if sc > 0.8 else ""
        print(f"  {li:>5} {j:>4} {cR:>9.4f} {cm:>10.4f} {wR:>9.4f} {sep:>8.4f} {sc:>8.4f}{flag}")
    if args.save:
        torch.save({"state": model.state_dict(), "cfg": cfg, "task": task}, args.save)

    # 증분 프로파일: 토큰 종류별 위상 증분이 §4 유도해와 맞는가
    # (유도해: blank 은 위상을 전진시키지 않고, data 와 marker 는 같은 c 만큼 전진)
    from selective_copy import BLANK, MARKER

    print("\n토큰 종류별 위상 증분  inc = tanh(angle)·DT·π   (§4 유도해: blank≈0, data≈marker=c)")
    with torch.no_grad():
        h = model.embed(x)
        n1, mix, _, _ = model.layers[0]
        u = n1(h)
        parts = mix.in_proj(u).split(
            [mix.d_inner, mix.d_inner, cfg.d_state, cfg.d_state,
             mix.nheads, mix.nheads, mix.nheads, mix.n_ang], dim=-1)
        _, _, _, _, dd_dt, _, _, ang = parts
        DT = torch.nn.functional.softplus(dd_dt + mix.dt_bias)
        inc = torch.tanh(ang) * DT.mean(-1, keepdim=True) * math.pi  # (B,T,n_ang)
        kinds = {"blank": x == BLANK, "data": x >= DATA_OFFSET, "marker": x == MARKER}
        print(f"  {'ch':>4} {'inc blank':>11} {'inc data':>11} {'inc marker':>12} "
              f"{'|blank| mod 2π':>15} {'|data-marker|':>14}")
        for j in range(mix.n_ang):
            v = {k: inc[m][:, j].mean().item() for k, m in kinds.items()}
            wrapped = abs((v["blank"] + math.pi) % (2 * math.pi) - math.pi)
            print(f"  {j:>4} {v['blank']:>11.4f} {v['data']:>11.4f} {v['marker']:>12.4f} "
                  f"{wrapped:>15.4f} {abs(v['data'] - v['marker']):>14.4f}")

    n_addr = sum(1 for b in best if b[0] > 0.8)
    print(f"\n  주소 채널 수 (점수>0.8): {n_addr} / {len(best)}")
    print(f"  최고 점수 {best[0][0]:.4f}   (우리 Model 1: R=1 → 1.234, R=4 → 1.233)")


if __name__ == "__main__":
    main()
