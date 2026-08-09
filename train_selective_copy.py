"""Model 0 를 selective copy 로 학습한다.

실행 예:
    python train_selective_copy.py --model model0 --task select
    python train_selective_copy.py --model model0 --task fixed     # LTI 통제군
    python train_selective_copy.py --model attn   --task select    # 태스크 난이도 기준선

기준선(attn)은 MLP 없는 단일헤드 causal softmax attention 이다. Model 0 와 같은
"MLP 없음 + 재귀 weight-tied" 조건을 유지하고 혼합 커널만 바꾼다. 따라서 두 모델의
차이는 오직 [signed 복소 위상 커널 + 유니터리성 잃은 감쇠 기억] 대 [softmax + 위치임베딩]
이다.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn as nn

from model0 import Model0, Model0Config, dirichlet_energy, euler_gain
from model1 import Model1, Model1Config
from selective_copy import DATA_OFFSET, BLANK, MARKER, TaskConfig, accuracy, loss_fn, make_batch


class AttnBaseline(nn.Module):
    """MLP 없는 단일헤드 causal softmax attention, weight-tied 로 R 번 반복."""

    def __init__(self, cfg: Model0Config, seq_len: int):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.embed = nn.Embedding(cfg.vocab_size, d)
        nn.init.normal_(self.embed.weight, std=1.0)
        self.pos = nn.Parameter(torch.randn(seq_len, d) * 0.02)
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / d))

    def forward(self, tokens):
        h = self.embed(tokens) + self.pos[: tokens.shape[1]]
        T = tokens.shape[1]
        mask = torch.ones(T, T, device=tokens.device, dtype=torch.bool).tril()
        for _ in range(self.cfg.R):
            q, k, v = self.Wq(h), self.Wk(h), self.Wv(h)
            s = (q @ k.transpose(1, 2)) / math.sqrt(self.cfg.d)
            s = s.masked_fill(~mask, float("-inf"))
            h = h + s.softmax(-1) @ v / self.cfg.R
        return self.logit_scale * (h @ self.embed.weight.t())

    def num_parameters(self):
        tot = sum(p.numel() for p in self.parameters())
        return {"total": tot}


def build(args, task: TaskConfig):
    kw = dict(
        d=args.d,
        p=args.p,
        R=args.R,
        vocab_size=task.vocab_size,
        w_scale=args.w_scale,
        r_min=args.r_min,
        r_max=args.r_max,
    )
    if args.model == "model1":
        mcfg = Model1Config(**kw, phase_gate=args.phase_gate, gate_bias_init=args.gate_bias,
                            polar=args.polar, use_wv=args.use_wv)
        return Model1(mcfg), mcfg
    mcfg = Model0Config(**kw)
    if args.model == "model0":
        return Model0(mcfg), mcfg
    return AttnBaseline(mcfg, task.seq_len), mcfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model0", choices=["model0", "model1", "attn"])
    ap.add_argument("--phase-gate", default="sigmoid", choices=["sigmoid", "softplus"])
    ap.add_argument("--gate-bias", type=float, default=-2.0)
    ap.add_argument("--polar", action="store_true")
    ap.add_argument("--use-wv", action="store_true")
    ap.add_argument("--task", default="select", choices=["select", "fixed"])
    ap.add_argument("--n-mem", type=int, default=4)
    ap.add_argument("--n-mem-min", type=int, default=None, help="설정 시 N 이 샘플마다 가변")
    ap.add_argument("--l-noise", type=int, default=32)
    ap.add_argument("--n-data", type=int, default=16)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--p", type=int, default=64)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--w-scale", type=float, default=0.5)
    ap.add_argument("--r-min", type=float, default=0.90)
    ap.add_argument("--r-max", type=float, default=0.999)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None, help="체크포인트 경로 (분석용)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    task = TaskConfig(
        n_data=args.n_data,
        n_mem=args.n_mem,
        l_noise=args.l_noise,
        mode=args.task,
        n_mem_min=args.n_mem_min,
    )
    model, mcfg = build(args, task)
    model = model.to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    print(
        f"model={args.model}  task={args.task}  T={task.seq_len}  V={task.vocab_size}  "
        f"n_mem={task.n_mem}  params={model.num_parameters()}"
    )
    print(f"{'step':>7} {'loss':>8} {'tok_acc':>8} {'seq_acc':>8} {'gnorm':>9} {'sec':>6}")

    chance = 1.0 / task.n_data
    hist = []
    t0 = time.time()
    best = 0.0
    for step in range(1, args.steps + 1):
        x, y = make_batch(task, args.batch, dev)
        logits = model(x)
        loss = loss_fn(logits, task, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
        opt.step()
        sched.step()

        if step % args.log_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                xe, ye = make_batch(task, 512, dev)
                le = model(xe)
                el = loss_fn(le, task, ye).item()
                ta, sa = accuracy(le, task, ye)
            model.train()
            best = max(best, sa)
            hist.append({"step": step, "loss": el, "tok_acc": ta, "seq_acc": sa})
            print(
                f"{step:>7} {el:>8.4f} {ta:>8.4f} {sa:>8.4f} {gnorm:>9.3f} "
                f"{time.time() - t0:>6.1f}"
            )
            if not math.isfinite(el):
                print("  발산 — 중단")
                break

    print(f"\nchance tok_acc = {chance:.4f},  최종 tok_acc = {ta:.4f},  최고 seq_acc = {best:.4f}")

    # Model 1: 학습된 위상 게이트가 §4 유도대로 되었는가
    if args.model == "model1":
        model.eval()
        blk = model.block
        with torch.no_grad():
            xe, ye = make_batch(task, 256, dev)
            _, tr = model(xe, return_trace=True)
            h0 = tr[0]["h"]
            g = blk.gate(h0)  # (B,T,p) 게이트 개방도
            dth = blk.s_raw * g  # (B,T,p) Δθ
            is_blank = xe == BLANK
            is_mark = xe == MARKER
            is_data = xe >= DATA_OFFSET

            # 채널별로 |Δθ(data) - Δθ(blank)| 가 큰 순으로 정렬 = '세는 채널'
            def sel(m):
                return dth[m].mean(0)  # (p,)

            db, dd, dm = sel(is_blank), sel(is_data), sel(is_mark)
            contrast = (dd - db).abs()
            order = contrast.argsort(descending=True)
            print("\n[검증 ③] 학습된 위상 증분 Δθ  — 상위 8개 '세는 채널'")
            print(
                f"  {'ch':>4} {'Δθ blank':>10} {'Δθ data':>10} {'Δθ marker':>11} "
                f"{'s_j':>8} {'θ_j':>8} {'|d-m|':>8}"
            )
            for j in order[:8].tolist():
                print(
                    f"  {j:>4} {db[j]:>10.4f} {dd[j]:>10.4f} {dm[j]:>11.4f} "
                    f"{blk.s_raw[j]:>8.4f} {blk.theta[j]:>8.4f} "
                    f"{(dd[j] - dm[j]).abs():>8.4f}"
                )
            print(
                f"\n  예측: Δθ(blank)≈0,  Δθ(data)≈Δθ(marker)=c,  θ_j≈0 (순서 채널은 절대시간 무시)"
            )
            print(
                f"  전체 평균:  blank={db.mean():.4f}  data={dd.mean():.4f}  "
                f"marker={dm.mean():.4f}   |data-marker|={(dd - dm).abs().mean():.4f}"
            )
            top = order[:8]
            print(
                f"  상위 8채널: blank={db[top].mean():.4f}  data={dd[top].mean():.4f}  "
                f"marker={dm[top].mean():.4f}   |data-marker|={(dd[top] - dm[top]).abs().mean():.4f}"
                f"   |θ|={blk.theta[top].abs().mean():.4f}"
            )
            print(f"\n{'r':>3} {'E_D':>12} {'ρ_R':>9} {'mean d_t':>10} {'max|a_tt|':>10}")
            for r, s in enumerate(tr):
                if "A" in s:
                    att = torch.diagonal(s["A"], dim1=-2, dim2=-1).abs().max().item()
                    print(
                        f"{r:>3} {s['E_D'].mean():>12.2f} {s['rho_R'].mean():>9.4f} "
                        f"{s['d_t'].mean():>10.4f} {att:>10.4f}"
                    )
                else:
                    print(f"{r:>3} {s['E_D'].mean():>12.2f}")

    # Model 0 의 내부 동역학 진단
    if args.model == "model0":
        model.eval()
        with torch.no_grad():
            xe, ye = make_batch(task, 128, dev)
            _, tr = model(xe, return_trace=True)
            print(f"\n{'r':>3} {'E_D':>12} {'ρ_R':>9} {'mean d_t':>10} {'max|a_tt|':>10}")
            for r, s in enumerate(tr):
                if "A" in s:
                    att = torch.diagonal(s["A"], dim1=-2, dim2=-1).abs().max().item()
                    print(
                        f"{r:>3} {s['E_D'].mean():>12.2f} {s['rho_R'].mean():>9.4f} "
                        f"{s['d_t'].mean():>10.4f} {att:>10.4f}"
                    )
                else:
                    print(f"{r:>3} {s['E_D'].mean():>12.2f}")
            tau = model.block.tau
            print(
                f"\n학습된 τ=1/α:  min={tau.min():.1f}  median={tau.median():.1f}  "
                f"max={tau.max():.1f}   (T={task.seq_len})"
            )
            print(
                f"학습된 |θ|:    min={model.block.theta.abs().min():.3f}  "
                f"median={model.block.theta.abs().median():.3f}  "
                f"max={model.block.theta.abs().max():.3f}"
            )
            cp = torch.cos(model.block.psi)
            print(
                f"cos ψ:         흥분성(>0) {int((cp > 0).sum())}/{len(cp)}  "
                f"mean={cp.mean():.3f}"
            )
            # blank 토큰이 상태에 쓰이지 않도록 학습됐는가 (content-aware 쓰기)
            W = torch.complex(model.block.W_re, model.block.W_im)
            zt = (model.embed.weight.to(W.dtype) @ W.t()).abs()  # (V,p)
            nrm = zt.norm(dim=-1)
            print(
                f"\n‖z‖ (쓰기 세기):  blank={nrm[0]:.3f}  marker={nrm[1]:.3f}  "
                f"data(mean)={nrm[2:].mean():.3f}   ← blank 억제 = content-aware 쓰기"
            )

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"args": vars(args), "hist": hist}, fh, indent=2)
    if args.save:
        torch.save(
            {"state": model.state_dict(), "task": task, "mcfg": mcfg, "args": vars(args)},
            args.save,
        )


if __name__ == "__main__":
    main()
