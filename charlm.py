"""바이트 단위 언어모델링 — Model 0 vs Model 1 vs softmax attention.

selective copy 는 순서 위상 주소지정이 정확히 최적해인 태스크였다 (§4 에서 닫힌 형태로
해를 유도할 수 있었던 것이 그 증거다). 따라서 거기서 이겼다는 것은 "메커니즘이 실재한다"
의 증거이지 "이 아키텍처가 쓸모 있다"의 증거가 아니다.

이 스크립트는 **유도된 해가 없는** 태스크에서 재본다. 코퍼스는 Python 표준 라이브러리
소스(바이트 단위)로, 이 메커니즘을 위해 설계되지 않았지만 실제 장거리 구조가 있다
(괄호 짝, 들여쓰기 블록, 식별자 재사용, import 된 이름의 재등장).

세 모델 모두 동일 조건:
  - MLP 없음, 재귀 깊이 R 을 weight-tied 로 반복
  - tied embedding head, 다음 바이트 CE
  - 같은 d, 같은 스텝, 같은 옵티마이저
차이는 토큰 혼합 커널뿐이다.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch

from model0 import Model0, Model0Config
from model1 import Model1, Model1Config
from mamba3_ref import Mamba3Config, Mamba3Model
from train_selective_copy import AttnBaseline

CACHE = "/workspace/corpus_bytes.npy"


def build_corpus(limit_mb: float = 8.0) -> np.ndarray:
    if os.path.exists(CACHE):
        return np.load(CACHE)
    files = sorted(glob.glob("/usr/lib/python3.12/**/*.py", recursive=True))
    buf, tot = [], 0
    for f in files:
        try:
            b = open(f, "rb").read()
        except OSError:
            continue
        buf.append(b)
        tot += len(b)
        if tot > limit_mb * 1e6:
            break
    data = np.frombuffer(b"\n".join(buf), dtype=np.uint8).copy()
    np.save(CACHE, data)
    return data


def get_batch(data: np.ndarray, T: int, batch: int, dev: str, rng: np.random.Generator):
    ix = rng.integers(0, len(data) - T - 1, size=batch)
    x = np.stack([data[i : i + T] for i in ix])
    y = np.stack([data[i + 1 : i + T + 1] for i in ix])
    return (
        torch.from_numpy(x).long().to(dev),
        torch.from_numpy(y).long().to(dev),
    )


def build(args):
    if args.model == "mamba3":
        cfg = Mamba3Config(
            d_model=args.d, d_state=args.d_state, headdim=args.headdim,
            n_layer=args.n_layer, vocab_size=256, rope_fraction=args.rope_fraction,
        )
        m = Mamba3Model(cfg)
        m.num_parameters = lambda: {"total": sum(q.numel() for q in m.parameters())}
        return m, cfg
    kw = dict(d=args.d, p=args.p, R=args.R, vocab_size=256, w_scale=args.w_scale)
    if args.model == "model1":
        kw.update(tied=not args.untied, use_wv=args.use_wv)
    if args.model == "model1":
        cfg = Model1Config(**kw)
        return Model1(cfg), cfg
    cfg = Model0Config(**kw)
    if args.model == "model0":
        return Model0(cfg), cfg
    return AttnBaseline(cfg, args.T), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model1", choices=["model0", "model1", "attn", "mamba3"])
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--d-state", type=int, default=32)
    ap.add_argument("--headdim", type=int, default=32)
    ap.add_argument("--rope-fraction", type=float, default=0.5)
    ap.add_argument("--untied", action="store_true")
    ap.add_argument("--use-wv", action="store_true")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--p", type=int, default=64)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--w-scale", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # 분할은 블록 인터리브로 한다. 파일을 경로순 정렬해 뒤쪽을 잘라내면 val 이
    # 단일 파일(pydoc_data/topics.py — 코드가 아니라 산문)이 되어 분포가 어긋난다.
    # 실제로 그렇게 했을 때 고유 바이트가 121 vs 170, 비ASCII 비율이 5배 차이났다.
    data = build_corpus()
    blocks = np.array_split(data, 512)
    val = np.concatenate([b for i, b in enumerate(blocks) if i % 16 == 0])
    train = np.concatenate([b for i, b in enumerate(blocks) if i % 16 != 0])

    model, cfg = build(args)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Adam 의 첫 스텝은 그래디언트와 무관하게 ±lr 만큼 움직인다. W_C 초기 std 는
    # w_scale/sqrt(2d) 라 d 가 클수록 작으므로 상대 교란이 lr·sqrt(d) 로 커지고,
    # 3차 자기항이 이를 세제곱으로 증폭한다 → 폭 확장 시 첫 스텝에서 발산.
    # 선형 warmup 으로 해소. 공정성을 위해 모든 모델에 동일 스케줄 적용.
    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    rng = np.random.default_rng(args.seed)
    vrng = np.random.default_rng(1234)

    print(
        f"model={args.model}  R={args.R}  T={args.T}  d={args.d}  "
        f"params={model.num_parameters()}  corpus={len(data) / 1e6:.1f}MB"
    )
    print(f"{'step':>7} {'train':>8} {'val bpb':>9} {'max logit':>11} {'gnorm':>9} {'sec':>7}")

    hist, best = [], float("inf")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train, args.T, args.batch, dev, rng)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 256).float(), y.reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
        opt.step()
        sched.step()

        if step % args.log_every == 0 or step == 1:
            model.eval()
            tot, amax = 0.0, 0.0
            with torch.no_grad():
                for _ in range(20):
                    xv, yv = get_batch(val, args.T, args.batch, dev, vrng)
                    lv = model(xv)
                    tot += torch.nn.functional.cross_entropy(
                        lv.reshape(-1, 256).float(), yv.reshape(-1)
                    ).item()
                    amax = max(amax, lv.abs().max().item())
            model.train()
            bpb = tot / 20 / math.log(2)  # bits per byte
            best = min(best, bpb)
            hist.append({"step": step, "train": loss.item(), "bpb": bpb, "logit_max": amax})
            print(
                f"{step:>7} {loss.item():>8.4f} {bpb:>9.4f} {amax:>11.3g} "
                f"{gn:>9.3f} {time.time() - t0:>7.1f}"
            )
            if not math.isfinite(bpb) or bpb > 100:
                print("  발산 — 중단")
                break

    print(f"\n최저 val bits-per-byte = {best:.4f}   (균등 분포 = 8.0)")
    if args.out:
        json.dump({"args": vars(args), "hist": hist, "best_bpb": best}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
