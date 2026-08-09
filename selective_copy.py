"""Selective Copy 토이 태스크.

표준 정의 (Jing et al. 2019 / Mamba 논문 Table 1):

    입력:  [blank, blank, D3, blank, D7, blank, blank, D1, ...,  M, M, M]
                          ^^      ^^            ^^         마커 n_mem 개
    목표:  i 번째 마커 위치에서 i 번째 데이터 토큰을 출력한다.

이 태스크가 중요한 이유는 두 능력을 분리해서 요구하기 때문이다.

  1. content-aware 쓰기 : blank 는 상태에 기록하지 않고 데이터 토큰만 기록해야 한다.
  2. content-aware 읽기 : "i 번째로 등장한 데이터 토큰"을 꺼내야 하는데,
                          그 절대 시간 지연 Δ 는 매 샘플마다 랜덤이다.

Model 0 의 전이 Λ = diag(e^{-α+iθ}) 는 입력과 무관한 LTI 다. 따라서 2번을
시간 지연만으로는 풀 수 없다. 풀 수 있다면 그 경로는 반드시
a_tn = p^{-1/2} Σ_j γ_j|z_tj||z_nj| e^{-α_jΔ} cos(arg z_nj - arg z_tj - ψ_j - Δθ_j)
의 *내용 의존 부분*(진폭 |z| 와 의미 위상 arg z)에서 나와야 한다.

  fixed  모드: 데이터 토큰 위치를 고정 → Δ 가 결정적. LTI 로 풀 수 있는 통제군.
  select 모드: 데이터 토큰 위치가 랜덤 → 진짜 selective copy.

두 모드의 격차가 곧 "이 구조의 선택성이 어디서 나오는가"에 대한 답이 된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

BLANK = 0
MARKER = 1
DATA_OFFSET = 2


@dataclass
class TaskConfig:
    n_data: int = 16  # 서로 다른 데이터 토큰 종류
    n_mem: int = 4  # 기억해야 할 개수 (n_mem_min 이 있으면 그 상한)
    l_noise: int = 32  # 데이터가 흩뿌려지는 구간 길이
    mode: str = "select"  # "select" (랜덤 위치) | "fixed" (고정 위치)

    # 시험 B: 샘플마다 N 이 달라진다. §4 유도의 공진 오프셋은 c·N 이므로
    # R=1 의 고정 φ_q 로는 단 하나의 N 에만 맞출 수 있다 → 깨져야 한다는 예측.
    n_mem_min: int | None = None

    @property
    def vocab_size(self) -> int:
        return DATA_OFFSET + self.n_data

    @property
    def seq_len(self) -> int:
        return self.l_noise + self.n_mem

    @property
    def answer_slice(self) -> slice:
        return slice(self.l_noise, self.l_noise + self.n_mem)


def make_batch(cfg: TaskConfig, batch: int, device="cpu", generator=None) -> tuple[Tensor, Tensor]:
    """→ (x: (B,T) long,  y: (B, n_mem) long)

    y[b,i] 는 x 의 위치 l_noise+i (i 번째 마커) 에서 예측해야 할 토큰이다.
    """
    # n_mem 개를 l_noise 칸에 겹치지 않게 놓아야 한다. 넘으면 argsort 가 조용히
    # 잘린 위치를 돌려주어 '풀 수 없는' 과제가 만들어진다 (실제로 한 번 겪었다).
    assert cfg.n_mem <= cfg.l_noise, (
        f"n_mem({cfg.n_mem}) > l_noise({cfg.l_noise}): 놓을 자리가 없다")
    g = generator
    x = torch.full((batch, cfg.l_noise), BLANK, dtype=torch.long, device=device)

    if cfg.mode == "fixed":
        # 균등 간격 고정 위치 — Δ 가 샘플마다 동일한 통제군
        pos = torch.linspace(0, cfg.l_noise - 1, cfg.n_mem).long().to(device)
        pos = pos.unsqueeze(0).expand(batch, -1)
    elif cfg.mode == "select":
        # 랜덤 위치 (정렬) — 매 샘플 Δ 가 달라진다
        noise = torch.rand(batch, cfg.l_noise, device=device, generator=g)
        pos = noise.argsort(dim=-1)[:, : cfg.n_mem].sort(dim=-1).values
    else:
        raise ValueError(cfg.mode)

    vals = torch.randint(
        DATA_OFFSET, DATA_OFFSET + cfg.n_data, (batch, cfg.n_mem), device=device, generator=g
    )
    markers = torch.full((batch, cfg.n_mem), MARKER, dtype=torch.long, device=device)

    if cfg.n_mem_min is not None:
        # 샘플마다 N ∈ [n_mem_min, n_mem]. 남는 슬롯은 데이터도 마커도 아닌 blank 로 두고,
        # 목표를 -100 으로 채워 손실·정확도에서 제외한다 (총 길이는 고정).
        N = torch.randint(
            cfg.n_mem_min, cfg.n_mem + 1, (batch, 1), device=device, generator=g
        )
        keep = torch.arange(cfg.n_mem, device=device).unsqueeze(0) < N  # (B,n_mem)
        vals = torch.where(keep, vals, torch.full_like(vals, BLANK))
        markers = torch.where(keep, markers, torch.full_like(markers, BLANK))
        x.scatter_(1, pos, vals)
        y = torch.where(keep, vals, torch.full_like(vals, -100))
        return torch.cat([x, markers], dim=1), y

    x.scatter_(1, pos, vals)
    return torch.cat([x, markers], dim=1), vals


def accuracy(logits: Tensor, cfg: TaskConfig, y: Tensor) -> tuple[float, float]:
    """(토큰 정확도, 시퀀스 완전 정확도). y == -100 인 슬롯은 제외한다."""
    pred = logits[:, cfg.answer_slice].argmax(-1)
    valid = y != -100
    hit = (pred == y) & valid
    tok = hit.sum().item() / valid.sum().clamp_min(1).item()
    seq = (hit.sum(-1) == valid.sum(-1)).float().mean().item()
    return tok, seq


def loss_fn(logits: Tensor, cfg: TaskConfig, y: Tensor) -> Tensor:
    lg = logits[:, cfg.answer_slice]  # (B, n_mem, V)
    return torch.nn.functional.cross_entropy(
        lg.reshape(-1, lg.shape[-1]), y.reshape(-1), ignore_index=-100
    )
