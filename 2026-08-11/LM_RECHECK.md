# LM 재실험 — C / D / E

2026-08-11, Python stdlib 8MB 바이트 LM, `T=128`, seed 0, 6,000스텝 기준.

## 결과 요약

| 실험 | 설정 | 결과 |
|---|---|---:|
| A | 기준선 | 2.9280 bpb |
| B | `psi_inhib + self_exp` | 2.9249 bpb |
| C | `psi_inhib + W_V` | **2.0065 bpb** |
| D | Mamba-3 L=1, 수치 버그 수정 | 사용자 요청으로 step 500에서 중단: 2.1538 bpb |
| E | C + `ordinal_only` | step 4500에서 **2.0202 bpb** — C와 사실상 동률로 수렴 중 |

## C — 올바른 `W_V` 조합

원래 시도한 `self_exp + W_V`는 구현 오류가 아니라 수학적으로 정의되지 않은 조합이다.
`self_exp`는 자기항이 `a_tt h_t`일 때의 스칼라 지수 적분인데, `W_V`를 넣으면 자기항이
`a_tt(h_t W_V)`인 행렬 동역학이 된다. 따라서 `self_exp`를 빼고 실행했다.

```bash
python charlm.py --model model1 --d 256 --p 128 --steps 6000 \
  --lr 1.414e-3 --warmup 300 --psi-inhib --use-wv
```

주요 val bpb: step 1000 `2.5273` → 3000 `2.1654` → 4500 `2.0066` →
6000 **`2.0065`**. `W_V`가 원본 value의 span 제약을 푸는 효과가 재현됐다.

## D — Mamba-3 NaN 원인과 수정

이차 dual form에서 미래 위치까지 누적 감쇠의 지수를 먼저 계산한 뒤 causal mask를
곱하고 있었다. 학습 중 미래 영역의 지수가 overflow하면서 `inf * 0 = nan`이 됐다.
causal 하삼각 감쇠만 안정적인 재귀 곱으로 구성하도록 수정했다.

수정 검증:

- 극단 감쇠에서도 finite, 상삼각은 정확히 0
- quadratic 경로와 recurrent 경로 상대오차 `1.4e-7`
- 기존에는 step 157에서 NaN, 수정 후 step 500에서 val bpb `2.1538`

본 실행은 사용자 요청으로 step 500에서 중단했다.

## E — `ordinal_only` LM

```bash
python charlm.py --model model1 --d 256 --p 128 --steps 6000 \
  --lr 1.414e-3 --warmup 300 --psi-inhib --use-wv --ordinal-only
```

`ordinal_only`는 모든 위치 정보를 없애는 설정이 아니다. 정적·주기적 위치 위상 `theta`만
0으로 고정하며, causal 순서와 실제 거리 감쇠 `exp(-alpha * delta)`는 남는다. 위상은
내용 게이트가 센 사건의 순번을 나타낸다.

초기에는 정적 위상 분산이 사라져 step 1 max logit이 `1.32e4`로 크게 튀었지만 warmup과
clipping으로 회복했다. val bpb는 step 500 `3.2041`, 1500 `2.4044`, 2500 `2.2840`,
3500 `2.1737`, 4500 **`2.0202`**였다. 같은 step의 C `2.0066`과 차이는 `0.0136`으로,
LM 적합도는 사실상 동률이다.

이 결과만으로 외삽 우위를 확정할 수는 없지만, `ordinal_only`가 LM 학습을 막는다는
우려는 반증됐다. C와 E의 차이는 일반 LM 적합도가 아니라 길이 외삽·기억 수명 시험에서
가려야 한다.
