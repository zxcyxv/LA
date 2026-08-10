# LA — 복소 선형어텐션의 위상 주소지정

입력 의존 **위상**(구동 해밀토니안)으로 내용 주소지정을 얻는 복소 선형어텐션의
유도·구현·검증. 최적 위상 코드를 닫힌 형태로 유도하고, 학습된 망이 그 코드를
채널 단위로 실제 구현함을 확인한다.

전체 실험 기록과 결론은 **[RESULTS.md](RESULTS.md)** 참조.

문서: [RESULTS.md](RESULTS.md) 전체 기록 · [EXTRAPOLATION.md](EXTRAPOLATION.md) 길이 외삽 ·
[CAPACITY.md](CAPACITY.md) 용량 · [EXCITABILITY.md](EXCITABILITY.md) 흥분성 ·
[PLASTICITY.md](PLASTICITY.md) 가소성/STDP · [FIELDS_REVIEW.md](FIELDS_REVIEW.md) 라그랑지안 검토 ·
[MISALIGNMENT.md](MISALIGNMENT.md) 생물학 층위 어긋남 진단

## 핵심 결과 요약

| | Model 0 (LTI 전이) | **Model 1 (구동 해밀토니안)** |
|---|---|---|
| selective copy, R=1 | 0.366 / 0.008 | **1.000 / 1.000** |
| selective copy, R=4 | 0.928 / 0.777 | **1.000 / 1.000** |
| 바이트 LM (파라미터 통제) | 3.2016 bpb | **3.0755 bpb** |

- §4에서 유도한 최적 위상 코드(`위상차 = c·N`, `ω(blank) ≡ 0 mod 2π`)를
  Model 1과 **Mamba-3 재현 구현 양쪽이 학습**한다 (빈칸 증분 0.002 vs 데이터 1.81).
- 재귀 깊이와 전이 위상은 내용 주소지정을 두고 **교환 가능한 자원**이다.
- 위상 코드 용량 한계 **`c·N < 2π`** — 데이터 의존 RoPE 일반에 적용되는 설계 규칙.

## 모델

```
Model 0:  U   = exp(-α + iθ)                 입력 무관 (LTI)
Model 1:  U_t = exp(-α + i(θ + Δθ(h_t)))     Δθ = s ⊙ sigmoid(W_θ h + b)
```

`|U_t| = e^{-α}` 는 그대로이므로 입력이 조절하는 것은 감쇠가 아니라 **위상 속도**다.
누적 위상 `Φ_t = Σ ω_m` 을 쓰면 시간 거리 `(t−n)θ` 자리에 **내용 가중 거리
`Φ_t − Φ_n`** 이 들어가고, 위상 자체가 순서 카운터가 된다.

`s=0` 이면 Model 1 은 Model 0 과 수치적으로 동일(1.1e-15)하므로, 성능 차이를
게이트 학습으로 귀속할 수 있다.

## 파일

| 파일 | 내용 |
|---|---|
| `model0.py` | 최소 복소 선형어텐션. 진단량 (`dirichlet_energy`, `spectral_gain`, `euler_gain`, `self_gain`) |
| `model1.py` | 입력 의존 위상. `driven=False` 또는 `s=0` 이면 Model 0 으로 환원 |
| `mamba3_ref.py` | Mamba-3 SISO 순수 PyTorch 재현 (공식 커널은 Triton/CuteDSL) |
| `selective_copy.py` | 태스크 (고정/랜덤 위치, 가변 `n_mem`) |
| `train_selective_copy.py` | 학습 + softmax attention 기준선 |
| `train_mamba3.py` | Mamba-3 학습 + 위상 코드 측정 + 증분 프로파일 |
| `charlm.py` | 바이트 단위 언어모델링 (Python stdlib 코퍼스) |
| `check_model0.py` / `check_model1.py` | 등가성·3차 동차성·감쇠·`4c` 불변량 검증 |
| `analyze_selective_copy.py` | 합성 연산자 주소지정 분해, 순서 카운터 프로브 |
| `analyze_phase.py` | 위상 불변량 원형 통계 (집중도, 각분리, 판별 점수) |
| `analyze_membrane.py` | 확산/반응 분해, 막전위 해석 검증 |
| `diag_misalign.py` | 생물학 층위 정합성 진단 (안정 정지점·포화 부호·이산화 안정성·흔적 커널) |
| `diag_membrane_c.py` | 갈래 C 흥분성 진단 (완화 수렴·문턱하 누적·전부아니면전무·불응기) |

## 실행

```bash
python check_model0.py                                     # 구현 검증
python check_model1.py                                     # 4c 불변량 확인
python train_selective_copy.py --model model1 --R 1        # 핵심 결과
python train_mamba3.py --n-layer 1                         # Mamba-3 위상 코드
python charlm.py --model model1 --d 128 --p 64             # 언어모델링
```

## 배경 문서

`대화내역1.txt`, `대화내역2.txt`, `초기구상.txt` — 아이디어의 출발점이 된 논의와
초기 설계. 유도가 어디서 왔는지, 무엇이 틀렸다가 어떻게 고쳐졌는지가 남아 있다.

## 알려진 한계

- 단일 시드, `d=128`, 6,000스텝, 8MB 코퍼스 — 장난감 규모
- 정규화가 없어 폭 확장 시 `lr ∝ 1/√d` 스케일링이 필요
- LM 의 softmax attention 기준선이 약함 (MLP 없는 tied 단일헤드)
- 핵심 재귀식과 "RoPE trick" 은 [Mamba-3 (ICLR 2026)](https://arxiv.org/abs/2603.15569) 에 선점됨.
  자세한 관계는 RESULTS.md §6 참조
