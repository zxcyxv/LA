"""Mamba-3(ref) 를 우리와 동일한 외삽 격자에서 잰다.
추가로 빈칸 토큰의 회전 증분과, 회전하지 않는 상태 차원 비율을 확인한다."""
import math, torch
from mamba3_ref import Mamba3Config, Mamba3Model
from selective_copy import TaskConfig, make_batch, accuracy, BLANK, DATA_OFFSET

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("ckpt_mamba3_64.pt", map_location=dev, weights_only=False)
m = Mamba3Model(ck["cfg"]).to(dev); m.load_state_dict(ck["state"]); m.eval()
LN = (32, 64, 96, 128, 192, 256)
accs = []
with torch.no_grad():
    for ln in LN:
        torch.manual_seed(1)
        t = TaskConfig(n_mem=4, l_noise=ln); x, y = make_batch(t, 256, dev)
        accs.append(accuracy(m(x), t, y)[1])
print("Mamba-3(ref)  seq_acc  (l_noise=64 에서 학습)")
print("  " + " ".join(f"{l:>7}" for l in LN))
print("  " + " ".join(f"{v:>7.3f}" for v in accs))

with torch.no_grad():
    torch.manual_seed(0)
    t = TaskConfig(n_mem=4, l_noise=64); x, _ = make_batch(t, 256, dev)
    _, phases = m(x, return_phase=True)
print("\n레이어별 회전 증분 |ΔΦ| (rad/토큰)")
print(f"{'layer':>6} {'빈칸':>9} {'데이터':>9} {'빈칸/데이터':>11}   빈칸 증분<0.01 인 각도채널")
for li, phi in enumerate(phases):
    d = phi[:, 1:] - phi[:, :-1]
    d = (d + math.pi) % (2 * math.pi) - math.pi          # 2π 감김 보정
    tok = x[:, 1:]
    b = d[tok == BLANK].abs().mean(0)
    dd = d[tok >= DATA_OFFSET].abs().mean(0)
    print(f"{li:>6} {b.mean():>9.4f} {dd.mean():>9.4f} {(b.mean()/dd.mean()):>11.4f}"
          f"   {(b < 0.01).sum().item():>3}/{b.numel()}")
lay = m.layers[0][1]
print(f"\nrope_fraction = {lay.cfg.rope_fraction}  →  d_state {lay.cfg.d_state} 중 "
      f"{int(lay.cfg.d_state*lay.cfg.rope_fraction)} 만 회전, 나머지는 주기 무한(별칭 없음)")
