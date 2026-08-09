"""읽기 시점 상태 S 를 '데이터가 쓴 것' 과 '빈칸이 쓴 것' 으로 정확히 분해한다.
S 는 쓰기에 대해 선형이므로 이 분해는 근사가 아니라 등식이다."""
import torch, math
from model1 import Model1, Model1Config
from selective_copy import TaskConfig, make_batch, BLANK, MARKER

ck = torch.load("ckpt_sig_fix.pt", map_location="cpu", weights_only=False); o = ck["mcfg"]
keep = ("d","p","R","vocab_size","w_scale","r_min","r_max","phase_gate")
c = Model1Config(**{k: getattr(o,k,Model1Config.__dataclass_fields__[k].default) for k in keep})
m = Model1(c); m.load_state_dict(ck["state"], strict=False); m.eval()
blk, a = m.block, m.block.alpha

print("읽기 시점 상태 S_T 의 기여 분해   (sig_fix, l_noise=64 에서 학습)")
print(f"{'l_noise':>8} {'‖k‖ 데이터':>11} {'‖k‖ 빈칸':>10} {'‖S_데이터‖':>11} {'‖S_빈칸‖':>10} {'신호/간섭':>10}")
torch.manual_seed(0)
with torch.no_grad():
    for ln in (32, 64, 96, 128, 192, 256):
        t = TaskConfig(n_mem=4, l_noise=ln); x, _ = make_batch(t, 64)
        h = m.embed(x); q, k, v = blk.project(h)
        T = x.shape[1]
        idx = torch.arange(T)
        decay = torch.exp(-a[None, :] * (T - 1 - idx)[:, None])      # (T,p)
        w = (k * decay.to(k.dtype)) .abs().pow(2).sum(-1).sqrt()      # 읽기시점 유효 쓰기세기
        is_d = (x >= 2); is_b = (x == BLANK)
        kn = k.abs().pow(2).sum(-1).sqrt()
        Sd = w[is_d].pow(2).sum().sqrt() / 64                         # 배치평균 프로베니우스 규모
        Sb = w[is_b].pow(2).sum().sqrt() / 64
        print(f"{ln:>8} {kn[is_d].mean():>11.4f} {kn[is_b].mean():>10.4f} "
              f"{Sd:>11.4f} {Sb:>10.4f} {Sd/Sb:>10.4f}")
