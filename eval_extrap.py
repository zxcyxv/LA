"""학습 범위 밖 길이에서의 외삽 성능과 위상 잔차를 함께 잰다."""
import argparse, math, torch
from model1 import Model1, Model1Config
from selective_copy import TaskConfig, make_batch, accuracy, BLANK

ap = argparse.ArgumentParser(); ap.add_argument("ckpts", nargs="+")
a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
LN = (32, 64, 96, 128, 192, 256)
print(f"{'체크포인트':<18} " + " ".join(f"{l:>7}" for l in LN) + "    최소잔차3(rad)        생존")
for cp in a.ckpts:
    ck = torch.load(cp, map_location=dev, weights_only=False)
    o = ck["mcfg"]
    keep = ("d","p","R","vocab_size","w_scale","r_min","r_max","phase_gate")
    c = Model1Config(**{k: getattr(o, k, Model1Config.__dataclass_fields__[k].default) for k in keep})
    m = Model1(c).to(dev); m.load_state_dict(ck["state"], strict=False); m.eval()
    torch.manual_seed(1); accs = []
    with torch.no_grad():
        for ln in LN:
            t = TaskConfig(n_mem=4, l_noise=ln)
            x, y = make_batch(t, 256, dev); accs.append(accuracy(m(x), t, y)[1])
        t = TaskConfig(n_mem=4, l_noise=48); x, _ = make_batch(t, 256, dev)
        om = m.block.phase_increment(m.embed(x))
        wb = om[x == BLANK].mean(0)
        res = ((wb + math.pi) % (2 * math.pi) - math.pi).abs()
    sm = res.sort().values[:3]
    surv = ((128 * res) < (math.pi / 2)).sum().item()
    print(f"{cp.replace('ckpt_','').replace('.pt',''):<18} "
          + " ".join(f"{v:>7.3f}" for v in accs)
          + "  " + " ".join(f"{v:.4f}" for v in sm) + f"  {surv:>3}/{c.p}")
print("\n seq_acc. 학습범위: *_rand 는 l_noise 16~64, 나머지 64 고정")
print(" 최소잔차3 = |w(blank)| 이 작은 순 3개 채널")
print(" 생존 = 128칸 누적 드리프트가 pi/2 미만인 채널 수 (l_noise=128 에서 공진 유지 가능)")
