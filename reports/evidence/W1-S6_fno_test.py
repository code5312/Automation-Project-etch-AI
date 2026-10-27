import time

import torch
from physicsnemo.models.fno import FNO

torch.manual_seed(0)
torch.set_num_threads(4)

m = FNO(
    in_channels=1,
    out_channels=1,
    decoder_layers=1,
    decoder_layer_size=32,
    dimension=2,
    latent_channels=32,
    num_fno_layers=4,
    num_fno_modes=12,
    padding=9,
)
n_par = sum(p.numel() for p in m.parameters())
print("params(M):", round(n_par / 1e6, 3))

x = torch.randn(1, 1, 128, 256)
opt = torch.optim.Adam(m.parameters(), 1e-3)

t0 = time.time()
out = m(x)
fwd_sec = time.time() - t0

loss = out.pow(2).mean()
t1 = time.time()
opt.zero_grad()
loss.backward()
opt.step()
bwd_sec = time.time() - t1

print("out shape:", tuple(out.shape))
print("forward_sec:", round(fwd_sec, 3), "backward_sec:", round(bwd_sec, 3))

# "1에폭" 근사: 같은 입력으로 10 step 반복(배치=1, 실데이터 없음 -> 단일 샘플 1에폭 근사)
t2 = time.time()
for _ in range(10):
    out = m(x)
    loss = out.pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
epoch_sec = time.time() - t2
print("10step_sec:", round(epoch_sec, 3), "per_step_ms:", round(epoch_sec / 10 * 1000, 1))
