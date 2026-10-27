import time, numpy as np, torch
from torch_geometric.data import Data
from physicsnemo.models.meshgraphnet import MeshGraphNet
torch.manual_seed(0); torch.set_num_threads(4)

def load(run, k):
    g = np.load(f"data/processed/{run}/g{k:03d}.npz")
    return (torch.tensor(g["node"]), torch.tensor(g["edge"]), torch.tensor(g["edge_index"]), torch.tensor(g["pos"]))

def chamfer(a, b):
    d = torch.cdist(a, b); return d.min(1).values.mean() + d.min(0).values.mean()

for run in ("r0000", "r0001"):
    nf, ef, ei, pos = load(run, 3); _, _, _, pos_next = load(run, 4)
    print(run, "nodes", nf.shape[0], "edges", ef.shape[0], "next nodes", pos_next.shape[0])
    for name, kw in (("default(15블록,128)", {}), ("small(6블록,64)", dict(processor_size=6, hidden_dim_processor=64, hidden_dim_node_encoder=64, hidden_dim_edge_encoder=64, hidden_dim_node_decoder=64))):
        m = MeshGraphNet(input_dim_nodes=14, input_dim_edges=3, output_dim=2, **kw)
        g = Data(edge_index=ei, num_nodes=nf.shape[0])
        opt = torch.optim.Adam(m.parameters(), 1e-3)
        t = time.time(); losses = []
        for step in range(5):
            out = m(nf, ef, g)                       # (N,2) = 변위 예측
            loss = chamfer(pos + out, pos_next)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        dt = (time.time() - t) / 5
        n_par = sum(p.numel() for p in m.parameters())
        print(f"  {name}: out {tuple(out.shape)}, params {n_par/1e6:.2f}M, {dt*1000:.0f} ms/step, loss {losses[0]:.3f}->{losses[-1]:.3f}")
