import numpy as np
import pyvista as pv

for fname in ["data/raw/r0000/snapshots/t000.vtp", "data/raw/r0000/snapshots/t002.vtp"]:
    m = pv.read(fname)
    pts = np.asarray(m.points)
    mat = np.asarray(m.cell_data["MaterialIds"]).astype(int)
    lines = m.lines.reshape(-1, 3)[:, 1:]
    # 셀(선분)의 중점 y좌표로 위/아래 판단
    mid_y = (pts[lines[:, 0], 1] + pts[lines[:, 1], 1]) / 2.0
    print(f"=== {fname} ===")
    print("point y range:", pts[:, 1].min(), pts[:, 1].max())
    print("unique MaterialIds:", sorted(set(mat.tolist())))
    for mid in sorted(set(mat.tolist())):
        ys = mid_y[mat == mid]
        print(f"  MaterialIds={mid}: n_cells={len(ys)}, y range=({ys.min():.3f}, {ys.max():.3f}), mean_y={ys.mean():.3f}")
