import json
import pathlib

import numpy as np
import pyvista as pv

timing = json.loads(pathlib.Path("reports/evidence/W1-S1_timing.json").read_text())
rows = []
for t in timing:
    rid = t["run_id"]
    snaps = sorted(pathlib.Path(f"data/raw/{rid}/snapshots").glob("t*.vtp"))
    if not snaps:
        continue
    last = pv.read(str(snaps[-1]))
    pts = np.asarray(last.points)
    depth = float(-pts[:, 1].min())  # etl.py 관례: Si 윗면 y=0, 깊이 = -ymin
    rows.append({"run_id": rid, "depth_um": round(depth, 3), **t})

depths = [r["depth_um"] for r in rows]
walls = [r["wall_sec"] for r in rows]
print(f"n={len(rows)}")
print(f"depth(um): min={min(depths):.2f} mean={sum(depths)/len(depths):.2f} max={max(depths):.2f}")
print(f"wall_sec: min={min(walls):.1f} mean={sum(walls)/len(walls):.1f} max={max(walls):.1f}")
pathlib.Path("reports/evidence/W1-S1_depth.json").write_text(json.dumps(rows, indent=1))
