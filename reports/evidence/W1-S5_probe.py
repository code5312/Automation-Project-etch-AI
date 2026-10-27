import sys
sys.path.insert(0, "src/etchpipe/sim")
from sampling import make_recipes
from run_sim import run_sim

recipes = make_recipes(1, seed=20261027)
r = recipes[0]
print("recipe:", r)
meta = run_sim(r["params"], f"data/raw/{r['run_id']}", K=2, rays=100)
print("meta:", meta)
