"""FNO 식각 SDF 대리 모델 — 불러오기와 롤아웃.
사용:
    from inference import load, rollout
    model, cfg = load('.')
    sdfs = rollout(model, cfg, sdf0, params, n_steps=20)   # sdf0: (NY, NX) µm, params: {이름: 원본 값}
"""
import json, pathlib
import numpy as np, torch
from physicsnemo.models.fno import FNO

def load(path='.', device=None):
    path = pathlib.Path(path); cfg = json.load(open(path / 'config.json'))
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = FNO(**cfg['model_kwargs'])
    f = path / 'model.safetensors'
    if f.exists():
        from safetensors.torch import load_file; state = load_file(str(f))
    else:
        state = torch.load(path / 'model.pt', map_location='cpu')
    model.load_state_dict(state); return model.to(device).eval(), cfg

def normalize_theta(params, cfg):
    out = []
    for k in cfg['input']['theta_keys']:
        r = cfg['input']['theta_range'][k]; v, lo, hi = float(params[k]), r['low'], r['high']
        if r['log']: v, lo, hi = np.log(v), np.log(lo), np.log(hi)
        out.append(2 * (v - lo) / (hi - lo) - 1)
    return np.array(out, np.float32)

@torch.no_grad()
def rollout(model, cfg, sdf0, params, n_steps):
    """t=0 SDF(µm)에서 시작해 1분씩 n_steps 예측. 반환: (n_steps+1, NY, NX) µm. 각 스텝 입력은 직전 예측."""
    dev = next(model.parameters()).device; g = cfg['grid']
    assert sdf0.shape == (g['ny'], g['nx']), f"sdf0 shape {sdf0.shape} != {(g['ny'], g['nx'])}"
    s_std, d_std = cfg['input']['sdf_std_um'], cfg['output']['delta_std_um']
    th = torch.tensor(normalize_theta(params, cfg), device=dev)[None, :, None, None].expand(1, -1, g['ny'], g['nx'])
    cur = torch.tensor(np.asarray(sdf0, np.float32), device=dev)[None, None]; out = [cur[0, 0].cpu().numpy()]
    for _ in range(n_steps):
        cur = cur + model(torch.cat([cur / s_std, th], 1)) * d_std
        out.append(cur[0, 0].cpu().numpy())
    return np.stack(out)

def grid_xy(cfg):
    g = cfg['grid']
    return np.linspace(g['x_min_um'], g['x_max_um'], g['nx']), np.linspace(g['y_bottom_um'], g['y_top_um'], g['ny'])

def phi_to_curve(phi, cfg):
    """SDF의 0 등고선(가장 긴 것)을 왼쪽→오른쪽 (N,2) µm 좌표로. 팀 골든 채점 규격과 같음."""
    from skimage.measure import find_contours
    xs, ys = grid_xy(cfg); c = max(find_contours(phi, 0.0), key=len)
    curve = np.c_[np.interp(c[:, 1], np.arange(len(xs)), xs), np.interp(c[:, 0], np.arange(len(ys)), ys)]
    return curve[::-1] if curve[0, 0] > curve[-1, 0] else curve

if __name__ == '__main__':                       # 예시 실행: 이 폴더 안에서 python inference.py
    model, cfg = load('.')
    ex = np.load('example/r0007.npz')
    params = dict(zip([str(k) for k in ex['theta_keys']], ex['theta']))
    pred = rollout(model, cfg, ex['sdf0'], params, n_steps=len(ex['sdf_true']) - 1)
    c_true, c_pred = phi_to_curve(ex['sdf_true'][-1], cfg), phi_to_curve(pred[-1], cfg)
    print('final depth true %.2f um / pred %.2f um' % (-c_true[:, 1].min(), -c_pred[:, 1].min()))
