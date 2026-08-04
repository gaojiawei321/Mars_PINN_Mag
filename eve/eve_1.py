import os, json, math

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ========= 可配置 =========
MODEL_PATH = "pinn_model_VSE_v4_L1_300.pth"   # 你的权重文件
MAT_PATH   = "Data_mse_z3_x10.mat"            # 数据 .mat 路径
MAT_VAR    = "z3"                              # .mat 里矩阵变量名
OUTDIR     = "eval_out"
BATCH_EVAL = 200_000                           # 推理批大小（按显存情况调整）
os.makedirs(OUTDIR, exist_ok=True)

# ========= 模型定义（与你训练一致） =========
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.net(x)

# ========= 读数据 & 标准化（与训练保持一致）=========
print("Loading data...")
mat = sio.loadmat(MAT_PATH)
Data = mat[MAT_VAR]              # 期望形状 [N, 8]：前5列输入，后3列目标
Data = torch.tensor(Data, dtype=torch.float32)

# 拆分输入/输出
X = Data[:, :5].clone()          # [x,y,z,imfv,imfb]
Y = Data[:, 5:8].clone()         # [Bx,By,Bz]

# 计算并应用与训练一致的标准化（注意：若你训练时保存了 mean/std，优先加载同一份）
mean = X.mean(dim=0)
std  = X.std(dim=0).clamp_min(1e-12)  # 防止除0
Xn   = (X - mean) / std

# ========= 加载模型 =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PINN().to(device)
ckpt = torch.load(MODEL_PATH, map_location=device)
# 兼容 state_dict 或完整模型
state = ckpt if isinstance(ckpt, dict) and all(k.startswith(("net.","0","1","2","3","4")) or "weight" in k for k in ckpt.keys()) else ckpt
if isinstance(state, dict):
    model.load_state_dict(state)
else:
    # 如果存的是 torch.save(model) 这类完整对象，直接替换
    model = state.to(device)
model.eval()
print(f"Model loaded on {device}: {MODEL_PATH}")

# ========= 推理（分批）=========
def batched_predict(model, Xn, batch=BATCH_EVAL):
    preds = []
    with torch.no_grad():
        for i in range(0, Xn.shape[0], batch):
            xb = Xn[i:i+batch].to(device)
            pb = model(xb).cpu()
            preds.append(pb)
    return torch.cat(preds, dim=0)

print("Running inference...")
Y_pred = batched_predict(model, Xn)   # [N,3]

# ========= 误差指标 =========

def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def metrics(y_true, y_pred):
    y_t = _to_numpy(y_true)        # [N,3]
    y_p = _to_numpy(y_pred)        # [N,3]
    assert y_t.shape == y_p.shape, f"shape mismatch: {y_t.shape} vs {y_p.shape}"

    err = y_p - y_t                # [N,3]

    # per-component metrics (lists of length 3)
    mse_vec  = np.mean(err**2, axis=0).tolist()
    rmse_vec = np.sqrt(np.mean(err**2, axis=0)).tolist()
    mae_vec  = np.mean(np.abs(err), axis=0).tolist()

    # overall scalar metrics (all components & samples)
    rmse_all = float(np.sqrt(np.mean(err**2)))

    # R^2 per component
    r2 = []
    for k in range(y_t.shape[1]):
        ss_res = np.sum((y_t[:,k]-y_p[:,k])**2)
        ss_tot = np.sum((y_t[:,k]-np.mean(y_t[:,k]))**2) + 1e-12
        r2.append(1 - ss_res/ss_tot)

    # vector magnitude errors
    mag_true = np.linalg.norm(y_t, axis=1)
    mag_pred = np.linalg.norm(y_p, axis=1)
    mag_mae  = float(np.mean(np.abs(mag_pred - mag_true)))
    mag_rmse = float(np.sqrt(np.mean((mag_pred - mag_true)**2)))

    out = {
        "MSE": mse_vec,
        "RMSE": rmse_vec,
        "RMSE_all_components_mean": rmse_all,
        "MAE": mae_vec,
        "R2": [float(v) for v in r2],
        "VectorMag_MAE": mag_mae,
        "VectorMag_RMSE": mag_rmse,
    }
    return out, err  # err is numpy [N,3]


m, RESID = metrics(Y, Y_pred)
with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump(m, f, indent=2)
print("Metrics:", json.dumps(m, indent=2))

# ========= 作图工具 =========
def diag_scatter(true, pred, name, unit="", lim=None):
    plt.figure(figsize=(4.2,4.2), dpi=160)
    plt.scatter(true, pred, s=1, alpha=0.3)
    if lim is None:
        lo = float(np.percentile(np.concatenate([true, pred]), 1))
        hi = float(np.percentile(np.concatenate([true, pred]), 99))
        lim = (lo, hi)
    plt.plot(lim, lim, 'r--', lw=1)
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel(f"True {name} {unit}")
    plt.ylabel(f"Pred {name} {unit}")
    plt.title(f"{name}: y_true vs y_pred")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"scatter_{name}.png"), dpi=300)
    plt.close()

def resid_hist(resid, name, unit=""):
    plt.figure(figsize=(4.4,3.3), dpi=160)
    plt.hist(resid, bins=100, alpha=0.85)
    plt.xlabel(f"Residual {name} (pred-true) {unit}")
    plt.ylabel("Count")
    plt.title(f"Residual histogram: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"resid_{name}.png"), dpi=300)
    plt.close()

# ========= 图1：逐分量 y_true vs y_pred =========
Y_np = Y.numpy(); Yp_np = Y_pred.numpy()
diag_scatter(Y_np[:,0], Yp_np[:,0], "Bx")
diag_scatter(Y_np[:,1], Yp_np[:,1], "By")
diag_scatter(Y_np[:,2], Yp_np[:,2], "Bz")

# ========= 图2：残差直方图 =========
res_np = RESID
res_np = res_np if isinstance(res_np, np.ndarray) else res_np.numpy()
res_np = np.asarray(res_np)
resid_hist(res_np[:,0], "Bx")
resid_hist(res_np[:,1], "By")
resid_hist(res_np[:,2], "Bz")

# ========= 图3：误差随 imfv 的统计（分桶均值±1σ）=========
imfv_phys = (X[:,3] * std[3] + mean[3]).numpy()
abs_err_mag = np.linalg.norm((Yp_np - Y_np), axis=1)

# 分桶
nbins = 15
q = np.quantile(imfv_phys, np.linspace(0,1,nbins+1))
idx = np.digitize(imfv_phys, q[1:-1], right=False)
means = [abs_err_mag[idx==b].mean() if np.any(idx==b) else np.nan for b in range(nbins)]
stds  = [abs_err_mag[idx==b].std()  if np.any(idx==b) else np.nan for b in range(nbins)]
centers = [(q[i]+q[i+1])/2 for i in range(nbins)]

plt.figure(figsize=(5.2,3.6), dpi=160)
plt.errorbar(centers, means, yerr=stds, fmt='o-', lw=1, ms=3, capsize=3)
plt.xlabel("IMFV (physical units)")
plt.ylabel("‖B_pred - B_true‖ (abs error)")
plt.title("Error vs IMFV (bin mean ± 1σ)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "error_vs_imfv.png"), dpi=300)
plt.close()

# ========= 可选：保存部分预测对照 =========
out_csv = os.path.join(OUTDIR, "sample_predictions.csv")
import pandas as pd
df = pd.DataFrame({
    "x": (X[:,0]*std[0]+mean[0]).numpy(),
    "y": (X[:,1]*std[1]+mean[1]).numpy(),
    "z": (X[:,2]*std[2]+mean[2]).numpy(),
    "imfv": imfv_phys,
    "imfb": (X[:,4]*std[4]+mean[4]).numpy(),
    "Bx_true": Y_np[:,0], "By_true": Y_np[:,1], "Bz_true": Y_np[:,2],
    "Bx_pred": Yp_np[:,0], "By_pred": Yp_np[:,1], "Bz_pred": Yp_np[:,2],
})
df.sample(min(100000, len(df))).to_csv(out_csv, index=False)  # 下采样保存，防止太大
print(f"Saved figures & metrics to: {OUTDIR}")
