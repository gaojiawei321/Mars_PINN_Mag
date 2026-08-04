# -*- coding: utf-8 -*-
# Evaluate one-orbit file (MATLAB v7.3 HDF5 or v5): Data_mse_z4_Psw.mat

import os, json
import numpy as np
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as sio

# --------------------- config ---------------------
MAT_PATH   = r"../train/Data_mse_z4_Psw.mat"            # 新的数据文件
MAT_VAR    = "z_4"                             # 视情况改成你真正的变量名
# 假设列顺序为 [x, y, z, Psw, Bimf, Bx, By, Bz, ...]
INPUT_COLS  = [0, 1, 2, 3, 4]                  # [x,y,z,Psw,Bimf]
TARGET_COLS = [7, 8, 9]                        # [Bx,By,Bz]
MODEL_PATH = r"../save/pinn_model_VSE_v3_L1_500_b1_1.pth"   # 新的 5-input 模型
OUTDIR     = "orbit_eval"
os.makedirs(OUTDIR, exist_ok=True)

# （可选）避免 OpenMP 噪声
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# --------------------- model (5-input, 3x128) ---------------------
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PINN().to(device)

state = torch.load(MODEL_PATH, map_location=device)
# 正常训练保存的是 state_dict
if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
    model.load_state_dict(state)
else:
    # 如果直接保存的是整个 nn.Module
    model = state.to(device)
model.eval()

# --------------------- load MAT ---------------------
print("Loading orbit data from:", MAT_PATH)
mat_data = sio.loadmat(MAT_PATH)

if MAT_VAR in mat_data:
    Data = mat_data[MAT_VAR]
else:
    # 如果变量名不叫 MAT_VAR，可以打印 keys 看一下
    print("Available keys:", mat_data.keys())
    raise RuntimeError(f"MAT_VAR='{MAT_VAR}' 不在文件里，请改成正确变量名。")

# 保证是 float32
Data = np.asarray(Data, dtype=np.float32)
print("Data shape:", Data.shape)

# --------------------- split features / targets ---------------------
X = torch.tensor(Data[:, INPUT_COLS], dtype=torch.float32)   # N×5
Y = torch.tensor(Data[:, TARGET_COLS], dtype=torch.float32)  # N×3

# --------------------- normalization ---------------------
# 用和训练时一模一样的 mean/std
STATS_PATH = r"../save/train_norm_stats_L1_b1_1.npz"
if os.path.exists(STATS_PATH):
    stats = np.load(STATS_PATH)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std  = torch.tensor(stats["std"],  dtype=torch.float32).clamp_min(1e-12)
    print("Loaded normalization stats from", STATS_PATH)
    if mean.numel() != X.shape[1]:
        raise RuntimeError(f"mean 维度 {mean.numel()} 与 X 特征数 {X.shape[1]} 不一致，请检查 stats 文件。")
else:
    mean = X.mean(dim=0)
    std  = X.std(dim=0).clamp_min(1e-12)
    print("WARNING: using orbit data to compute mean/std (最好用训练集统计).")

Xn = (X - mean) / std

# --------------------- inference ---------------------
def batched_predict(model, Xn, batch=200_000):
    outs = []
    with torch.no_grad():
        for i in range(0, Xn.shape[0], batch):
            xb = Xn[i:i+batch].to(device)
            outs.append(model(xb).cpu())
    return torch.cat(outs, 0)

Y_pred = batched_predict(model, Xn)

# --------------------- metrics ---------------------
def metrics(y_true, y_pred):
    y_t = y_true.numpy()
    y_p = y_pred.numpy()
    err = y_p - y_t

    mae  = np.mean(np.abs(err), axis=0).tolist()
    rmse = np.sqrt(np.mean(err**2, axis=0)).tolist()
    rmse_all = float(np.sqrt(np.mean(err**2)))

    r2 = []
    for k in range(3):
        ss_res = np.sum((y_t[:,k]-y_p[:,k])**2)
        ss_tot = np.sum((y_t[:,k]-np.mean(y_t[:,k]))**2) + 1e-12
        r2.append(1 - ss_res/ss_tot)

    mag_true = np.linalg.norm(y_t, axis=1)
    mag_pred = np.linalg.norm(y_p, axis=1)
    mag_mae  = float(np.mean(np.abs(mag_pred - mag_true)))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "RMSE_all": rmse_all,
        "R2": [float(v) for v in r2],
        "Mag_MAE": mag_mae
    }, err

m, err = metrics(Y, Y_pred)
with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump(m, f, indent=2, ensure_ascii=False)
print(json.dumps(m, indent=2, ensure_ascii=False))

# --------------------- save to MAT ---------------------
Y_pred_np = Y_pred.detach().cpu().numpy().astype(np.float64, copy=False)
X_phys_np = X.detach().cpu().numpy().astype(np.float64, copy=False)
Y_true_np = Y.detach().cpu().numpy().astype(np.float64, copy=False)

save_path_v5  = os.path.join(OUTDIR, "Y_pred_z4_Psw.mat")
save_path_v73 = os.path.join(OUTDIR, "Y_pred_z4_Psw_v73.mat")

# 如果文件很大，用 v7.3
if Y_pred_np.nbytes > int(1.9 * 1024**3):
    with h5py.File(save_path_v73, "w") as f:
        dset = f.create_dataset("Y_pred", data=Y_pred_np,
                                compression="gzip", compression_opts=4, shuffle=True)
        dset.attrs["columns"] = np.array([b"Bx_pred", b"By_pred", b"Bz_pred"], dtype="S")
        f.create_dataset("X_phys", data=X_phys_np,
                         compression="gzip", compression_opts=4, shuffle=True)
        f.create_dataset("Y_true", data=Y_true_np,
                         compression="gzip", compression_opts=4, shuffle=True)
    print("Saved MATLAB v7.3 file to:", save_path_v73)
else:
    mdict = {
        "Y_pred": Y_pred_np,
        "X_phys": X_phys_np,
        "Y_true": Y_true_np,
        "columns": np.array(["Bx_pred","By_pred","Bz_pred"], dtype=object)
    }
    sio.savemat(save_path_v5, mdict)
    print("Saved MATLAB v5/v7 file to:", save_path_v5)
