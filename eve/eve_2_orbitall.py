# -*- coding: utf-8 -*-
# Evaluate one-orbit file (MATLAB v7.3 HDF5): Zeve_1.MAT

import os, json
import numpy as np
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --------------------- config ---------------------
MAT_PATH   = r"Zeve_1.MAT"                     # 你的单轨道文件
MAT_VAR    = None                                                              # 若知道变量名（例如 "z3"），填名字；否则自动寻找
INPUT_COLS = [0,1,2,3,4]                        # [x,y,z,imfv,imfb] 在 Data 的列索引
TARGET_COLS= [5,6,7]                            # [Bx,By,Bz]         在 Data 的列索引
MODEL_PATH = r"pinn_model_VSE_v4_L1_300.pth"    # 训练好的模型
OUTDIR     = "orbit_eval"
os.makedirs(OUTDIR, exist_ok=True)

# （可选）避免 OpenMP 噪声
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# --------------------- model (与你训练一致) ---------------------
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5,128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
            nn.Linear(128,3)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PINN().to(device)
state = torch.load(MODEL_PATH, map_location=device)
if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
    model.load_state_dict(state)
else:
    model = state.to(device)
model.eval()

# --------------------- HDF5 loader ---------------------
def list_datasets(h5file):
    items = []
    def _v(name, obj):
        if isinstance(obj, h5py.Dataset):
            items.append((name, obj.shape, obj.dtype))
    h5file.visititems(_v)
    return items

def load_mat_v73(path, var_name=None):
    with h5py.File(path, "r") as f:
        if var_name is None:
            # 自动挑选：二维、数值型、样本数×特征数
            ds = [(n,s,d) for (n,s,d) in list_datasets(f) if (len(s)==2 and np.issubdtype(d, np.number))]
            if not ds:
                raise RuntimeError("未找到二维数值数据集，请指定 MAT_VAR 或检查文件结构。找到的对象：\n" +
                                   "\n".join([str(x) for x in list_datasets(f)]))
            # 按元素总数最大挑
            name = max(ds, key=lambda t: t[1][0]*t[1][1])[0]
        else:
            name = var_name if var_name in f else var_name.lstrip("/")

        data = np.array(f[name][()])  # HDF5 -> numpy
    # 让它变成 N×D（N>=D），必要时转置
    if data.ndim != 2:
        data = np.squeeze(data)
    if data.shape[0] < data.shape[1]:
        data = data.T
    return data

print("Loading orbit data from:", MAT_PATH)
Data = load_mat_v73(MAT_PATH, MAT_VAR).astype(np.float32)  # N×D
print("Data shape:", Data.shape)

# --------------------- split features / targets ---------------------
X = torch.tensor(Data[:, INPUT_COLS], dtype=torch.float32)
Y = torch.tensor(Data[:, TARGET_COLS], dtype=torch.float32)

# --------------------- normalization (与训练一致) ---------------------
# 如果你训练时保存过 mean/std（推荐），放在 npz 里加载；否则用当前数据估计（可能有偏差）
STATS_PATH = "train_norm_stats.npz"
if os.path.exists(STATS_PATH):
    stats = np.load(STATS_PATH)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std  = torch.tensor(stats["std"],  dtype=torch.float32).clamp_min(1e-12)
    print("Loaded normalization stats from", STATS_PATH)
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
    y_t = y_true.numpy(); y_p = Y_pred.numpy()
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
    return {"MAE": mae, "RMSE": rmse, "RMSE_all": rmse_all,
            "R2": [float(v) for v in r2], "Mag_MAE": mag_mae}, err

m, err = metrics(Y, Y_pred)
with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump(m, f, indent=2, ensure_ascii=False)
print(json.dumps(m, indent=2, ensure_ascii=False))

# 确保是 numpy，并放到 double（MATLAB 默认 double）
Y_pred_np = Y_pred.detach().cpu().numpy() if hasattr(Y_pred, "detach") else np.asarray(Y_pred)
Y_pred_np = Y_pred_np.astype(np.float64, copy=False)

# 可选：一并保存真值和物理坐标，便于在 MATLAB 里对比
X_phys_np = ((X * std) + mean).detach().cpu().numpy().astype(np.float64, copy=False) if hasattr(X, "detach") else None
Y_true_np = Y.detach().cpu().numpy().astype(np.float64, copy=False) if hasattr(Y, "detach") else None

import scipy.io as sio

save_path_v5  = os.path.join(OUTDIR, "Y_pred.mat")
save_path_v73 = os.path.join(OUTDIR, "Y_pred_v73.mat")

# 如果文件很大（>~1.9GB），写 v7.3（需要 pip install h5py）
if Y_pred_np.nbytes > int(1.9 * 1024**3):
    import h5py
    with h5py.File(save_path_v73, "w") as f:
        dset = f.create_dataset("Y_pred", data=Y_pred_np, compression="gzip", compression_opts=4, shuffle=True)
        # 附带列名
        dset.attrs["columns"] = np.array([b"Bx_pred", b"By_pred", b"Bz_pred"], dtype="S")
        if X_phys_np is not None:
            f.create_dataset("X_phys", data=X_phys_np, compression="gzip", compression_opts=4, shuffle=True)
        if Y_true_np is not None:
            f.create_dataset("Y_true", data=Y_true_np, compression="gzip", compression_opts=4, shuffle=True)
    print(f"Saved MATLAB v7.3 file to: {save_path_v73}")
else:
    mdict = {"Y_pred": Y_pred_np,
             "columns": np.array(["Bx_pred","By_pred","Bz_pred"], dtype=object)}
    if X_phys_np is not None: mdict["X_phys"] = X_phys_np
    if Y_true_np is not None: mdict["Y_true"] = Y_true_np
    sio.savemat(save_path_v5, mdict)
    print(f"Saved MATLAB v5/v7 file to: {save_path_v5}")
