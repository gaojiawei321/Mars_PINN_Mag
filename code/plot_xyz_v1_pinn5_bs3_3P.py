# -*- coding: utf-8 -*-
"""
Movie for PINN-A1 (pinn_model_VSE_v3_L1_500_b1_1): 3-plane slices (XY, XZ, YZ)
varying Psw OR B_IMF.

Model input: [x, y, z, Psw, B_IMF]

Panels:
 (a) XY plane, |B|   (0–20 nT)
 (b) XZ plane, B_y   (–10–10 nT)
 (c) YZ plane, B_z   (–10–10 nT)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import colors
from matplotlib.patches import Circle

# ========== 全局画图风格 ==========
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

# ========== 模型路径（PINN-A1: [x,y,z,Psw,B_IMF]）==========
MODEL_PATH = "../save/pinn_model_VSE_v3_L1_500_b1_1.pth"
STATS_PATH = "../save/train_norm_stats_L1_b1_1.npz"   # 前5维是 [x,y,z,Psw,B_IMF]

# ========== 动画 / 切片基础配置 ==========
AX_MIN, AX_MAX, N = -3.0, 3.0, 120
PLANET_R = 1.0

OUT_GIF = "../fig/pinnA1_3planes_movie.gif"
OUT_MP4 = "../fig/pinnA1_3planes_movie.mp4"

# ========== 选择随 Psw 或 B_IMF 变化 ==========
# MODE = "Psw"  : 扫描 Psw, 固定 B_IMF
# MODE = "Bimf" : 扫描 B_IMF, 固定 Psw
MODE = "Psw"          # <<<<<< 在这里切换 "Psw" / "Bimf"

# 随 Psw 变化（单位 nPa），B_IMF 固定：
PSW_VALUES  = np.linspace(0.2, 2.0, 31)
BIMF_CONST  = 2.0     # nT

# 随 B_IMF 变化（单位 nT），Psw 固定：
BIMF_VALUES = np.linspace(1.0, 5.0, 31)
PSW_CONST   = 0.637   # nPa

# ================== Nemec 边界 + IMF 配置 ==================
USE_BS_MIX   = True   # True: BS 外用 IMF；False: 全部 PINN
NEMEC_BS  = dict(a=4.219, c=1.464, b=-0.063, gamma=0.205, delta=0.018)
NEMEC_MPB = dict(a=1.567, c=1.187, b=-0.065, gamma=0.094,  delta=0.038)
NEMEC_F   = 1.087

# ================== 每个面板配置 ==================
PANEL_CONFIGS = {
    "xy": {
        "bg": "Bmag",
        "uv": ("Bx", "By"),
        "cmap_range": (0.0, 20.0),
        "title": "(a) XY plane, |B|",
        "cbar_label": r"$|B|$ (nT)",
    },
    "xz": {
        "bg": "By",
        "uv": ("Bx", "Bz"),
        "cmap_range": (-15.0, 15.0),
        "title": "(b) XZ plane, $B_y$",
        "cbar_label": r"$B_y$ (nT)",
    },
    "yz": {
        "bg": "By",                  # 注意这里是 Bz
        "uv": ("By", "Bz"),
        "cmap_range": (-15.0, 15.0),
        "title": "(c) YZ plane, $B_y$",
        "cbar_label": r"$B_y$ (nT)",
    },
}

PLANES = ["xy", "xz", "yz"]
QUIVER_STRIDE = 5

# ========== 工具函数 ==========
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def current_psw_bimf(frame):
    """根据 MODE 和帧号，给出当前 Psw, B_IMF"""
    if MODE == "Psw":
        idx = frame % len(PSW_VALUES)
        return PSW_VALUES[idx], BIMF_CONST
    else:
        idx = frame % len(BIMF_VALUES)
        return PSW_CONST, BIMF_VALUES[idx]

# ========== 5-input PINN（PINN-A1）==========
class PINN5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PINN5().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ========== 归一化参数 [x,y,z,Psw,B_IMF] ==========
raw = np.load(STATS_PATH)
m = raw["mean"]
s = raw["std"]

if m.shape[0] >= 5:
    m5 = m[:5]
    s5 = s[:5]
else:
    m5 = np.zeros(5, dtype=np.float32)
    s5 = np.ones(5, dtype=np.float32)
    print(f"[WARN] stats shape {m.shape}, using identity normalization.")

mean = torch.tensor(m5, dtype=torch.float32, device=device)
std  = torch.tensor(s5, dtype=torch.float32, device=device).clamp_min(1e-12)

# ========== 公共 2D 网格（绘图坐标）==========
axis = np.linspace(AX_MIN, AX_MAX, N).astype(np.float32)
Xp, Yp = np.meshgrid(axis, axis)

# ========== Nemec 边界相关 ==========
def _nemec_x0(params, Psw, B, F):
    return params["c"] * (Psw**params["b"]) * (F**params["gamma"]) * (B**params["delta"])

def bow_shock_mask(xg, yg, zg, Psw, Bimf):
    """True = BS 外；False = BS 内（Nemec BS）"""
    if not USE_BS_MIX:
        return np.zeros_like(xg, dtype=bool)
    a  = NEMEC_BS["a"]
    x0 = _nemec_x0(NEMEC_BS, Psw, Bimf, NEMEC_F)
    rb2 = yg**2 + zg**2
    return (xg > x0) | (rb2 > a * (x0 - xg))

def nemec_curves_on_plane(plane, Psw, Bimf):
    """
    返回 BS/MPB 在指定平面上的曲线：
    plane='xy'/'xz'/'yz' → (bs_x,bs_y), (mpb_x,mpb_y)
    """
    xs = np.linspace(AX_MIN, AX_MAX, 2000)

    a_bs  = NEMEC_BS["a"]
    x0_bs = _nemec_x0(NEMEC_BS,  Psw, Bimf, NEMEC_F)

    a_mpb  = NEMEC_MPB["a"]
    x0_mpb = _nemec_x0(NEMEC_MPB, Psw, Bimf, NEMEC_F)

    def _branch(xs, rad2):
        ok = rad2 > 0
        if not np.any(ok):
            return np.array([]), np.array([])
        xs2 = xs[ok]
        rs2 = rad2[ok]
        return xs2, np.sqrt(rs2)

    if plane in ("xy", "xz"):
        # y^2(or z^2)= a(x0 - x)
        rad2_bs  = a_bs  * (x0_bs  - xs)
        rad2_mpb = a_mpb * (x0_mpb - xs)

        xs_bs,  y_bs  = _branch(xs, rad2_bs)
        xs_mpb, y_mpb = _branch(xs, rad2_mpb)

        return (np.r_[xs_bs,  xs_bs[::-1]],
                np.r_[y_bs,  -y_bs[::-1]]), \
               (np.r_[xs_mpb, xs_mpb[::-1]],
                np.r_[y_mpb, -y_mpb[::-1]])

    elif plane == "yz":
        # x=0 → y^2+z^2 = a(x0-0)
        R2_bs  = a_bs  * (x0_bs  - 0.0)
        R2_mpb = a_mpb * (x0_mpb - 0.0)
        th = np.linspace(0, 2*np.pi, 600)

        if R2_bs <= 0:
            y_bs = z_bs = np.array([])
        else:
            R_bs = np.sqrt(R2_bs)
            y_bs = R_bs * np.cos(th)
            z_bs = R_bs * np.sin(th)
        if R2_mpb <= 0:
            y_mpb = z_mpb = np.array([])
        else:
            R_mpb = np.sqrt(R2_mpb)
            y_mpb = R_mpb * np.cos(th)
            z_mpb = R_mpb * np.sin(th)

        return (y_bs, z_bs), (y_mpb, z_mpb)

    else:
        raise ValueError("plane must be 'xy'|'xz'|'yz'")


# ========== IMF & 预测 ==========
def imf_vector_components(Bimf):
    """简单 IMF：模长=Bimf，方向沿 +Y"""
    Bmag = float(Bimf)
    return 0.0, Bmag, 0.0

@torch.no_grad()
def predict_on_grid(xg, yg, zg, Psw, Bimf):
    """在给定三维网格上预测 B(x,y,z; Psw,B_IMF)。"""
    x_flat = xg.ravel()
    y_flat = yg.ravel()
    z_flat = zg.ravel()

    ps = np.full_like(x_flat, Psw,  dtype=np.float32)
    bi = np.full_like(x_flat, Bimf, dtype=np.float32)

    arr = np.stack([x_flat, y_flat, z_flat, ps, bi], axis=1)   # (N*N, 5)

    tin = torch.from_numpy(arr).to(device)
    tin = (tin - mean) / std
    Bout = model(tin).cpu().numpy().reshape(xg.shape[0], xg.shape[1], 3)

    if USE_BS_MIX:
        mask_bs_out = bow_shock_mask(x_flat, y_flat, z_flat, Psw, Bimf).reshape(xg.shape)
        if np.any(mask_bs_out):
            Bx_imf, By_imf, Bz_imf = imf_vector_components(Bimf)
            Bout[mask_bs_out, 0] = Bx_imf
            Bout[mask_bs_out, 1] = By_imf
            Bout[mask_bs_out, 2] = Bz_imf

    return Bout

def pick_background(B, bg_name):
    if bg_name == "Bx":
        return B[:, :, 0]
    if bg_name == "By":
        return B[:, :, 1]
    if bg_name == "Bz":
        return B[:, :, 2]
    if bg_name == "Bmag":
        return np.sqrt((B**2).sum(axis=2))
    raise ValueError("bg_name must be 'Bx','By','Bz','Bmag'")

def get_uv(B, uv_names):
    comp = {"Bx": 0, "By": 1, "Bz": 2}
    U = B[:, :, comp[uv_names[0]]]
    V = B[:, :, comp[uv_names[1]]]
    return U, V


# ========== 为每个平面准备网格 & 物理坐标 ==========
plane_data = {}

for pl in PLANES:
    if pl == "xy":
        xg = Xp.copy()
        yg = Yp.copy()
        zg = np.zeros_like(Xp)    # z = 0
        xlabel = r"$X_{\mathrm{MSE}}\ (R_{M})$"
        ylabel = r"$Y_{\mathrm{MSE}}\ (R_{M})$"
    elif pl == "xz":
        xg = Xp.copy()
        yg = np.zeros_like(Xp)    # y = 0
        zg = Yp.copy()
        xlabel = r"$X_{\mathrm{MSE}}\ (R_{M})$"
        ylabel = r"$Z_{\mathrm{MSE}}\ (R_{M})$"
    elif pl == "yz":
        xg = np.zeros_like(Xp)    # x = 0
        yg = Xp.copy()
        zg = Yp.copy()
        xlabel = r"$Y_{\mathrm{MSE}}\ (R_{M})$"
        ylabel = r"$Z_{\mathrm{MSE}}\ (R_{M})$"
    else:
        raise ValueError

    r3 = np.sqrt(xg**2 + yg**2 + zg**2)
    mask_inner = r3 < PLANET_R

    plane_data[pl] = {
        "xg": xg,
        "yg": yg,
        "zg": zg,
        "mask_inner": mask_inner,
        "xlabel": xlabel,
        "ylabel": ylabel,
    }


# ========== 初始化 figure 和 三个子图 ==========
ensure_dir(OUT_GIF)
ensure_dir(OUT_MP4)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.12, top=0.88, wspace=0.25)

ims   = {}      # 背景图像 im 对象
quivs = {}      # quiver 对象
bs_lines  = {}
mpb_lines = {}
cbars = {}

# 初始 Psw,B_IMF
P0, B0_val = current_psw_bimf(0)

for ax, pl in zip(axes, PLANES):
    cfg   = PANEL_CONFIGS[pl]
    pdata = plane_data[pl]

    xg = pdata["xg"]; yg = pdata["yg"]; zg = pdata["zg"]
    mask_inner = pdata["mask_inner"]

    # 初始帧
    B0 = predict_on_grid(xg, yg, zg, P0, B0_val)
    BG0 = pick_background(B0, cfg["bg"])
    BG0 = np.ma.masked_where(mask_inner, BG0)

    # colormap
    norm = colors.Normalize(vmin=cfg["cmap_range"][0],
                            vmax=cfg["cmap_range"][1])
    cmap = plt.cm.coolwarm

    # 背景
    im = ax.pcolormesh(
        Xp, Yp, BG0, shading="nearest",
        cmap=cmap, norm=norm, antialiased=False
    )
    ims[pl] = (im, norm, cmap)

    # quiver
    U0, V0 = get_uv(B0, cfg["uv"])
    U0 = np.ma.masked_where(mask_inner, U0)
    V0 = np.ma.masked_where(mask_inner, V0)
    step = QUIVER_STRIDE
    q = ax.quiver(
        Xp[::step, ::step], Yp[::step, ::step],
        U0[::step, ::step], V0[::step, ::step],
        color="black", scale=100, width=0.003
    )
    quivs[pl] = q

    # 火星圆盘
    circle = Circle((0, 0), radius=PLANET_R,
                    facecolor="white", edgecolor="k", lw=1.0, zorder=5)
    ax.add_patch(circle)

    # Nemec 边界
    (bs_x, bs_y), (mpb_x, mpb_y) = nemec_curves_on_plane(pl, P0, B0_val)
    l_bs,  = ax.plot(bs_x,  bs_y,  "r-", lw=1.5, alpha=0.95)
    l_mpb, = ax.plot(mpb_x, mpb_y, "m-", lw=1.3, alpha=0.9)
    bs_lines[pl]  = l_bs
    mpb_lines[pl] = l_mpb

    # 坐标
    ax.set_xlim(AX_MIN, AX_MAX)
    ax.set_ylim(AX_MIN, AX_MAX)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.set_xlabel(pdata["xlabel"])
    ax.set_ylabel(pdata["ylabel"])

    # 标题
    ax.set_title(cfg["title"], fontsize=13, fontweight="bold")

    # colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label(cfg["cbar_label"])
    cbars[pl] = cbar

# suptitle
if MODE == "Psw":
    sup = fig.suptitle(
        rf"PINN-A1  slices, varying $P_{{\mathrm{{SW}}}}$ (B$_{{\mathrm{{IMF}}}}$={B0_val:.2f} nT)",
        fontsize=14
    )
else:
    sup = fig.suptitle(
        rf"PINN-A1  slices, varying B$_{{\mathrm{{IMF}}}}$ (P$_{{\mathrm{{SW}}}}$={P0:.3f} nPa)",
        fontsize=14
    )

# ========== 动画 update ==========
def update(frame_idx):
    Psw, Bimf = current_psw_bimf(frame_idx)

    for pl, ax in zip(PLANES, axes):
        cfg   = PANEL_CONFIGS[pl]
        pdata = plane_data[pl]
        im, norm, cmap = ims[pl]
        q = quivs[pl]

        xg = pdata["xg"]; yg = pdata["yg"]; zg = pdata["zg"]
        mask_inner = pdata["mask_inner"]

        B = predict_on_grid(xg, yg, zg, Psw, Bimf)
        BG = pick_background(B, cfg["bg"])
        BG = np.ma.masked_where(mask_inner, BG)

        # 更新背景
        im.set_array(BG.ravel())

        # 更新向量
        U, V = get_uv(B, cfg["uv"])
        U = np.ma.masked_where(mask_inner, U)
        V = np.ma.masked_where(mask_inner, V)
        step = QUIVER_STRIDE
        q.set_UVC(U[::step, ::step], V[::step, ::step])

        # Nemec 边界更新
        (bs_x, bs_y), (mpb_x, mpb_y) = nemec_curves_on_plane(pl, Psw, Bimf)
        bs_lines[pl].set_data(bs_x, bs_y)
        mpb_lines[pl].set_data(mpb_x, mpb_y)

    # 更新 suptitle
    if MODE == "Psw":
        sup.set_text(
            rf"PINN-A1  slices, $P_{{\mathrm{{SW}}}}$={Psw:.3f} nPa, B$_{{\mathrm{{IMF}}}}$={Bimf:.2f} nT"
        )
    else:
        sup.set_text(
            rf"PINN-A1  slices, B$_{{\mathrm{{IMF}}}}$={Bimf:.2f} nT, P$_{{\mathrm{{SW}}}}$={Psw:.3f} nPa"
        )

    return []

# ========== 保存 GIF + MP4 ==========
if MODE == "Psw":
    n_frames = len(PSW_VALUES)
else:
    n_frames = len(BIMF_VALUES)

ani = FuncAnimation(fig, update, frames=n_frames, interval=300)

writer = FFMpegWriter(fps=3, metadata=dict(artist="Jiawei"), bitrate=2400)
ani.save(OUT_MP4, writer=writer, dpi=200)
print("Saved:", OUT_MP4)

ani.save(OUT_GIF, writer="pillow", dpi=200)
print("Saved:", OUT_GIF)
# plt.show()
