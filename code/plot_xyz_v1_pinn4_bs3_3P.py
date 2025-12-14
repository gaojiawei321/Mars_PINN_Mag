# -*- coding: utf-8 -*-
"""
Movie for p43 PINN: 3-plane slices (XY, XZ, YZ) varying cone angle.

Model input: [x, y, z, CONE_ANGLE]
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

# ========== 模型路径（p43: [x,y,z,CONE_ANGLE]）==========
MODEL_PATH = "../save/pinn_model_VSE_v3_L1_500_b1_p43.pth"
STATS_PATH = "../save/train_norm_stats_L1_b1_p43.npz"   # 前4维是 [x,y,z,CONE_ANGLE]

# ========== 动画 / 切片基础配置 ==========
AX_MIN, AX_MAX, N = -3.0, 3.0, 100
PLANET_R = 1.0

OUT_GIF = "../fig/p43_3planes_cone.gif"
OUT_MP4 = "../fig/p43_3planes_cone.mp4"

# cone angle 扫描：用弧度（0–180°）
CONE_VALUES = np.linspace(0.0, np.pi, 61)

# IMF 参数（用于 BS 外 IMF 替代）
USE_BS_MIX   = True          # True：BS 外用 IMF；False：全场 PINN
B_SW_CONST   = 2.0           # IMF 模长 (nT)

# Nemec 参数（同之前）
NEMEC_BS  = dict(a=4.219, c=1.464, b=-0.063, gamma=0.205, delta=0.018)
NEMEC_MPB = dict(a=1.567, c=1.187, b=-0.065, gamma=0.094,  delta=0.038)
NEMEC_F   = 1.087
NEMEC_PSW = 0.637            # nPa

# 每个面板的配置
PANEL_CONFIGS = {
    "xy": {
        "bg": "Bmag",
        "uv": ("Bx", "By"),
        "cmap_range": (0.0, 20.0),
        "title": "(a) XY plane, |B| ",
    },
    "xz": {
        "bg": "By",
        "uv": ("Bx", "Bz"),
        "cmap_range": (-15.0, 15.0),
        "title": "(b) XZ plane, $B_y$ ",
    },
    "yz": {
        "bg": "By",
        "uv": ("By", "Bz"),
        "cmap_range": (-15.0, 15.0),
        "title": "(c) YZ plane, $B_y$ ",
    },
}

PLANES = ["xy", "xz", "yz"]
QUIVER_STRIDE = 5
STREAM_DENS   = 0.5
VECTOR_STYLE  = "quiver"     # 这里用 quiver，想改 stream 也可以


# ========== 工具函数 ==========
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# ========== 4-input PINN（p43）==========
class PINN4(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PINN4().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ========== 归一化参数 [x,y,z,CONE_ANGLE] ==========
raw = np.load(STATS_PATH)
m = raw["mean"]
s = raw["std"]

if m.shape[0] >= 4:
    m4 = m[:4]
    s4 = s[:4]
else:
    m4 = np.zeros(4, dtype=np.float32)
    s4 = np.ones(4, dtype=np.float32)
    print(f"[WARN] stats shape {m.shape}, using identity normalization.")

mean = torch.tensor(m4, dtype=torch.float32, device=device)
std  = torch.tensor(s4, dtype=torch.float32, device=device).clamp_min(1e-12)

# ========== 公共 2D 网格（在各个切面上投影）==========
axis = np.linspace(AX_MIN, AX_MAX, N).astype(np.float32)
Xp, Yp = np.meshgrid(axis, axis)   # 这一对只是“绘图平面坐标”，物理坐标随平面变


# ========== Nemec 边界相关 ==========
def _nemec_x0(params, Psw, B, F):
    c = params["c"]
    b = params["b"]
    g = params["gamma"]
    d = params["delta"]
    return c * (Psw**b) * (F**g) * (B**d)


def _continuous_branch(xs, rad2):
    ok = rad2 >= 0
    if not np.any(ok):
        return np.array([]), np.array([])
    idx = np.where(ok)[0]
    splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    seg = max(splits, key=len)
    i0, i1 = seg[0], seg[-1]
    xs_seg = xs[i0:i1+1]
    r2_seg = rad2[i0:i1+1].copy()
    r2_seg[0]  = max(r2_seg[0],  0.0)
    r2_seg[-1] = max(r2_seg[-1], 0.0)
    return xs_seg, np.sqrt(r2_seg)


def nemec_curves_on_plane(plane):
    """返回 BS/MPB 在指定平面上的曲线：
       plane='xy'/'xz'/'yz' → (bs_x,bs_y), (mpb_x,mpb_y)
       其中 'yz' 是圆周：x=0 截面
    """
    xs = np.linspace(AX_MIN, AX_MAX, 2000)

    # --- BS ---
    a_bs = NEMEC_BS["a"]
    x0_bs = _nemec_x0(NEMEC_BS, Psw=NEMEC_PSW, B=B_SW_CONST, F=NEMEC_F)

    # --- MPB ---
    a_mpb = NEMEC_MPB["a"]
    x0_mpb = _nemec_x0(NEMEC_MPB, Psw=NEMEC_PSW, B=B_SW_CONST, F=NEMEC_F)

    if plane in ("xy", "xz"):
        # z (或 y) = 0 截面：y^2(or z^2)= a(x0 - x)
        rad2_bs  = a_bs  * (x0_bs  - xs)
        rad2_mpb = a_mpb * (x0_mpb - xs)
        xs_bs,  y_bs  = _continuous_branch(xs, rad2_bs)
        xs_mpb, y_mpb = _continuous_branch(xs, rad2_mpb)
        # 返回两条对称分支需要合成一下：上 + 下
        return (np.r_[xs_bs,  xs_bs[::-1]],
                np.r_[y_bs,  -y_bs[::-1]]), \
               (np.r_[xs_mpb, xs_mpb[::-1]],
                np.r_[y_mpb, -y_mpb[::-1]])

    elif plane == "yz":
        # x = 0 截面：y^2+z^2 = a(x0-0) → 圆
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


def bow_shock_mask(xg, yg, zg):
    """True = BS 外（Nemec BS 之外）；False = BS 内。"""
    a  = NEMEC_BS["a"]
    x0 = _nemec_x0(NEMEC_BS, Psw=NEMEC_PSW, B=B_SW_CONST, F=NEMEC_F)
    rb2 = yg**2 + zg**2
    outside = (xg > x0) | (rb2 > a * (x0 - xg))
    return outside


# ========== 场与 IMF ==========

def imf_vector_components(cone_angle):
    """
    IMF 方向随 cone_angle 变化（假设在 XY 平面转）：
    - cone_angle 为 IMF 与 +X 轴夹角
    - |B| = B_SW_CONST
    """
    Bmag = float(B_SW_CONST)
    theta = float(cone_angle)
    Bx = Bmag * np.cos(theta)
    By = Bmag * np.sin(theta)
    Bz = 0.0
    return Bx, By, Bz


@torch.no_grad()
def predict_on_grid(xg, yg, zg, cone_angle):
    """在给定三维网格上预测 B(x,y,z; cone_angle)。"""
    x_flat = xg.ravel()
    y_flat = yg.ravel()
    z_flat = zg.ravel()
    CA = np.full_like(x_flat, cone_angle, dtype=np.float32)
    arr = np.stack([x_flat, y_flat, z_flat, CA], axis=1)   # (N*N, 4)

    tin = torch.from_numpy(arr).to(device)
    tin = (tin - mean) / std
    Bout = model(tin).cpu().numpy().reshape(xg.shape[0], xg.shape[1], 3)

    if USE_BS_MIX:
        mask_bs_out = bow_shock_mask(xg, yg, zg)
        if np.any(mask_bs_out):
            Bx_imf, By_imf, Bz_imf = imf_vector_components(cone_angle)
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
    # 绘图平面坐标：Xp, Yp
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
cbars = {}      # colorbar 对象（新增）

cone0 = CONE_VALUES[0]

for ax, pl in zip(axes, PLANES):
    cfg = PANEL_CONFIGS[pl]
    pdata = plane_data[pl]

    xg = pdata["xg"]; yg = pdata["yg"]; zg = pdata["zg"]
    mask_inner = pdata["mask_inner"]

    # 初始帧
    B0 = predict_on_grid(xg, yg, zg, cone0)
    BG0 = pick_background(B0, cfg["bg"])
    BG0 = np.ma.masked_where(mask_inner, BG0)

    # colormap
    norm = colors.Normalize(vmin=cfg["cmap_range"][0],
                            vmax=cfg["cmap_range"][1])
    cmap = plt.cm.coolwarm

    # --- 背景贴图 ---
    im = ax.pcolormesh(
        Xp, Yp, BG0, shading="nearest",
        cmap=cmap, norm=norm, antialiased=False
    )
    ims[pl] = (im, norm, cmap)

    # --- 向量 ---
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

    # --- 星体 ---
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=1.0)

    # --- Nemec 边界 ---
    (bs_x, bs_y), (mpb_x, mpb_y) = nemec_curves_on_plane(pl)
    if bs_x.size:  ax.plot(bs_x,  bs_y,  "r-", lw=1.5)
    if mpb_x.size: ax.plot(mpb_x, mpb_y, "m-", lw=1.3)

    # 坐标范围
    ax.set_xlim(AX_MIN, AX_MAX)
    ax.set_ylim(AX_MIN, AX_MAX)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)

    # 标签
    ax.set_xlabel(pdata["xlabel"])
    ax.set_ylabel(pdata["ylabel"])

    # --- panel title ---
    ax.set_title(cfg["title"], fontsize=13, fontweight="bold")

    # --- ⭐ 在这里加 colorbar ⭐ ---
    cbar = fig.colorbar(im, ax=ax ,shrink=0.6)
    cbar.set_label(cfg["title"].split(",")[1].strip())  # 自动取分量
    cbars[pl] = cbar


# suptitle
fig.suptitle(f"PINN-B,  Cone angle = {np.rad2deg(cone0):.1f}°", fontsize=14)

# ======================================
# ========== update() 更新动画 ==========
# ======================================
def update(frame_idx):

    cone_angle = CONE_VALUES[frame_idx]

    for pl, ax in zip(PLANES, axes):
        cfg = PANEL_CONFIGS[pl]
        pdata = plane_data[pl]
        im, norm, cmap = ims[pl]
        q = quivs[pl]

        xg, yg, zg = pdata["xg"], pdata["yg"], pdata["zg"]
        mask_inner = pdata["mask_inner"]

        B = predict_on_grid(xg, yg, zg, cone_angle)
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

    # 更新总标题
    fig.suptitle(f"PINN-B,  Cone angle = {np.rad2deg(cone_angle):.1f}°", fontsize=14)

    return []

# ---------- 保存 GIF + MP4 ----------
ani = FuncAnimation(fig, update, frames=len(CONE_VALUES), interval=400)

writer = FFMpegWriter(fps=2, metadata=dict(artist="Jiawei"), bitrate=2400)
ani.save(OUT_MP4, writer=writer, dpi=200)
print("Saved:", OUT_MP4)

ani.save(OUT_GIF, writer="pillow", dpi=200)
print("Saved:", OUT_GIF)


