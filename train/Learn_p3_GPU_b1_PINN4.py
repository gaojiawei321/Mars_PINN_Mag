import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.__file__)

import pandas as pd

import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from torch.utils.data import DataLoader, TensorDataset, random_split

# 1. 检查 GPU 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 读取 .mat 文件并预处理数据
mat_file = 'Data_mse_z4_Psw.mat'
mat_data = sio.loadmat(mat_file)
Data1 = mat_data['z_4']

#inp_idx = [0,1,2,3,4,7,8,9]     # x,y,z,  Psw, sw_b, Bx By Bz

inp_idx = [0,1,2,4,3,7,8,9]     # x,y,z,  sw_b,Psw, Bx By Bz


#out_idx = [7,8,9]
Data=Data1[:,inp_idx]

Data[:, 4] = Data1[:, 4] * np.sin((Data1[:, 6]))  # Bimf= B*sin(cone angle)

train_data = torch.tensor(Data, dtype=torch.float32).to(device)

# 数据标准化（提高训练稳定性）
mean = train_data[:, :5].mean(dim=0)
std = train_data[:, :5].std(dim=0)
train_data[:, :5] = (train_data[:, :5] - mean) / std  # 标准化输入 [x, y, z, imfv, imfb]

# 保存
STATS_PATH = "../save/train_norm_stats_L1_b1_p42.npz"
np.savez(STATS_PATH,
         mean=mean.detach().cpu().numpy().astype(np.float32),
         std=std.detach().cpu().numpy().astype(np.float32))
print(f"Saved normalization stats to {STATS_PATH}")

# 创建数据集和数据加载器
dataset = TensorDataset(train_data)
train_size = int(0.8 * len(dataset))  # 80% 用于训练
val_size = len(dataset) - train_size  # 20% 用于验证
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
batch_size = 500000  # 可根据 GPU 内存调整
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# 3. 定义改进的 PINN 模型（更深网络）
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),  # 增加神经元数量
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),  # 增加一层
            nn.Tanh(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)


# 4. 计算散度（保持不变）
#def compute_divergence(model, inputs):
#    inputs = inputs.clone().detach().requires_grad_(True)
#    B = model(inputs)
#    B_x, B_y, B_z = B[:, 0], B[:, 1], B[:, 2]

#    grad_B_x = torch.autograd.grad(B_x.sum(), inputs, create_graph=True)[0]
#    dB_x_dx = grad_B_x[:, 0]

#    grad_B_y = torch.autograd.grad(B_y.sum(), inputs, create_graph=True)[0]
#    dB_y_dy = grad_B_y[:, 1]

#    grad_B_z = torch.autograd.grad(B_z.sum(), inputs, create_graph=True)[0]
#    dB_z_dz = grad_B_z[:, 2]

#    div_B = dB_x_dx + dB_y_dy + dB_z_dz
#    return div_B

def compute_divergence(model, inputs, std_xyz=None):
    """
    inputs: 标准化后的 [N,5] = (x,y,z, imfv, imfb)
    std_xyz: (sx, sy, sz)，若提供则按链式法则折回物理坐标
    """
    xyz   = inputs[:, :3].clone().detach().requires_grad_(True)  # 只让 xyz 可导
    conds = inputs[:, 3:].clone().detach()
    net_in = torch.cat([xyz, conds], dim=1)
    B = model(net_in)

    Bx, By, Bz = B[:,0], B[:,1], B[:,2]
    gBx = torch.autograd.grad(Bx.sum(), xyz, create_graph=True)[0]  # [N,3]
    gBy = torch.autograd.grad(By.sum(), xyz, create_graph=True)[0]
    gBz = torch.autograd.grad(Bz.sum(), xyz, create_graph=True)[0]

    if std_xyz is not None:
        sx, sy, sz = std_xyz
        div = gBx[:,0]/sx + gBy[:,1]/sy + gBz[:,2]/sz
    else:
        div = gBx[:,0] + gBy[:,1] + gBz[:,2]
    return div


# === 新增：IMF fang xiang ===
def imf_vector_from_imfb(imfb_phys, direction=(0.0, 1.0, 0.0), device="cpu"):
    dir_vec = torch.tensor(direction, dtype=torch.float32, device=device)
    dir_vec = dir_vec / (torch.norm(dir_vec) + 1e-12)
    return imfb_phys.unsqueeze(1) * dir_vec  # [N,1] * [3] -> [N,3]

def imf_y_cone_dependent(sw_b_phys, cone_deg, device="cpu"):
    """只考虑 B_IMF 在 y 方向，幅值 ∝ sw_b * sin(cone)"""
    theta_rad = torch.deg2rad(cone_deg)
    mag = sw_b_phys * torch.sin(theta_rad)
    zeros = torch.zeros_like(mag)
    B = torch.stack([zeros, mag, zeros], dim=1)
    return B  # [N,3]

# === 新增：在半径 Rm 的球面上采样一些点（均匀近似）===
def sample_surface_points(N, Rm_phys=1.0, device="cpu"):
    # 均匀采样球面：phi ~ U(0,2π), cos(theta) ~ U(-1,1)
    u = torch.rand(N, device=device)
    v = torch.rand(N, device=device)
    theta = torch.acos(2*u - 1.0)
    phi = 2 * torch.pi * v
    x = Rm_phys * torch.sin(theta) * torch.cos(phi)
    y = Rm_phys * torch.sin(theta) * torch.sin(phi)
    z = Rm_phys * torch.cos(theta)
    xyz = torch.stack([x, y, z], dim=1)  # [N,3]
    n_hat = xyz / (Rm_phys + 1e-12)      # 单位法向
    return xyz, n_hat

history = {
    "epoch":        [],
    "train_loss":   [],
    "val_loss":     [],
    "data_loss":    [],
    "phys_loss":    [],
    "rel_l2_train": [],
    "rel_l2_val":   [],
    "loss_bc_x3":   [],
    "loss_bc_surf": [],
}

# 5. 训练 PINN（批量训练 + 验证 + 学习率调度）
model = PINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)  # 每 100 epoch 学习率减半

#decay_rate = 0.9
#decay_steps = 200
# 衰减函数（step 是 step 数）
#lr_lambda = lambda epoch: decay_rate ** (epoch / decay_steps)
#scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)  # 每 100 epoch 学习率减半

num_epochs = 500
lambda_phys = 1  # 调整物理损失权重（可实验 0.1、1.0 等）

best_val_loss = float('inf')
patience = 500  # 等待 50 个 epoch
patience_counter = 0

# === 新增/配置 ===
Rm_in_units = 1.0     # 你坐标单位中 1 Rm 的物理值（若本来单位就是 Rm，设为1.0）
x_bc_phys   = 3.0 * Rm_in_units
x_bc_norm   = (x_bc_phys - mean[0]) / std[0]

IMF_DIR = (0.0, 1.0, 0.0)  # IMF 方向（默认沿 +Y）；如有钟角/锥角信息可在此替换

lambda_bc_x3   = 1.0       # x=3Rm 边界的权重
lambda_bc_surf = 0.5       # 火星表面“切向”（Bn=0）权重
Nsurf_per_step = 4096      # 每个 batch 采样的球面点数量（不要太大以免显存暴涨）


for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss_total = 0

    rel_l2_train_num = 0.0  # 累加 ||B_pred - B||^2
    rel_l2_train_den = 0.0  # 累加 ||B||^2

    loss_bc_x3_total = 0
    loss_bc_surf_total = 0

    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch[0].to(device)
        inputs = batch[:, :4].requires_grad_(True)
        targets = batch[:, 5:]

        B_pred = model(inputs)
        loss_data = nn.MSELoss()(B_pred, targets)
        div_B = compute_divergence(model, inputs)
        loss_phys = torch.mean(div_B ** 2)

        rel_l2_train_num += torch.norm(B_pred - targets).pow(2).item()
        rel_l2_train_den += torch.norm(targets).pow(2).item()


        # ========== 新增：边界 1（x=3Rm：B = B_IMF） ==========
        bc_inputs = inputs.detach().clone()
        bc_inputs[:, 0] = x_bc_norm  # 仅把 x 改成 3Rm，其它 y,z,imfv,imfb 不变
        B_bc_pred = model(bc_inputs)  # [N,3]

#        sw_b_phys = bc_inputs[:, 3] * std[3] + mean[3]  # sw_b 是第 4 列
#        cone_deg = batch[:, 5]  # 原始 cone 值不标准化（注意不是 bc_inputs
#        B_imf = imf_y_cone_dependent(sw_b_phys, cone_deg, device=device)  # 只在 y 方向
#        loss_bc_x3 = nn.MSELoss()(B_bc_pred, B_imf)

        # 还原 imfb 到物理量级
#        imfb_phys = bc_inputs[:, 4] * std[4] + mean[4]
#        imfb_phys = batch[:, 4]  * std[4] + mean[4]
        imfb_phys = batch[:, 3]  * std[3] + mean[3]   # For P42 IMF

        B_imf = imf_vector_from_imfb(imfb_phys, direction=IMF_DIR, device=device)  # [N,3]
        loss_bc_x3 = nn.MSELoss()(B_bc_pred, B_imf)

        # ========== 新增：边界 2（火星表面：Bn=0） ==========
        Ns = min(Nsurf_per_step, inputs.shape[0])  # 为了匹配条件参数个数
        xyz_surf_phys, n_hat = sample_surface_points(Ns, Rm_phys=Rm_in_units, device=device)  # 物理坐标
        # 标准化回输入坐标
        xyz_surf_norm = torch.stack([
            (xyz_surf_phys[:,0] - mean[0]) / std[0],
            (xyz_surf_phys[:,1] - mean[1]) / std[1],
            (xyz_surf_phys[:,2] - mean[2]) / std[2],
        ], dim=1)  # [Ns,3]
        # 条件参数：从当前 batch 随机抽样 Ns 个（覆盖各种 imfv/imfb）
        rand_idx = torch.randint(0, inputs.shape[0], (Ns,), device=device)
        conds = inputs[rand_idx, 3:5].detach()  # [Ns,2]
        surf_inputs = torch.cat([xyz_surf_norm, conds], dim=1)  # [Ns,5]
        B_surf_pred = model(surf_inputs)  # [Ns,3]
        # 法向分量（与表面平行 => 法向为0）
        Bn = torch.sum(B_surf_pred * n_hat, dim=1)  # [Ns]
        loss_bc_surf = torch.mean(Bn**2)

#        loss = loss_data + lambda_phys * loss_phys
        loss = loss_data + lambda_phys * loss_phys + lambda_bc_x3 * loss_bc_x3 + lambda_bc_surf * loss_bc_surf
#        loss = loss_data

        loss.backward()
        optimizer.step()

        train_loss_total += loss_data.item()
        loss_bc_x3_total += loss_bc_x3.item()
        loss_bc_surf_total += loss_bc_surf.item()

    avg_train_loss = train_loss_total / len(train_loader)
    rel_l2_train = np.sqrt(rel_l2_train_num / rel_l2_train_den)

    # 验证阶段
    model.eval()
    val_loss_total = 0
    rel_l2_val_num = 0.0
    rel_l2_val_den = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch[0].to(device)
            inputs = batch[:, :4]
            targets = batch[:, 5:]
            B_pred = model(inputs)
            loss_data = nn.MSELoss()(B_pred, targets)

            rel_l2_val_num += torch.norm(B_pred - targets).pow(2).item()
            rel_l2_val_den += torch.norm(targets).pow(2).item()

            # 临时启用梯度计算
            with torch.enable_grad():
                div_B = compute_divergence(model, inputs)
                loss_phys = torch.mean(div_B ** 2)

#            div_B = compute_divergence(model, inputs)
#            loss_phys = torch.mean(div_B ** 2)

#            loss = loss_data + lambda_phys * loss_phys
            loss = loss_data

            val_loss_total += loss.item()


    avg_val_loss = val_loss_total / len(val_loader)
    rel_l2_val = np.sqrt(rel_l2_val_num / rel_l2_val_den)

    scheduler.step()  # 更新学习率


    # ---------- 记录 ----------
    history["epoch"].append(epoch)
    history["train_loss"].append(avg_train_loss)
    history["rel_l2_train"].append(rel_l2_train)

    history["val_loss"].append(avg_val_loss)
    history["rel_l2_val"].append(rel_l2_val)

    history["data_loss"].append(loss_data.item())
    history["phys_loss"].append(loss_phys.item())

    history["loss_bc_x3"].append(loss_bc_x3_total / len(train_loader))
    history["loss_bc_surf"].append(loss_bc_surf_total / len(train_loader))


    # 每 1 个 epoch 打印损失
    if epoch % 1 == 0:
#        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
#              f"Data Loss: {loss_data.item():.6f}, Phy Loss: {loss_phys.item():.6f}")
#        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
#              f"Data Loss: {loss_data.item():.6f}, Phy Loss: {loss_phys.item():.6f}, "
#              f"RelL2 Train: {rel_l2_train:.4f}, RelL2 Val: {rel_l2_val:.4f}")

        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
              f"Data Loss: {loss_data.item():.6f}, Phy Loss: {loss_phys.item():.6f}, "
              f"BC_x3: {loss_bc_x3_total / len(train_loader):.6f}, "
              f"BC_surf: {loss_bc_surf_total / len(train_loader):.6f}")

    # ... 训练和验证代码 ...
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), '../save/best_model_b1_p42.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break


pd.DataFrame(history).to_csv("../save/training_history_V3_L1_500_b1_p42.csv", index=False)
print("Loss history saved to training_history.csv")

# 保存

torch.save(model.state_dict(), '../save/pinn_model_VSE_v3_L1_500_b1_p42.pth')

a=1

