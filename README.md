# Mars_PINN_Mag

This repository provides code and data for modeling the induced magnetic fields around Mars using Physics-Informed Neural Networks (PINNs). The model integrates MAVEN magnetic field observations with physical constraints and boundary conditions to reconstruct the three-dimensional magnetic field configuration under varying upstream solar wind conditions.

---

## 🔧 Getting Started

### Code Structure

- `/Code`: Python scripts for generating figures/movies.
- `/fig`: Output figures and visualizations, including Movies S1–S3.
- `/save`: The PINN model and train_norm_stats, including PINN-A1 and PINN-B.

### How to Reproduce the Movies

To generate Movies S1 and S2, modify the `MODE` setting in the script `plot_xyz_v1_pinn5_bs3_3P.py`:

```python
MODE = "Psw"  # <<<<<< Switch between "Psw" and "Bimf"

```

To generate Movies S3, using To gener`plot_xyz_v1_pinn4_bs3_3P.py`:


## 🎬 Movies

### 📽️ Movie S1: IMF Strength Dependence

**Title:** Magnetic field distribution in the induced magnetosphere under varying upstream IMF strengths (PINN-A1 model)

- **Model:** PINN-A1
- **Upstream Conditions:** Varying B_IMF; P_SW fixed at 0.64 nPa
- **Planes Displayed:**
  - (a) XY_MSE
  - (b) XZ_MSE
  - (c) YZ_MSE
- **Color Bar:**
  - XY_MSE → Magnetic field intensity
  - XZ_MSE and YZ_MSE → B_y component
- **Annotations:** Red and magenta lines represent the bow shock and magnetic pileup boundary (MPB), respectively.

---

### 📽️ Movie S2: Dynamic Pressure Dependence

**Title:** Magnetic field distribution in the induced magnetosphere under varying solar wind dynamic pressure (PINN-A1 model)

- **Model:** PINN-A1
- **Upstream Conditions:** Varying P_SW; B_IMF fixed at 2 nT


### 📽️ Movie S3: Cone Angle Dependence

**Title:** Magnetic field distribution in the induced magnetosphere under varying IMF cone angles (PINN-B model)

- **Model:** PINN-B
- **Upstream Conditions:** Varying IMF cone angle
