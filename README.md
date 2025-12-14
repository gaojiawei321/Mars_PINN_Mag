# Mars_PINN_Mag

This repository provides code and data for modeling the induced magnetic fields around Mars using Physics-Informed Neural Networks (PINNs). The model integrates MAVEN magnetic field observations with physical constraints and boundary conditions to reconstruct the three-dimensional magnetic field configuration under varying upstream solar wind conditions.

---

## 🔧 Getting Started

### Code Structure

- `/Code`: Python scripts for generating figures/movies.
- `/Fig`: Output figures and visualizations, including Movies S1–S3.

### How to Reproduce the Movies

To generate Movies S1 and S2, modify the `MODE` setting in the script `plot_xyz_v1_pinn5_bs3_3P.py`:

```python
MODE = "Psw"  # <<<<<< Switch between "Psw" and "Bimf"

```

### Movies

Movie S1. Magnetic field distribution in the induced magnetosphere under varying upstream IMF strengths from the PINN-A1 model. Magnetic field vectors are shown in the slices of the (a) 〖XY〗_MSE, (b) 〖XZ〗_MSE, and (c) 〖YZ〗_MSE planes, respectively. The P_SW is fixed at 0.64 nPa in all cases. The red and magenta lines denote the shape of the bow shock and the magnetic pileup boundary (MPB). Note that in the 〖XY〗_MSE plane, the color bar represents the magnetic field intensity, whereas in the 〖XZ〗_MSE and 〖YZ〗_MSE planes, it represents the B_y component.

Movie S2. Magnetic field distribution in the induced magnetosphere under varying upstream solar wind dynamic pressure from the PINN-A1 model. Magnetic field vectors are shown in the slices of the (a) 〖XY〗_MSE, (b) 〖XZ〗_MSE, and (c) 〖YZ〗_MSE planes, respectively. The B_IMF is fixed at 2 nT in all cases. 

Movie S3. Magnetic field distribution in the induced magnetosphere under varying IMF cone angle from the PINN-B model. Magnetic field vectors are shown in the slices of the (a) 〖XY〗_MSE, (b) 〖XZ〗_MSE, and (c) 〖YZ〗_MSE planes, respectively. 
