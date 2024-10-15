from pathlib import Path

from transportmd import Trajectory

if __name__ == "__main__":
    traj = Trajectory(
        _id="run001",
        _dir=Path("test/viscosity/runs/run001"),
        substances=["N2", "O2"],
        num_atoms=[100, 400],
        runtime=100 * 1e-12,
        T=300,
    )

    traj.setup()
    traj.run_trajectory()

# # CONSTANTES
# KB = Boltzmann  # [J/K]

# # create LAMMPS instance
# lmp = lammps.lammps()

# # get and print numerical version code
# print("LAMMPS Version: ", lmp.version())

# # 1) Initialization
# lmp.command("units si")
# lmp.command("dimension 3")
# lmp.command("atom_style atomic")
# lmp.command("pair_style lj/cut 13e-10")


# lmp.command("boundary p p p")


# lmp.command("neighbor 2.0e-10 bin")


# lmp.command("variable    T     equal 200.0")  # run temperature
# lmp.command("variable    Tinit equal 250.0")  # equilibration temperature

# lmp.command("variable    V     equal vol")
# lmp.command("variable    P     equal press")
# lmp.command("variable    rho   equal density")


# lmp.command(f"variable KB equal {KB}")

# # 2) System definition
# lmp.command(f"region      simulation_box block 0 2e-8 0 2e-8 0 2e-8")
# lmp.command("create_box 2 simulation_box")

# lmp.command("labelmap atom 1 N2")
# lmp.command("labelmap atom 2 O2")

# lmp.command(f"create_atoms 1 random 100 341341 simulation_box")
# lmp.command(f"create_atoms 2 random 400 127569 simulation_box")

# # 3) Simulation settings

# lmp.command("variable  dt  equal 4.0e-15")
# lmp.command("timestep  ${dt}")

# lmp.command("thermo    2000")
# lmp.command("thermo_style custom step temp press density")

# lmp.command("minimize 1.0e-4 1.0e-6 1000 10000")

# # equilibration and thermalization
# lmp.command("velocity all create   ${Tinit} ${SEED}  mom yes rot yes dist gaussian")
# lmp.command("fix      NVT all nvt temp ${Tinit} ${Tinit} $(100*dt) drag 0.2")
# lmp.command("run 8000")

# # # viscosity calculation
# lmp.command("reset_timestep 0")

# lmp.command(f"velocity all create $T {SEED} mom yes rot yes dist gaussian")
# lmp.command("fix       NVT all nvt temp $T $T $(100*dt) drag 0.2")


# lmp.command("variable step  equal step")
# lmp.command("variable ndens equal count(all)/vol")

# # Tensor de pressão
# lmp.command("variable pxy equal pxy")
# lmp.command("variable pxz equal pxz")
# lmp.command("variable pyz equal pyz")

# # Fluxo de calor
# lmp.command("compute myKE all ke/atom")
# lmp.command("compute myPE all pe/atom")
# lmp.command("compute myStress all stress/atom NULL virial")
# lmp.command("compute flux all heat/flux myKE myPE myStress")

# lmp.command("variable Jx equal c_flux[1]/vol")
# lmp.command("variable Jy equal c_flux[2]/vol")
# lmp.command("variable Jz equal c_flux[3]/vol")


# PRESSURE_TENSOR_FILE = _dir.joinpath("pressure_tensor.dat")
# HEAT_FLUX_FILE = _dir.joinpath("heat_flux.dat")

# PRINT_STEP = 5

# lmp.command(
#     f'fix pt all print {PRINT_STEP} "${{step}} ${{pxy}} ${{pxz}} ${{pxz}}" file {str(PRESSURE_TENSOR_FILE)} screen no'
# )
# lmp.command(
#     f'fix hf all print {PRINT_STEP} "${{step}} ${{Jx}} ${{Jz}} ${{Jz}}" file {str(HEAT_FLUX_FILE)} screen no'
# )

# lmp.command(f"run {2**20}")

# PRESSURE_TENSOR_FILE = _dir.joinpath("pressure_tensor.dat")
# HEAT_FLUX_FILE = _dir.joinpath("heat_flux.dat")

# p = np.loadtxt(str(PRESSURE_TENSOR_FILE))

# t = 4 * p[:, 0]
# pxy = p[:, 1]
# pxz = p[:, 2]
# pyz = p[:, 3]


# def acf(data):
#     steps = data.shape[0]
#     lag = steps // 2

#     # Nearest size with power of 2 (for efficiency) to zero-pad the input data
#     size = 2 ** np.ceil(np.log2(2 * steps - 1)).astype("int")

#     # Compute the FFT
#     FFT = np.fft.fft(data, size)

#     # Get the power spectrum
#     PWR = FFT.conjugate() * FFT

#     # Calculate the auto-correlation from inverse FFT of the power spectrum
#     COR = np.fft.ifft(PWR)[:steps].real

#     autocorrelation = COR / np.arange(steps, 0, -1)

#     return autocorrelation[:lag]


# def green_kubo_viscosity():
#     # Calculate the ACFs
#     Pxy_acf = acf(pxy)
#     Pxz_acf = acf(pxz)
#     Pyz_acf = acf(pyz)

#     avg_acf = (Pxy_acf + Pxz_acf + Pyz_acf) / 3

#     # Integrate the average ACF to get the viscosity
#     timestep = 5 * 4e-15
#     integral = integrate.cumulative_trapezoid(y=avg_acf, dx=timestep)
#     viscosity = integral * (V / kBT)

#     return viscosity


# viscosity = green_kubo_viscosity()
# time_data = np.arange(1, len(viscosity) + 1) * (5 * 4e-15) * 1e12

# print(f"Viscosidade: {viscosity[-1]*1e6} uPa.s")


# plt.plot(time_data, viscosity * 1e6)
# plt.xlabel("Time (ps)")
# plt.ylabel("Viscosity (uPa.s)")
# plt.xlim(0, time_data[-1])

# plt.title("Green-Kubo Viscosity over Time")

# plt.savefig(_dir.joinpath("viscosity.png"))
