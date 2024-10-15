from lammps import lammps

args = ["-screen", "none", "-log", "tutorial/lj_fluid/lj_fluid.log"]

# create LAMMPS instance
lmp = lammps(cmdargs=args)

# get and print numerical version code
print("LAMMPS Version: ", lmp.version())

# PART A - ENERGY MINIMIZATION
# 1) Initialization
lmp.command("units lj")
lmp.command("dimension 3")
lmp.command("atom_style atomic")
lmp.command("pair_style lj/cut 2.5")
lmp.command("boundary p p p")

# 2) System definition
lmp.command("region simulation_box block -20 20 -20 20 -20 20")
lmp.command("create_box 2 simulation_box")
lmp.command("create_atoms 1 random 100 341341 simulation_box")
lmp.command("create_atoms 2 random 100 127569 simulation_box")

# 3) Simulation settings
lmp.command("mass 1 1")
lmp.command("mass 2 1")
lmp.command("pair_coeff 1 1 1.0 1.0")
lmp.command("pair_coeff 2 2 0.5 3.0")

# 4) Visualization
lmp.command("thermo 10")
lmp.command("thermo_style custom step temp pe ke etotal press")

# 5) Run
lmp.command("minimize 1.0e-4 1.0e-6 1000 10000")

# PART B - MOLECULAR DYNAMICS
lmp.command("thermo 50")

# 5) Run
lmp.command("dump D1 all atom 10 tutorial/lj_fluid/lj_fluid.lammpstrj")

lmp.command("fix mynve all nve")
lmp.command("fix mylgv all langevin 1.0 1.0 0.1 1530917")
lmp.command("timestep 0.005")
lmp.command("run 10000")

# explicitly close and delete LAMMPS instance (optional)
lmp.close()
