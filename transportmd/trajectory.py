"""
Trajetória de dinâmica molecular para o cálculo de propriedades de transporte.

Gabriel Braun, 2024.
"""

from dataclasses import dataclass
from pathlib import Path

import lammps
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.constants import Boltzmann


@dataclass
class Trajectory(lammps.PyLammps):
    """
    Trajetória de dinâmica molecular para o cálculo de propriedades de transporte.
    """

    _id: str
    _dir: Path

    # parâmetros do fluido
    substances: list[str]
    num_atoms: np.ndarray
    T: float

    # parâmetros da simulação
    L: float = 2.0e-8
    dt: float = 1.0e-15
    runtime: float = 1.0e-12
    savestep: int = 5

    seed: int = 12345

    def __lammps_init__(self):
        """
        Inicializa instância do LAMMPS para simulação.
        """
        self.log_path = self._dir.joinpath(self._id).with_suffix(".log")

        super().__init__(
            cmdargs=[
                "-screen",
                "none",
                "-log",
                str(self.log_path),
            ],
        )

    def __post_init__(self):
        """
        Inicializa as instâncias com os valores da inicialização da classe.
        """
        self.rng = np.random.default_rng(self.seed)
        self._dir.mkdir(parents=True, exist_ok=True)

        self.__lammps_init__()

    def generate_seed(self):
        """
        Retorna: SEED "filho" para uso do LAMMPS.
        """
        return self.rng.integers(1e5, 1e6)

    def _setup_initialize(self):
        """
        SETUP: Inicializa a simulação.
        """
        self.units("si")
        self.dimension(3)
        self.atom_style("atomic")
        self.neighbor(2e-10, "bin")
        self.pair_style("lj/cut", 12e-10)

    def _setup_system(self):
        """
        SETUP: Cria a caixa de simulação.
        """
        self.region("box block", 0, self.L, 0, self.L, 0, self.L)
        self.boundary("p p p")
        self.create_box(len(self.substances), "box")

        for i, substance in enumerate(self.substances):
            self.labelmap("atom", i + 1, substance)

        # self.command("labelmap atom 1 N2")
        # self.command("labelmap atom 2 O2")

        for num, substance in zip(self.num_atoms, self.substances):
            self.create_atoms(substance, "random", num, self.generate_seed(), "box")

        self.mass("N2", 2.3258671e-26)
        self.mass("O2", 2.6566962e-26)
        self.pair_coeff("N2", "N2", 71.4 * Boltzmann, 3.798e-10)
        self.pair_coeff("O2", "O2", 106.7 * Boltzmann, 3.467e-10)

    def _setup_variables(self):
        """
        SETUP: Define as variáveis e os computáveis.
        """
        self.variable("step equal step")

        # Variáveis termodinâmicas
        self.variable("V equal vol")
        self.variable("P equal press")
        self.variable("D equal density")

        # Tensor de pressão
        self.variable("pxy equal pxy")
        self.variable("pxz equal pxz")
        self.variable("pyz equal pyz")

        # Fluxo de calor
        self.compute("myKE all ke/atom")
        self.compute("myPE all pe/atom")
        self.compute("myStress all stress/atom NULL virial")
        self.compute("flux all heat/flux myKE myPE myStress")

        self.variable("Jx equal c_flux[1]/vol")
        self.variable("Jy equal c_flux[2]/vol")
        self.variable("Jz equal c_flux[3]/vol")

    def _setup_equilibrate(self):
        """
        SETUP: Equilibra o sistema.
        """
        self.timestep(self.dt)
        self.thermo(2000)
        self.thermo_style("custom step temp press density")
        self.minimize(1.0e-4, 1.0e-6, 1000, 10000)

        T_init = 1.2 * self.T

        self.velocity(
            "all create", T_init, self.generate_seed(), "mom yes rot yes dist gaussian"
        )
        self.fix("NVT all nvt temp", T_init, T_init, 100 * self.dt, "drag", 0.2)
        self.run(8000)

        self.reset_timestep(0)

    def setup(self):
        """
        Prepara o sistema para o início da simulação
        """
        self._setup_initialize()
        self._setup_system()
        self._setup_variables()
        self._setup_equilibrate()

    def run_trajectory(self):
        """
        Roda a simulação.
        """
        self.velocity(
            f"all create {self.T} {self.generate_seed()} mom yes rot yes dist gaussian"
        )
        self.fix("NVT all nvt temp", self.T, self.T, 100 * self.dt, "drag", 0.2)

        self.variable("ndens equal count(all)/vol")

        PRESSURE_TENSOR_FILE = self._dir.joinpath("pressure_tensor.dat")
        HEAT_FLUX_FILE = self._dir.joinpath("heat_flux.dat")

        self.fix(
            "pt all print",
            self.savestep,
            '"${step} ${pxy} ${pxz} ${pxz}"',
            "file",
            PRESSURE_TENSOR_FILE,
            "screen no",
        )
        self.fix(
            f'hf all print {self.savestep} "${{step}} ${{Jx}} ${{Jz}} ${{Jz}}" file {str(HEAT_FLUX_FILE)} screen no'
        )

        self.run(int(self.runtime / self.dt))

        print(self.lmp.last_thermo())

        steps = np.array(self.runs[1].thermo.Step)
        press = np.array(self.runs[1].thermo.Press)
        print(steps)
        print(press)

        print(f"Densidade: {self.variables["D"].value:.2f} kg/m3")
        print(f"Pressão: {self.variables["P"].value/101325:.2f} atm")
