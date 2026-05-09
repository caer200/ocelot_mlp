import numpy as np
from ase.io import read
from ase.calculators.singlepoint import SinglePointCalculator


class ParseCifMetadata:
    """
    Class to parse metadata (structure, energy, forces, stress) from a CIF file
    """
    def __init__(self, cif_path):
        self.cif_path = cif_path
        self.forces = []
        self.stress = []
        self.energy = None
        self.atoms = None
        self.lines = None

        # call methods to get metadata
        self.get_lines()
        self.parse_metadata()

    def get_lines(self):
        """
        Reads lines from a CIF file
        """
        with open(self.cif_path, "r") as f:
            self.lines = f.readlines()
        
    def parse_metadata(self):
        """
        Parses metadata (structure, energy, forces, stress) from comment lines in a CIF file
        """
        for line in self.lines:
            line = line.strip()
            if line.startswith("# Total Energy:"):
                self.energy = float(line.split(":")[1].split()[0])
            elif line.startswith("#   Atom"):
                parts = line.split(":")[1].strip().split()
                force = [float(p) for p in parts]
                self.forces.append(force)
            elif line.startswith("#   ") and len(self.forces) > 0:  # stress lines
                stress_row = [float(x) for x in line.strip("# ").split()]
                self.stress.append(stress_row)
            elif not line.startswith("#"):
                break  # Exit after metadata block

        self.forces = np.array(self.forces)
        self.stress = np.array(self.stress)
        self.atoms = read(self.cif_path)
    

if __name__ == "__main__":
    # provide the path to the CIF file
    filename = "ocelot_mlp/test.cif"

    # initialize the ParseCifMetadata class
    PCM = ParseCifMetadata(filename)
    # get atoms, energy, forces and stress
    atoms, energy, forces, stress = PCM.atoms, PCM.energy, PCM.forces, PCM.stress

    print("Energy (eV):", energy)
    print("Forces:\n", forces)
    print("Stress tensor:\n", stress)
    print("Structure:", atoms)