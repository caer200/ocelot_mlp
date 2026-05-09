from m3gnet.models import Relaxer, Potential
import warnings
import json
import tensorflow as tf
from pymatgen.core import Lattice, Structure

for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category, module="tensorflow")


def load_model_weights(path_to_json, weights_path):
    with open(path_to_json) as f:
        model_serialized = json.load(f)

    model_relaxer = tf.keras.models.model_from_json(model_serialized, custom_objects={})
    model_relaxer.load_weights(weights_path)
    return model_relaxer



def get_relaxer(model_relaxer, optimizer="BFGS"):
    relaxer = Relaxer(potential=Potential(model_relaxer),optimizer=optimizer)
    return relaxer


def get_structure(cif_path):
    structure = Structure.from_file(cif_path)
    return structure


def relax_structure(relaxer, structure, steps=300, verbose=True):
    """
    Overkill, but lets you skip repeating args.
    """
    relax_results = relaxer.relax(structure, steps=steps, verbose=verbose)
    return relax_results


if __name__ == "__main__":

    # provide paths to the necessary files
    # TODO: Enclose in a class so we can get paths from a config file?
    cif_path = "ocelot_mlp/test.cif"
    json_path = "ocelot_mlp/m3gnet/m3gnet.json"
    weights_path = "ocelot_mlp/m3gnet/m3gnet"

    # TODO: make a class maybe?
    # load weights
    model_relaxer = load_model_weights(json_path, weights_path)
    
    # get Relaxer object
    relaxer = get_relaxer(model_relaxer)
    
    # get the structure
    structure = get_structure(cif_path)
    
    # relax the structure using the ml-force fields
    relax_results = relax_structure(relaxer, structure)

    print(relax_results["final_structure"])
    print(relax_results["trajectory"].energies[-1])
    