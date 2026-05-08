from lpnets.config.constants import LPNETS
import json
import itertools


def load_grid(config_file):
    """Load experiment grid from JSON file inside config folder or absolute path."""

    if config_file is None:
        raise ValueError("config grid path not provided")

    path = LPNETS / "config" / f"{config_file}.json"

    with open(path) as f:
        return json.load(f)
    
def parse_grid(config_file, verbose=True):
    ARGS_GRID = load_grid(config_file)
    keys = list(ARGS_GRID.keys())
    combos = list(itertools.product(*ARGS_GRID.values()))

    if verbose:
        print("\nGrid:")
        for combo in combos:
            print(dict(zip(keys, combo)))

    return keys, combos

def print_config(config, params=None):
    print("\nRunning experiment with config:")

    for k, v in config.items():
        if params is not None:
            if k in params:
                print(f"{k}: {v}")
        else:
            print(f"{k}: {v}")
