import os, json, shutil, re
import numpy as np
import time
from tqdm import trange

import jax.numpy as jnp
from jax import vmap
from functools import partial

from models.physical_class.universe import Universe
from models.ticking_class.ticking_earth import TickingEarth
from models.ticking_class.ticking_sun import TickingSun
from constants import DTYPE_ACCURACY

np.random.seed(0)

def simulation(grid_shape, nb_steps, backend="numpy"):
    iter_steps = np.zeros(nb_steps, dtype=float)
    compile_time = time.time()

    universe = Universe()
    universe.sun = TickingSun()  # Can be replaced with Sun()
    print("Running model with backend:", backend)
    print("Generating the earth...")
    universe.earth = TickingEarth(shape=grid_shape, backend=backend)
    universe.discover_everything()

    # Fills the earth with random GridChunk of water
    universe.earth.fill_with_water()

    print("Done.")
    print("Updating Universe 10 times...")
    print(universe)

    for i in trange(nb_steps):
        iter = time.time()
        universe.update_all()
        iter_steps.put(i, time.time() - iter)

    elapsed = time.time() - compile_time
    mean = np.mean(iter_steps)

    print(universe)
    print(f"Average time per step: {mean} seconds")
    print(f"Simulation took {elapsed} seconds")

    return elapsed, mean

if __name__ == "__main__":
    backend = "dace:gpu"  # Change this to the desired backend
    grid_shape = (400, 400, 80)
    nb_steps = 50

    strip_word = re.search(r"\.(\w+)'", str(DTYPE_ACCURACY))
    dtype = strip_word.group(1)

    data= {}
    elapsed, mean = simulation(grid_shape, nb_steps, backend)

    data[backend] = { "total": elapsed, "mean": mean}

    if os.path.isdir(".gt_cache"):
        shutil.rmtree(".gt_cache")
    if os.path.isdir(".dacecache"):
        shutil.rmtree(".dacecache")

    path= f"{grid_shape[0]}_{dtype}_devicesync.json"
    if not os.path.isfile(path):
        with open(path, "w") as f:
            json.dump({}, f, indent=4)

    with open(path, "r") as f:
        data_f = json.load(f)
        data_f[backend] = data[backend]

    with open(path, "w") as f:
        json.dump(data_f, f, indent=4)

    os.system('say "finished"')